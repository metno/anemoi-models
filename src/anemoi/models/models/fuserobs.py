import logging
from typing import Optional

import einops
import torch
from anemoi.utils.config import DotDict
from functools import cached_property
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.graph import AnemoiGraphSchema
from anemoi.models.layers.graph import TrainableTensor
from .encoder_processor_decoder import AnemoiModelEncProcDec

LOGGER = logging.getLogger(__name__)



class AnemoiObsFuser(nn.Module):
    def __init__(
            self,
            *,
            config: DotDict,
            data_indices : dict[str, IndexCollection],
            graph_data: dict 
    ) -> None:
        
        super().__init__()
        # Move these to config later, can also use them for asserts etc.
        self.data_mesh_name = "stretched_grid"
        self.obs_mesh_name = "netatmo"
        
        self.graph = AnemoiGraphSchema(graph_data, config)
        self.num_channels = config.model.num_channels
        #TODO implement a separate num_channels for fuser / anemoi decoder - 1024 will probably be overkill
        self.multi_step = config.training.multistep_input

        self._calculate_shapes_and_indices(data_indices) #TODO:implement
        self._assert_matching_indices(data_indices) #TODO: implement
        self._create_trainable_attributes()

        #Register lat/lon
        for mesh_key in self.graph.mesh_names:
            self._register_latlon(mesh_key, graph_data[mesh_key]["coords"])

        input_dim = {mesh: self.multi_step * self.num_input_channels[mesh] for mesh in self.num_input_channels.keys()}

        #num_input_channels = config.model.num_channels_obs
        # Encoder data -> hidden
        # we dont want to create an encoder for Netatmo 
        # or yet, if we do, please remove "netatmo". It will create
        # automatically an extra encoder
        #input_dim_x = self.multi_step * self.num_input_channels 
        #input_dim_obs = self.multi_step * num_input_channels

        self.encoders = nn.ModuleDict()
        for in_mesh in self.graph.input_meshes:
            if self.obs_mesh_name in in_mesh: # comment out this if you want an extra encoder for the second grid.
                continue
            else:
                self.encoders[in_mesh] = instantiate(
                    config.model.encoder,
                    in_channels_src=input_dim[in_mesh] + self.graph.get_node_emb_size(in_mesh),
                    in_channels_dst=self.graph.get_node_emb_size(self.graph.hidden_name),
                    hidden_dim=self.num_channels,
                    sub_graph=graph_data[(in_mesh, "to", self.graph.hidden_name)],
                    src_grid_size=self.graph.num_nodes[in_mesh],
                    dst_grid_size=self.graph.num_nodes[self.graph.hidden_name],
                )
        
        # skeleton, this is not the end product
        self.fuser = instantiate(
            config.model.fuser,
            in_channels_src_x = self.num_channels,
            in_channels_src_obs = input_dim[self.obs_mesh_name],
            hidden_dim = self.num_channels,
            sub_graph = graph_data[(self.obs_mesh_name,"to",self.graph.hidden_name)],
            src_grid_size =  self.graph.num_nodes[self.obs_mesh_name],
            dst_grid_size = self.graph.num_nodes[self.graph.hidden_name]
        )

        # Processor hidden -> hidden
        self.processor = instantiate(
            config.model.processor,
            num_channels=self.num_channels,
            sub_graph=graph_data.get((self.graph.hidden_name, "to", self.graph.hidden_name), None),
            src_grid_size=self.graph.num_nodes[self.graph.hidden_name],
            dst_grid_size=self.graph.num_nodes[self.graph.hidden_name],
        )

        # Decoder hidden -> data
        self.decoders = nn.ModuleDict()
        for out_mesh in self.graph.output_meshes:
            self.decoders[out_mesh] = instantiate(
                config.model.decoder,
                in_channels_src=self.num_channels,
                in_channels_dst=input_dim + self.graph.get_node_emb_size(out_mesh),
                hidden_dim=self.num_channels,
                out_channels_dst=self.num_output_channels,
                sub_graph=graph_data[(self.graph.hidden_name, "to", out_mesh)],
                src_grid_size=self.graph.num_nodes[self.graph.hidden_name],
                dst_grid_size=self.graph.num_nodes[out_mesh],
            )        

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.num_input_channels = {mesh: len(data_indices[mesh].model.input) for mesh in data_indices.keys()}
        self.num_output_channels = {mesh: len(data_indices[mesh].model.output) for mesh in data_indices.keys()}
        self._internal_input_idx = {mesh: data_indices[mesh].model.input.prognostic for mesh in data_indices.keys()}
        self._internal_output_idx = {mesh: data_indices[mesh].model.output.prognostic for mesh in data_indices.keys()}

    def _assert_matching_indices(self, data_indices: dict) -> None:
        assert (
            all(len(self._internal_output_idx[mesh]) == len(data_indices[mesh].model.output.full) - 
                    len(data_indices[mesh].model.output.diagnostic) for mesh in data_indices.keys())
        )
        for mesh in data_indices.keys():
            assert (len(self._internal_output_idx[mesh]) == len(data_indices[mesh].model.output.full) - 
                    len(data_indices[mesh].model.output.diagnostic)
                    ), (
                        f"Mismatch between the internal data indices ({len(self._internal_output_idx[mesh])}) and the output indices "
                        f"excluding prognostic variables ({len(data_indices[mesh].model.output.full) - len(data_indices[mesh].model.output.diagnostic)}) "
                        f"in dataset {mesh}"
                    )
            assert (len(self._internal_input_idx[mesh]) == len(self._internal_output_idx[mesh])
                    ), f"Model indices mismatch {self._internal_input_idx[mesh]} != {len(self._internal_output_idx[mesh])} in dataset {mesh}"
    
    def _create_trainable_attributes(self) -> None:
        """Create all trainable attributes."""
        self.trainable_tensors = nn.ModuleDict()
        for mesh in self.graph.mesh_names:
            self.trainable_tensors[mesh] = TrainableTensor(
                trainable_size=self.graph.num_trainable_params[mesh], tensor_size=self.graph.num_nodes[mesh]
            )    

    def _register_latlon(self, name: str, coords: torch.Tensor) -> None:
        """Register lat/lon buffers.

        Parameters
        ----------
        name : str
            Name of grid to map
        coords: torch.Tensor
            Coordinates of the grid
        """
        self.register_buffer(
            f"latlons_{name}", torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1), persistent=True
        )


    def _run_mapper_obs(
        self,
        mapper: nn.Module,
        x: tuple[Tensor],
        obs: tuple[Tensor],
        batch_size: int,
        shard_shapes_x: tuple[tuple[int, int], tuple[int, int]],
        shard_shapes_obs:tuple[tuple[int, int], tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
        use_reentrant: bool = False,
    ) -> Tensor:
        """Run mapper with activation checkpoint.

        Parameters
        ----------
        mapper : nn.Module
            Which processor to use
        data : tuple[Tensor]
            tuple of data to pass in
        batch_size: int,
            Batch size
        shard_shapes : tuple[tuple[int, int], tuple[int, int]]
            Shard shapes for the data
        model_comm_group : ProcessGroup
            model communication group, specifies which GPUs work together
            in one model instance
        use_reentrant : bool, optional
            Use reentrant, by default False

        Returns
        -------
        TensorBatch
            Mapped data
        """
        return checkpoint(
            mapper,
            x,
            obs,
            batch_size=batch_size,
            shard_shapes_x=shard_shapes_x,
            shard_shapes_obs=shard_shapes_obs,
            model_comm_group=model_comm_group,
            use_reentrant=use_reentrant,
        )
    
    def forward(self, x: dict[str, Tensor], model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
        # for now only x as input. This is dependent on how anemoi.dataset.datamodule works
        #assert x.shape[0] == obs.shape[0]
        assert (
            all(next(iter(x).shape[0]) == x[mesh].shape[0] for mesh in x.keys())
        ), "Batch shape must match for input datasets"
        batch_size = next(iter(x)).shape[0]
        assert (
            all(next(iter(x).shape[2]) == x[mesh].shape[2] for mesh in x.keys())
        ), "Ensemble shape must match for input datasets"
        ensemble_size = next(iter(x)).shape[2]

        #TODO: Fix everything below
        input_data = [x,obs]

        # add data positional info (lat/lon)
        # includes all extra grids, i.e era-meps and netatmo
        # meaning all data is send to latent data space

        x_data_latent = {}
        for in_mesh,data in zip(self.graph.input_meshes, x):
            x_data_latent[in_mesh] = torch.cat(
                (
                    einops.rearrange(data[in_mesh], "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                    self.trainable_tensors[in_mesh](getattr(self, f"latlons_{in_mesh}"), batch_size=batch_size),
                ),
                dim=-1,  # feature dimension
            )
            print(in_mesh)
        
        x_hidden_latent = self.trainable_tensors[self.graph.hidden_name](
            getattr(self, f"latlons_{self.graph.hidden_name}"), batch_size=batch_size
        )

        # get shard shapes
        shard_shapes_data = {name: get_shape_shards(data, 0, model_comm_group) for name, data in x_data_latent.items()}
        # hmm this has to be changed, and be generalized.. for more input data
        #shard_shapes_data = {x_data_latent.keys(): get_shape_shards(x_data_latent.values(), 0 , model_comm_group)}
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)

        # Run encoders
        x_latents = []
        for in_data_name, encoder in self.encoders.items():
            x_data_latent[in_data_name], x_latent = self._run_mapper(
                encoder,
                (x_data_latent[in_data_name], x_hidden_latent),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_data[in_data_name], shard_shapes_hidden),
                model_comm_group=model_comm_group,
            )
            x_latents.append(x_latent)

        new_x_latents = []
        new_data_latent = {} # <- maybe not needed?
        #for latent in x_latent:
            #new_data_latent fuser
        # notice we cannot state era and netatmo. This has to accept n numbers of inputs
        # temp solution (below)
        out1, x_latent_out = self._run_mapper_obs(
            self.fuser, # this has to be implemented. This is cross-attention
            x= x_latents,
            obs=x_data_latent["netatmo"], #, x_hidden_latent),
            batch_size=batch_size,
            #shard_shapes_x=(shard_shapes_data["era"],shard_shapes_hidden),
            shard_shapes_obs=(shard_shapes_data["netatmo"], shard_shapes_hidden),
            model_comm_group=model_comm_group,
        )
        new_x_latents.append(x_latent_out)
        print("end")
             
        # each latent data has to be sent to
        # fuser (simple GT for cross attention)
        # the output of this fuser is sent to PROCESSOR

        # TODO: This operation can be a design choice (sum, mean, attention, ...)
        x_latent = torch.stack(new_x_latents).sum(dim=0) if len(new_x_latents) > 1 else new_x_latents[0]

        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # add skip connection (hidden -> hidden)
        x_latent_proc = x_latent_proc + x_latent

        # Run decoders
        x_out = {}
        for out_data_name, decoder in self.decoders.items():
            x_out[out_data_name] = self._run_mapper(
                decoder,
                (x_latent_proc, x_data_latent[out_data_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, shard_shapes_data[out_data_name]),
                model_comm_group=model_comm_group,
            )

            x_out[out_data_name] = (
                einops.rearrange(
                    x_out[out_data_name],
                    "(batch ensemble grid) vars -> batch ensemble grid vars",
                    batch=batch_size,
                    ensemble=ensemble_size,
                )
                .to(dtype=x.dtype)
                .clone()
            )

            if out_data_name in self.graph.input_meshes:  # check if the mesh is in the input meshes
                # residual connection (just for the prognostic variables)
                x_out[out_data_name][..., self._internal_output_idx] += x[:, -1, :, :, self._internal_input_idx]

        return x_out[self.graph.output_meshes[0]]


"""print("encoder")
        input_dim = self.multi_step * self.num_input_channels
        self.encoders = nn.ModuleDict()
        for in_mesh in self.graph.input_meshes:
            print("in_mesh", in_mesh)
            print(input_dim + self.graph.get_node_emb_size(in_mesh))
            print(self.graph.get_node_emb_size(self.graph.hidden_name))
            print(self.num_channels)
            print("sub_graph", graph_data[(in_mesh, "to", self.graph.hidden_name)])
            print("src_grid", self.graph.num_nodes[in_mesh])
            print("dst_grid", self.graph.num_nodes[self.graph.hidden_name])
        

        print("decoder")
        input_dim = self.multi_step * self.num_input_channels
        self.encoders = nn.ModuleDict()
        for out_mesh in self.graph.output_meshes:
            print("in_mesh", out_mesh)
            print(input_dim + self.graph.get_node_emb_size(out_mesh))
            #print(self.graph.get_node_emb_size(self.graph.hidden_name))
            print(self.num_output_channels)
            print(self.num_channels)
            print("sub_graph", graph_data[(self.graph.hidden_name, "to", out_mesh)])
            print("src_grid", self.graph.num_nodes[self.graph.hidden_name])
            print("dst_grid", self.graph.num_nodes[out_mesh])
        """