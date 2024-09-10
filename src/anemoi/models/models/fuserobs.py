import logging
from typing import Optional

import einops
import torch
from anemoi.utils.config import DotDict
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



class AnemoiObsFuser(AnemoiModelEncProcDec):
    def __init__(
            self,
            *,
            config: DotDict,
            data_indices : IndexCollection,
            graph_data: dict 
    ) -> None:
        
        super().__init__(config=config, data_indices=data_indices, graph_data=graph_data)
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
        # Encoder data -> hidden
        # we dont want to create an encoder for Netatmo 
        # or yet, if we do, please remove "netatmo". It will create
        # automatically an extra encoder
        input_dim = self.multi_step * self.num_input_channels

        self.encoders = nn.ModuleDict()
        for in_mesh in self.graph.input_meshes:
            if "netatmo" in in_mesh: # comment out this if you want an extra encoder for the second grid.
                continue
            else:
                self.encoders[in_mesh] = instantiate(
                    config.model.encoder,
                    in_channels_src=input_dim + self.graph.get_node_emb_size(in_mesh),
                    in_channels_dst=self.graph.get_node_emb_size(self.graph.hidden_name),
                    hidden_dim=self.num_channels,
                    sub_graph=graph_data[(in_mesh, "to", self.graph.hidden_name)],
                    src_grid_size=self.graph.num_nodes[in_mesh],
                    dst_grid_size=self.graph.num_nodes[self.graph.hidden_name],
                )
        
        # skeleton, this is not the end product
        self.fuser = instantiate(
            config.model.fuser,
            num_channels = self.num_channels,
            

        )
        
    def forward(self, x: Tensor, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]

        # add data positional info (lat/lon)
        # includes all extra grids, i.e era-meps and netatmo
        # meaning all data is send to latent data space

        x_data_latent = {}
        for in_mesh in self.graph.input_meshes:
            x_data_latent[in_mesh] = torch.cat(
                (
                    einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                    self.trainable_tensors[in_mesh](getattr(self, f"latlons_{in_mesh}"), batch_size=batch_size),
                ),
                dim=-1,  # feature dimension
            )
        

        x_hidden_latent = self.trainable_tensors[self.graph.hidden_name](
            getattr(self, f"latlons_{self.graph.hidden_name}"), batch_size=batch_size
        )

        # get shard shapes
        shard_shapes_data = {name: get_shape_shards(data, 0, model_comm_group) for name, data in x_data_latent.items()}
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
        out1, x_latent_out = self._run_mapper(
            fuser, # this has to be implemented. This is cross-attention
            (x_data_latent["era"], x_data_latent["netatmo"], x_hidden_latent)
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data["era"], shard_shapes_data["netatmo"], shard_shapes_hidden),
            model_comm_group=model_comm_group,
        )
        new_x_latents.append(x_latent_out)

             
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
