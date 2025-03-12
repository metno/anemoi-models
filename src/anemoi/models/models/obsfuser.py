import logging
from typing import Optional

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
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.layers.graph import NamedNodesAttributes

LOGGER = logging.getLogger(__name__)

class AnemoiObsFuser(nn.Module):
    def __init__(
            self,
            *,
            model_config: DotDict,
            data_indices : tuple,
            graph_data: tuple
    ) -> None:
        super().__init__()

        self.use_obs_fuser = model_config.model.use_obs_fuser

        self._graph_data = graph_data
        self._graph_name_hidden = model_config.graph.hidden
        self._graph_names_data = tuple(name for name in model_config.graph.input_nodes)

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)
        self.data_indices = data_indices

        self.multi_step = model_config.training.multistep_input
        self.num_channels = model_config.model.num_channels

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        input_dim = tuple(self.multi_step * self.num_input_channels[dset_idx] + self.node_attributes.attr_ndims[dset] for dset_idx, dset in enumerate(self._graph_names_data))
    
        self.encoder_data = instantiate(
            model_config.model.encoder_data,
            in_channels_src=input_dim[0],
            in_channels_dst=self.node_attributes.attr_ndims[self._graph_name_hidden],
            hidden_dim=self.num_channels,
            sub_graph=self._graph_data[(self._graph_names_data[0], "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_names_data[0]],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
        )

        if self.use_obs_fuser:
            self.encoders_obs = nn.ModuleList(
                [
                instantiate(
                    model_config.model.encoder_obs,
                    in_channels_src=input_dim[dset_idx],
                    in_channels_dst=self.num_channels,
                    hidden_dim=self.num_channels,
                    sub_graph=self._graph_data[(self._graph_names_data[dset_idx], "to", self._graph_name_hidden)],
                    src_grid_size=self.node_attributes.num_nodes[self._graph_names_data[dset_idx]],
                    dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                )
                for dset_idx, dset in enumerate(self._graph_names_data) if dset != self._graph_names_data[0]
                ]
            )

        self.processor = instantiate(
            model_config.model.processor,
            num_channels=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
        )

        self.decoder_data = instantiate(
            model_config.model.decoder_data,
            in_channels_src=self.num_channels,
            in_channels_dst=input_dim[0],
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels[0],
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_names_data[0])],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_names_data[0]]
        )

        self.decoders_obs = nn.ModuleList(
            [
            instantiate(
                model_config.model.decoder_obs,
                in_channels_src=self.num_channels,
                in_channels_dst=input_dim[dset_idx],
                hidden_dim=self.num_channels,
                out_channels_dst=self.num_output_channels[dset_idx],
                sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_names_data[dset_idx])],
                src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                dst_grid_size=self.node_attributes.num_nodes[self._graph_names_data[dset_idx]],
            )
            for dset_idx, dset in enumerate(self._graph_names_data) if dset != self._graph_names_data[0]
            ]
        )

        self.boundings = nn.ModuleList(
            [nn.ModuleList(
                [instantiate(cfg, name_to_index=self.data_indices[dset_index].internal_model.output.name_to_index)
                for cfg in dset_boundings]
                ) 
                for dset_index, dset_boundings in enumerate(getattr(model_config.model, "bounding",[]))
            ]
        )   

    def _calculate_shapes_and_indices(self, data_indices: tuple) -> None:
        self.num_input_channels = tuple(len(indices.internal_model.input) for indices in data_indices)
        self.num_output_channels = tuple(len(indices.internal_model.output) for indices in data_indices)
        self._internal_input_idx = tuple(indices.internal_model.input.prognostic for indices in data_indices)
        self._internal_output_idx = tuple(indices.internal_model.output.prognostic for indices in data_indices)
    
    def _assert_matching_indices(self, data_indices: dict) -> None:
        for dset, indices in enumerate(data_indices):
            assert len(self._internal_output_idx[dset]) == len(indices.internal_model.output.full) - len(
                indices.internal_model.output.diagnostic
            ), (
                f"Mismatch between the internal data indices ({len(self._internal_output_idx[dset])}) and "
                f"the internal output indices excluding diagnostic variables "
                f"({len(indices.internal_model.output.full) - len(indices.internal_model.output.diagnostic)}) "
                f"in dataset {dset}",
            )
            assert len(self._internal_input_idx[dset]) == len(
                self._internal_input_idx[dset]
            ), (
                f"Internal model indices must match {self._internal_input_idx} != {self._internal_output_idx} "
                f"in dataset {dset}"
            )

    def _run_mapper(
        self,
        mapper: nn.Module,
        data: tuple[Tensor],
        batch_size: int,
        shard_shapes: tuple[tuple[int, int], tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
        use_reentrant: bool = False,
    ) -> Tensor:

        return checkpoint(
            mapper,
            data,
            batch_size=batch_size,
            shard_shapes=shard_shapes,
            model_comm_group=model_comm_group,
            use_reentrant=use_reentrant,
        )

    def forward(self, x: list, model_comm_group: Optional[ProcessGroup] = None) -> Tensor:
        batch_size = x[0].shape[0]
        ensemble_size = x[0].shape[2]

        x_data_latent = torch.cat(
            (
                einops.rearrange(x[0], "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.node_attributes(self._graph_names_data[0], batch_size=batch_size),
            ),
            dim=-1,
        )

        x_obs_latent = [torch.cat(
            (
                einops.rearrange(x_elem, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.node_attributes(self._graph_names_data[dset], batch_size=batch_size),
            ),
            dim=-1, 
        ) for dset, x_elem in enumerate(x[1:], start=1)]

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)

        shard_shapes_data = get_shape_shards(x_data_latent, 0, model_comm_group)
        shard_shapes_obs = [get_shape_shards(x_data, 0, model_comm_group) for x_data in x_obs_latent]
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)

        #Data encoder
        x_data_latent, x_latent = self._run_mapper(
            self.encoder_data,
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
        )

        if self.use_obs_fuser:
            #Need a gather tensor here for x_latent
            x_latent = gather_tensor(x_latent, 0, change_channels_in_shape(shard_shapes_hidden, self.num_channels), model_comm_group)
            shard_shapes_latent = get_shape_shards(x_latent, 0, model_comm_group)
            #Obs fusers
            for dset, obs_encoder in enumerate(self.encoders_obs):
                x_obs_latent[dset], x_latent = self._run_mapper(
                    obs_encoder,
                    (x_obs_latent[dset], x_latent),
                    batch_size=batch_size,
                    shard_shapes=(shard_shapes_obs[dset], shard_shapes_latent),
                    model_comm_group=model_comm_group,
                )
        
        #Processor
        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes = shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        #Processor skip
        x_latent_proc = x_latent_proc + x_latent

        x_out = [None for _ in range(len(x))]

        #Data decoder
        x_out[0] = self._run_mapper(
            self.decoder_data,
            (x_latent_proc, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
            )

        #Obs decoder
        for dset, obs_decoder in enumerate(self.decoders_obs):
            x_out[dset+1] = self._run_mapper(
                    obs_decoder,
                    (x_latent_proc, x_obs_latent[dset]),
                    batch_size=batch_size,
                    shard_shapes=(shard_shapes_hidden, shard_shapes_obs[dset]),
                    model_comm_group=model_comm_group
                )
        for dset, x_out_elem in enumerate(x_out):
            x_out[dset] = (
                einops.rearrange(
                x_out_elem,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=x[0].dtype)
            .clone()
            )

            x_out[dset][..., self._internal_output_idx[dset]] += x[dset][:, -1, :, :, self._internal_input_idx[dset]]

            for bounding in self.boundings[dset]:
                x_out[dset] = bounding(x_out[dset])

        return list(x_out)


        

        

