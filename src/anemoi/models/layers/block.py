# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from abc import ABC
from abc import abstractmethod
from typing import Optional

import einops
import torch
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import Adj
from torch_geometric.typing import OptPairTensor
from torch_geometric.typing import Size

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.graph import sync_tensor
from anemoi.models.distributed.transformer import shard_heads
from anemoi.models.distributed.transformer import shard_sequence
from anemoi.models.layers.attention import MultiHeadSelfAttention
from anemoi.models.layers.conv import GraphConv
from anemoi.models.layers.conv import GraphTransformerConv
from anemoi.models.layers.mlp import MLP

LOGGER = logging.getLogger(__name__)


class BaseBlock(nn.Module, ABC):
    """Base class for network blocks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def forward(
        self,
        x: OptPairTensor,
        edge_attr: torch.Tensor,
        edge_index: Adj,
        shapes: tuple,
        batch_size: int,
        size: Optional[Size] = None,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class TransformerProcessorBlock(BaseBlock):
    """Transformer block with MultiHeadSelfAttention and MLPs."""

    def __init__(self, num_channels, hidden_dim, num_heads, activation, window_size: int):
        super().__init__()

        try:
            act_func = getattr(nn, activation)
        except AttributeError as ae:
            LOGGER.error("Activation function %s not supported", activation)
            raise RuntimeError from ae

        self.layer_norm1 = nn.LayerNorm(num_channels)

        self.attention = MultiHeadSelfAttention(
            num_heads=num_heads,
            embed_dim=num_channels,
            window_size=window_size,
            bias=False,
            is_causal=False,
            dropout=0.0,
        )

        self.mlp = nn.Sequential(
            nn.Linear(num_channels, hidden_dim),
            act_func(),
            nn.Linear(hidden_dim, num_channels),
        )
        self.layer_norm2 = nn.LayerNorm(num_channels)

    def forward(
        self, x: Tensor, shapes: list, batch_size: int, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:
        # Need to be out of place for gradient propagation
        x = x + self.attention(self.layer_norm1(x), shapes, batch_size, model_comm_group=model_comm_group)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class GraphConvBaseBlock(BaseBlock):
    """Message passing block with MLPs for node embeddings."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        update_src_nodes: bool = True,
        num_chunks: int = 1,
        **kwargs,
    ) -> None:
        """Initialize GNNBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        mlp_extra_layers : int, optional
            Extra layers in MLP, by default 0
        activation : str, optional
            Activation function, by default "SiLU"
        update_src_nodes: bool, by default True
            Update src if src and dst nodes are given
        num_chunks : int, by default 1
            do message passing in X chunks
        """
        super().__init__(**kwargs)

        self.update_src_nodes = update_src_nodes
        self.num_chunks = num_chunks

        self.node_mlp = MLP(
            2 * in_channels,
            out_channels,
            out_channels,
            n_extra_layers=mlp_extra_layers,
            activation=activation,
        )

        self.conv = GraphConv(
            in_channels=in_channels,
            out_channels=out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
        )

    @abstractmethod
    def forward(
        self,
        x: OptPairTensor,
        edge_attr: Tensor,
        edge_index: Adj,
        shapes: tuple,
        model_comm_group: Optional[ProcessGroup] = None,
        size: Optional[Size] = None,
    ) -> tuple[Tensor, Tensor]: ...


class GraphConvProcessorBlock(GraphConvBaseBlock):

    def __ini__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        update_src_nodes: bool = True,
        num_chunks: int = 1,
        **kwargs,
    ):
        super().__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
            update_src_nodes=update_src_nodes,
            num_chunks=num_chunks,
            **kwargs,
        )

    def forward(
        self,
        x: OptPairTensor,
        edge_attr: Tensor,
        edge_index: Adj,
        shapes: tuple,
        model_comm_group: Optional[ProcessGroup] = None,
        size: Optional[Size] = None,
    ) -> tuple[Tensor, Tensor]:

        x_in = sync_tensor(x, 0, shapes[1], model_comm_group)

        if self.num_chunks > 1:
            edge_index_list = torch.tensor_split(edge_index, self.num_chunks, dim=1)
            edge_attr_list = torch.tensor_split(edge_attr, self.num_chunks, dim=0)
            edges_out = []
            for i in range(self.num_chunks):
                out1, edges_out1 = self.conv(x_in, edge_attr_list[i], edge_index_list[i], size=size)
                edges_out.append(edges_out1)
                if i == 0:
                    out = torch.zeros_like(out1)
                out = out + out1
            edges_new = torch.cat(edges_out, dim=0)
        else:
            out, edges_new = self.conv(x_in, edge_attr, edge_index, size=size)

        out = shard_tensor(out, 0, shapes[1], model_comm_group, gather_in_backward=False)

        nodes_new = self.node_mlp(torch.cat([x, out], dim=1)) + x

        return nodes_new, edges_new


class GraphConvMapperBlock(GraphConvBaseBlock):

    def __ini__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        update_src_nodes: bool = True,
        num_chunks: int = 1,
        **kwargs,
    ):
        super().__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
            update_src_nodes=update_src_nodes,
            num_chunks=num_chunks,
            **kwargs,
        )

    def forward(
        self,
        x: OptPairTensor,
        edge_attr: Tensor,
        edge_index: Adj,
        shapes: tuple,
        model_comm_group: Optional[ProcessGroup] = None,
        size: Optional[Size] = None,
    ) -> tuple[Tensor, Tensor]:

        x_src = sync_tensor(x[0], 0, shapes[0], model_comm_group)
        x_dst = sync_tensor(x[1], 0, shapes[1], model_comm_group)
        x_in = (x_src, x_dst)

        if self.num_chunks > 1:
            edge_index_list = torch.tensor_split(edge_index, self.num_chunks, dim=1)
            edge_attr_list = torch.tensor_split(edge_attr, self.num_chunks, dim=0)
            edges_out = []
            for i in range(self.num_chunks):
                out1, edges_out1 = self.conv(x_in, edge_attr_list[i], edge_index_list[i], size=size)
                edges_out.append(edges_out1)
                if i == 0:
                    out = torch.zeros_like(out1)
                out = out + out1
            edges_new = torch.cat(edges_out, dim=0)
        else:
            out, edges_new = self.conv(x_in, edge_attr, edge_index, size=size)

        out = shard_tensor(out, 0, shapes[1], model_comm_group, gather_in_backward=False)

        nodes_new_dst = self.node_mlp(torch.cat([x[1], out], dim=1)) + x[1]

        # update only needed in forward mapper
        nodes_new_src = x[0] if not self.update_src_nodes else self.node_mlp(torch.cat([x[0], x[0]], dim=1)) + x[0]

        nodes_new = (nodes_new_src, nodes_new_dst)

        return nodes_new, edges_new


class GraphTransformerBaseBlock(BaseBlock, ABC):
    """Message passing block with MLPs for node embeddings."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        edge_dim: int,
        num_heads: int = 16,
        bias: bool = True,
        activation: str = "GELU",
        num_chunks: int = 1,
        update_src_nodes: bool = False,
        **kwargs,
    ) -> None:
        """Initialize GraphTransformerBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        edge_dim : int,
            Edge dimension
        num_heads : int,
            Number of heads
        bias : bool, by default True,
            Add bias or not
        activation : str, optional
            Activation function, by default "GELU"
        update_src_nodes: bool, by default False
            Update src if src and dst nodes are given
        """
        super().__init__(**kwargs)

        self.update_src_nodes = update_src_nodes

        self.out_channels_conv = out_channels // num_heads
        self.num_heads = num_heads

        self.num_chunks = num_chunks

        self.lin_key = nn.Linear(in_channels, num_heads * self.out_channels_conv)
        self.lin_query = nn.Linear(in_channels, num_heads * self.out_channels_conv)
        self.lin_value = nn.Linear(in_channels, num_heads * self.out_channels_conv)
        self.lin_self = nn.Linear(in_channels, num_heads * self.out_channels_conv, bias=bias)
        self.lin_edge = nn.Linear(edge_dim, num_heads * self.out_channels_conv)  # , bias=False)

        self.conv = GraphTransformerConv(out_channels=self.out_channels_conv)

        self.projection = nn.Linear(out_channels, out_channels)

        try:
            act_func = getattr(nn, activation)
        except AttributeError as ae:
            LOGGER.error("Activation function %s not supported", activation)
            raise RuntimeError from ae

        self.node_dst_mlp = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, hidden_dim),
            act_func(),
            nn.Linear(hidden_dim, out_channels),
        )

        self.layer_norm1 = nn.LayerNorm(in_channels)

        if self.update_src_nodes:
            self.node_src_mlp = nn.Sequential(
                nn.LayerNorm(out_channels),
                nn.Linear(out_channels, hidden_dim),
                act_func(),
                nn.Linear(hidden_dim, out_channels),
            )

    def shard_qkve_heads(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        edges: Tensor,
        shapes: tuple,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Shards qkv and edges along head dimension."""
        shape_src_nodes, shape_dst_nodes, shape_edges = shapes

        query, key, value, edges = (
            einops.rearrange(
                t,
                "(batch grid) (heads vars) -> batch heads grid vars",
                heads=self.num_heads,
                vars=self.out_channels_conv,
                batch=batch_size,
            )
            for t in (query, key, value, edges)
        )
        query = shard_heads(query, shapes=shape_dst_nodes, mgroup=model_comm_group)
        key = shard_heads(key, shapes=shape_src_nodes, mgroup=model_comm_group)
        value = shard_heads(value, shapes=shape_src_nodes, mgroup=model_comm_group)
        edges = shard_heads(edges, shapes=shape_edges, mgroup=model_comm_group)

        query, key, value, edges = (
            einops.rearrange(t, "batch heads grid vars -> (batch grid) heads vars") for t in (query, key, value, edges)
        )

        return query, key, value, edges

    def shard_output_seq(
        self,
        out: Tensor,
        shapes: tuple,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        """Shards Tensor sequence dimension."""
        shape_dst_nodes = shapes[1]

        out = einops.rearrange(out, "(batch grid) heads vars -> batch heads grid vars", batch=batch_size)
        out = shard_sequence(out, shapes=shape_dst_nodes, mgroup=model_comm_group)
        out = einops.rearrange(out, "batch heads grid vars -> (batch grid) (heads vars)")

        return out

    @abstractmethod
    def forward(
        self,
        x: OptPairTensor,
        edge_attr: Tensor,
        edge_index: Adj,
        shapes: tuple,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
        size: Optional[Size] = None,
    ): ...


class GraphTransformerMapperBlock(GraphTransformerBaseBlock):
    """Graph Transformer Block for node embeddings."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        edge_dim: int,
        num_heads: int = 16,
        bias: bool = True,
        activation: str = "GELU",
        num_chunks: int = 1,
        update_src_nodes: bool = False,
        **kwargs,
    ) -> None:
        """Initialize GraphTransformerBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        edge_dim : int,
            Edge dimension
        num_heads : int,
            Number of heads
        bias : bool, by default True,
            Add bias or not
        activation : str, optional
            Activation function, by default "GELU"
        update_src_nodes: bool, by default False
            Update src if src and dst nodes are given
        """
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            edge_dim=edge_dim,
            num_heads=num_heads,
            bias=bias,
            activation=activation,
            num_chunks=num_chunks,
            update_src_nodes=update_src_nodes,
            **kwargs,
        )

        self.layer_norm2 = nn.LayerNorm(in_channels)

    def forward(
        self,
        x: OptPairTensor,
        edge_attr: Tensor,
        edge_index: Adj,
        shapes: tuple,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
        size: Optional[Size] = None,
    ):
        x_skip = x

        x = (
            self.layer_norm1(x[0]),
            self.layer_norm2(x[1]),
        )  # Why does this use layer_norm2? And only is a mapper thing?
        x_r = self.lin_self(x[1])
        query = self.lin_query(x[1])
        key = self.lin_key(x[0])
        value = self.lin_value(x[0])
        edges = self.lin_edge(edge_attr)

        if model_comm_group is not None:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded across GPUs"

        query, key, value, edges = self.shard_qkve_heads(query, key, value, edges, shapes, batch_size, model_comm_group)

        # TODO: remove magic number
        num_chunks = self.num_chunks if self.training else 4  # reduce memory for inference

        if num_chunks > 1:
            edge_index_list = torch.tensor_split(edge_index, num_chunks, dim=1)
            edge_attr_list = torch.tensor_split(edges, num_chunks, dim=0)
            for i in range(num_chunks):
                out1 = self.conv(
                    query=query,
                    key=key,
                    value=value,
                    edge_attr=edge_attr_list[i],
                    edge_index=edge_index_list[i],
                    size=size,
                )
                if i == 0:
                    out = torch.zeros_like(out1)
                out = out + out1
        else:
            out = self.conv(query=query, key=key, value=value, edge_attr=edges, edge_index=edge_index, size=size)

        out = self.shard_output_seq(out, shapes, batch_size, model_comm_group)
        out = self.projection(out + x_r)

        out = out + x_skip[1]
        nodes_new_dst = self.node_dst_mlp(out) + out

        nodes_new_src = self.node_src_mlp(x_skip[0]) + x_skip[0] if self.update_src_nodes else x_skip[0]

        nodes_new = (nodes_new_src, nodes_new_dst)

        return nodes_new, edge_attr

class GraphTransformerProcessorBlock(GraphTransformerBaseBlock):
    """Graph Transformer Block for node embeddings."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        edge_dim: int,
        num_heads: int = 16,
        bias: bool = True,
        activation: str = "GELU",
        num_chunks: int = 1,
        update_src_nodes: bool = False,
        **kwargs,
    ) -> None:
        """Initialize GraphTransformerBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        edge_dim : int,
            Edge dimension
        num_heads : int,
            Number of heads
        bias : bool, by default True,
            Add bias or not
        activation : str, optional
            Activation function, by default "GELU"
        update_src_nodes: bool, by default False
            Update src if src and dst nodes are given
        """

        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            edge_dim=edge_dim,
            num_heads=num_heads,
            bias=bias,
            activation=activation,
            num_chunks=num_chunks,
            update_src_nodes=update_src_nodes,
            **kwargs,
        )

    def forward(
        self,
        x: OptPairTensor,
        edge_attr: Tensor,
        edge_index: Adj,
        shapes: tuple,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
        size: Optional[Size] = None,
    ):
        x_skip = x

        x = self.layer_norm1(x)
        x_r = self.lin_self(x)
        query = self.lin_query(x)
        key = self.lin_key(x)
        value = self.lin_value(x)

        edges = self.lin_edge(edge_attr)

        if model_comm_group is not None:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded across GPUs"

        query, key, value, edges = self.shard_qkve_heads(query, key, value, edges, shapes, batch_size, model_comm_group)

        # TODO: Is this alright?
        num_chunks = self.num_chunks if self.training else 4  # reduce memory for inference

        if num_chunks > 1:
            edge_index_list = torch.tensor_split(edge_index, num_chunks, dim=1)
            edge_attr_list = torch.tensor_split(edges, num_chunks, dim=0)
            for i in range(num_chunks):
                out1 = self.conv(
                    query=query,
                    key=key,
                    value=value,
                    edge_attr=edge_attr_list[i],
                    edge_index=edge_index_list[i],
                    size=size,
                )
                if i == 0:
                    out = torch.zeros_like(out1)
                out = out + out1
        else:
            out = self.conv(query=query, key=key, value=value, edge_attr=edges, edge_index=edge_index, size=size)

        out = self.shard_output_seq(out, shapes, batch_size, model_comm_group)
        out = self.projection(out + x_r)

        out = out + x_skip
        nodes_new = self.node_dst_mlp(out) + out

        return nodes_new, edge_attr


class GraphTransformerFuserBaseBlock(GraphTransformerBaseBlock):
    def __init__(
        self,
        in_channels_x: int,
        in_channels_obs: int,
        hidden_dim: int,
        out_channels: int,
        edge_dim_x: int,
        edge_dim_obs: int,
        num_heads: int = 16,
        bias: bool = True,
        activation: str = "GELU",
        num_chunks: int = 1,
        update_src_nodes: bool = False,
        **kwargs,
    ) -> None:
        
        super().__init__(
            in_channels_x, 
            hidden_dim, 
            out_channels, 
            edge_dim_x, 
            num_heads, 
            bias, 
            activation, 
            update_src_nodes,
            num_chunks,
            **kwargs
            )
        """
            comment: we override the whole graphtransformerbaseblock
            constructor. Not interested in shared weights. This
            block is independent. Just for resuse of class methods
        
        """
        # initialize Q,K, V funcs for x and obs respectively
        self.lin_query = nn.Linear(in_channels_x, num_heads*self.out_channels_conv)
        self.lin_key = nn.Linear(in_channels_obs, num_heads*self.out_channels_conv)
        self.lin_value = nn.Linear(in_channels_obs, num_heads*self.out_channels_conv)
        self.lin_self = nn.Linear(in_channels_x, num_heads * self.out_channels_conv, bias=bias)

        # initialize layer normalization for x and obs respectively
        self.layer_normalization_x = nn.LayerNorm(in_channels_x)
        self.layer_normalization_obs = nn.LayerNorm(in_channels_obs)
        
        # initialize linear transformation for edges 
        self.lin_edge_x = nn.Linear(edge_dim_x, num_heads * self.out_channels_conv)  # , bias=False)
        self.lin_edge_obs = nn.Linear(edge_dim_obs, num_heads * self.out_channels_conv)  # , bias=False)

        # initialize layer normalization for outputs
        self.layer_norm_output = nn.LayerNorm(out_channels) # layer normalize output after fuse

        # initialize fuse for projection layer
        self.fuse_projection_layer = nn.Linear(out_channels, out_channels)

        # initialize GraphTransformerConv (cross-attention)
        self.cross_attn = GraphTransformerConv(out_channels=out_channels)

        try:
            act_func = getattr(nn, activation)
        except AttributeError as ae:
            LOGGER.error("Activation function %s not supported", activation)
            raise RuntimeError from ae

        self.node_dst_mlp = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, hidden_dim),
            act_func(),
            nn.Linear(hidden_dim, out_channels),
        )

        if self.update_src_nodes:
            self.node_src_mlp = nn.Sequential(
                nn.LayerNorm(out_channels),
                nn.Linear(out_channels, hidden_dim),
                act_func(),
                nn.Linear(hidden_dim, out_channels),
            )
    def shard_qkve_heads_obs(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            edges_x: Tensor,
            edges_obs: Tensor,
            shapes: tuple,
            batch_size: int,
            model_comm_group: Optional[ProcessGroup] = None,
            ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Shards qkv and edges along head dimension."""
        # should it be an extra shape wrt obs and x?
        # maybe create an another baseclass to include this part?
        shape_src_nodes, shape_dst_nodes, shape_edges = shapes

        query, key, value, edges_x, edges_obs = (
            einops.rearrange(
                t,
                "(batch grid) (heads vars) -> batch heads grid vars",
                heads=self.num_heads,
                vars=self.out_channels_conv,
                batch=batch_size,
            )
            for t in (query, key, value, edges_x, edges_obs)
        )
        query = shard_heads(query, shapes=shape_dst_nodes, mgroup=model_comm_group)
        key = shard_heads(key, shapes=shape_src_nodes, mgroup=model_comm_group)
        value = shard_heads(value, shapes=shape_src_nodes, mgroup=model_comm_group)
        edges_x = shard_heads(edges_x, shapes=shape_edges, mgroup=model_comm_group)
        edges_obs = shard_heads(edges_obs, shapes=shape_edges, mgroup=model_comm_group)

        query, key, value, edges_x, edges_obs = (
            einops.rearrange(t, "batch heads grid vars -> (batch grid) heads vars") for t in (query, key, value, edges)
        )

        return query, key, value, edges_x, edges_obs

class GraphTransformerFuserBlock(GraphTransformerFuserBaseBlock):
    """Graph Transformer Fuser Block for fusing node embeddings."""

    def __init__(
        self,
        in_channels_x: int,
        in_channels_obs: int,
        hidden_dim: int,
        out_channels: int,
        edge_dim_x: int,
        edge_dim_obs: int,
        num_heads: int = 16,
        bias: bool = True,
        activation: str = "GELU",
        num_chunks: int = 1,
        update_src_nodes: bool = False,
        **kwargs,
    ) -> None:
        """Initialize GraphTransformerFuserBlock.

        Parameters
        ----------
        in_channels_x : int
            Number of input channels (first grid).
        in_channels_obs : int
            Number of input channels (observation grid).
        out_channels : int
            Number of output channels.
        edge_dim_x : int,
            Edge dimension for x
        edge_dim_obs : int,
            Edge dimension for observation
        num_heads : int,
            Number of heads
        bias : bool, by default True,
            Add bias or not
        activation : str, optional
            Activation function, by default "GELU"
        update_src_nodes: bool, by default False
            Update src if src and dst nodes are given
        """
        super().__init__(
            in_channels_x,
            in_channels_obs,
            hidden_dim,
            out_channels,
            edge_dim_x,
            edge_dim_obs,
            num_heads,
            bias,
            activation,
            num_chunks,
            update_src_nodes,
            **kwargs
        )
    
    def forward(
            self,
            x : OptPairTensor,
            obs : OptPairTensor,
            edge_attr_x : Tensor,
            edge_index_x : Adj,
            edge_attr_obs : Tensor,
            edge_index_obs: Adj,
            shapes_x: tuple, # <- should it be shapes_x and shapes_obs?
            shapes_obs: tuple,
            batch_size: int,
            model_comm_group: Optional[ProcessGroup] = None,
            size_x: Optional[Size] = None,
            size_obs: Optional[Size] = None
            ):
        x_skip = x # saving a copy, for skip connection
        #obs_skip = obs # saving a copy, for skip connection

        # combine shape and size
        size = (size_x[0] + size_obs[0], size_x[1] + size_obs[1])
        shapes = (shapes_x[0] + shapes_obs[0], shapes_x[1] + shapes_obs[1])

        # normalize
        x = self.layer_normalization_x(x)
        obs = self.layer_normalization_obs(obs)
        
        # generate feature maps for residual connection
        # is this needed?
        x_r = self.lin_self(x)
        #obs_r = self.lin_self_obs(obs)

        # Gather Q (from x input)
        query = self.lin_query(x)

        # Gather K,V (from obs input)
        key = self.lin_key(obs)
        value = self.lin_value(obs)

        edges_x = self.lin_edge_x(edge_attr_x)
        edges_obs = self.lin_edge_obs(edge_attr_obs)

        if model_comm_group is not None:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded across GPUs"

        # TODO: find how edges_x and edges_obs heads is going to be sharded (tuple for now, but might not work)
        # TODO: should it be shape_x and shape_obs? find out
        query, key, value, edges_x, edges_obs = self.shard_qkve_heads_obs(
            query, 
            key, 
            value, 
            edges_x, 
            edges_obs, 
            shapes, 
            batch_size, 
            model_comm_group
            )

        num_chunks = self.num_chunks if self.training else 4  # reduce memory for inference

        if num_chunks > 1:
            #TODO: is this correct?
            # combine edge attr (x and obs) and index (x and obs)
            edge_index_list_combined = torch.cat([edge_index_x, edge_index_obs], dim = 1)
            edge_attr_list_combined = torch.cat([edges_x, edges_obs], dim = 0)

            edge_index_list = torch.tensor_split(edge_index_list_combined, num_chunks, dim=1)
            edge_attr_list = torch.tensor_split(edge_attr_list_combined, num_chunks, dim=0)

            for i in range(num_chunks):
                out1 = self.cross_attn(
                    query=query,
                    key=key,
                    value=value,
                    edge_attr=edge_attr_list[i],
                    edge_index=edge_index_list[i],
                    size=size,
                )
                if i == 0:
                    out = torch.zeros_like(out1)
                out = out + out1
        else:
            edge_index_list_combined = torch.cat([edge_index_x, edge_index_obs], dim = 1)
            edge_attr_list_combined = torch.cat([edge_attr_x, edge_attr_obs], dim = 0)

            out = self.cross_attn(
                query=query, 
                key=key, 
                value=value, 
                edge_attr=edge_attr_list_combined, 
                edge_index=edge_index_list_combined, 
                size=size #TODO: find out if the size correct, or needs to be changed
                )

        out = self.shard_output_seq(out, shapes, batch_size, model_comm_group)
        out = self.fuse_projection_layer(out + x_r) # + obs_r) dont think obs_r is needed

        out = out + x_skip #+ obs_skip # do we need skip connection for obs??

        nodes_new_dst = self.node_dst_mlp(out) + out

        nodes_new_src = self.node_src_mlp(x_skip) + x_skip if self.update_src_nodes else x_skip

        nodes_new = (nodes_new_src, nodes_new_dst)

        return nodes_new, edge_attr_x, edge_attr_obs
