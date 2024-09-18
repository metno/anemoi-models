# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import uuid

import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.models.fuserobs import AnemoiObsFuser
from anemoi.models.preprocessing import Processors


class AnemoiModelInterface(torch.nn.Module):
    """Anemoi model on torch level."""

    def __init__(
        self, *, config: DotDict, graph_data: dict, statistics: dict, data_indices: IndexCollection, metadata: dict
    ) -> None:
        super().__init__()
        self.config = config
        self.id = str(uuid.uuid4())
        self.multi_step = self.config.training.multistep_input
        self.graph_data = graph_data
        self.statistics = statistics
        self.metadata = metadata
        self.data_indices = data_indices
        self._build_model()

    def _build_model(self) -> None:
        """Build the model and pre- and post-processors."""
        # Instantiate processors
        processors = [
            [name, instantiate(processor, statistics=self.statistics, data_indices=self.data_indices)]
            for name, processor in self.config.data.processors.items()
        ]

        # Assign the processor list pre- and post-processors
        self.pre_processors = Processors(processors)
        self.post_processors = Processors(processors, inverse=True)

        # Instantiate the model (Can be generalised to other models in the future, here we use AnemoiModelEncProcDec)
        self.model = AnemoiModelEncProcDec(
            config=self.config, data_indices=self.data_indices, graph_data=self.graph_data
        )

        # Use the forward method of the model  directly
        self.forward = self.model.forward

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Prediction step for the model.

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data.

        Returns
        -------
        torch.Tensor
            Predicted data.
        """
        batch = self.pre_processors(batch, in_place=False)

        with torch.no_grad():

            assert (
                len(batch.shape) == 4
            ), f"The input tensor has an incorrect shape: expected a 4-dimensional tensor, got {batch.shape}!"
            # Dimensions are
            # batch, timesteps, horizonal space, variables
            x = batch[:, 0 : self.multi_step, None, ...]  # add dummy ensemble dimension as 3rd index

            y_hat = self(x)

        return self.post_processors(y_hat, in_place=False)
    
class AnemoiFuserInterface(torch.nn.Module):

    def __init__(
        self, *, config: DotDict, graph_data: dict, statistics: dict, data_indices: dict, metadata: dict 
    ) -> None:
        super().__init__()
        self.config = config
        self.id = str(uuid.uuid4())
        self.multistep = self.config.training.multistep_input
        self.graph_data = graph_data
        self.statistics = statistics
        self.metadata = metadata
        self.data_indices = data_indices
        self._build_model()

    def _build_model(self) -> None:
        assert set(self.statistics.keys()) == set(self.data_indices.keys()), "Mismatch between mesh keys in datamodule statistics and data_indices"
        assert set(self.data_indices.keys()).issubset(set(self.graph_data.keys())), "Mesh keys for data_indices in datamodule are not all in graph mesh keys"

        processors = {mesh: [[name, instantiate(processor, statistics=self.statistics[mesh], data_indices=self.data_indices[mesh])]
        for name, processor in self.config.data.processors.items() ]
        for mesh in self.data_indices.keys()}

        self.pre_processors = {mesh: Processors(processors[mesh]) for mesh in processors.keys()}
        self.post_processors = {mesh: Processors(processors[mesh], inverse=True) for mesh in processors.keys()}

        self.model = AnemoiObsFuser(
            config=self.config, data_indices=self.data_indices, graph_data=self.graph_data
        )

        self.forward = self.model.forward

    def predict_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        batch = {mesh: self.pre_processors(batch[mesh], in_place=False) for mesh in batch.keys()}

        with torch.no_grad():

            assert (
                all(len(values) == 4 for values in batch.values())
            ), f"Input tensor from one or more of the datasets have incorrect shape: expected 4-dimensional"
            x = {mesh: batch[mesh][:, 0, self.multi_step, None, ...] for mesh in batch.keys()}

            y_hat = self(x) #Need to implement model forward so it uses dict as input / output

        return {mesh: self.post_processors[mesh](y_hat[mesh], in_place=False) for mesh in y_hat.keys()}

