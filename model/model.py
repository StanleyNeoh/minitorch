from __future__ import annotations
from typing import Callable, Iterator
import time

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
import numpy as np

from autograd.variable import V

from .layer import Layer
from .optimiser import Optimiser


class Model:
    """
    Model acts as a wrapper for the layer, optimiser and loss function.

    It tracks various statistics of the model and provides a convenient interface for training the model.
    It also provides a convenient interface for plotting the graph of the model.

    Hence, due to the convenience, it is recommended to wrap the layer, optimiser and loss function in a model.
    """

    @classmethod
    def copy_from(
        cls,
        layer: Layer,
        optimiser: Optimiser,
        loss: Callable[..., V],
        reinitialise: bool = True,
    ) -> Model:
        """
        Copy the layer, optimiser and loss function into a model.
        Hence, original layer, optimiser and loss function will not be modified.
        If reinitialise is True, the layer will be reinitialised.

        Args:
            layer (Layer): Layer of the model.

        Returns:
            Model: Model derived from the layer, optimiser and loss function.
        """
        return Model(layer.copy(reinitialise=reinitialise), optimiser.copy(), loss)


    def __init__(
        self, layer: Layer, optimiser: Optimiser, loss: Callable[..., V]
    ) -> None:
        """
        Initialise the model with the layer, optimiser and loss function.
        Layer will be modified when training the model.

        Args:
            layer (Layer): Layer of the model.
            optimiser_type (type[Optimiser]): Optimiser constructor.
            loss (Callable[[V, V], V]): Loss function.

        Returns:
            None
        """
        optimiser.set_parameters(layer.parameters())
        self.optimiser = optimiser
        self.layer = layer
        self.loss = loss

        # Statistics
        self.loss_history: list[np.float128] = []
        self.training_time = 0.0

    def copy(self, reinitialise: bool = True) -> Model:
        """
        Copy the model. If reinitialise is True, the layer will be reinitialised.

        Returns:
            Model: Model derived from the layer, optimiser and loss function.
        """
        return Model.copy_from(
            self.layer, self.optimiser, self.loss, reinitialise=reinitialise
        )

    def __call__(self, input: V) -> V:
        """
        Forward pass of the model.

        Args:
            input (V): Input to the model.

        Returns:
            V: Output of the model.
        """
        return self.layer.forward(input)

    # Training methods
    def train(self, gen_data: Iterator[tuple[V, V]], epoch: int = 1000) -> None:
        """
        Train the model and record the loss.

        Args:
            gen_data (Iterator[tuple[V, V]]): Data generator.
            epoch (int, optional): Number of epochs. Defaults to 1000.

        Returns:
            None
        """
        time_start = time.time()
        for _ in range(epoch):
            try:
                x, y = next(gen_data)
            except StopIteration:
                break

            y_hat = self.layer.forward(x)
            loss = self.loss(y_hat, y)
            self.loss_history.append(loss.item())
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

        self.training_time += time.time() - time_start

    # Statistics methods
    def get_loss(self, window=100) -> np.float128:
        """
        Get the loss of the model.

        Args:
            last_n (int, optional): Number of last losses to average. Defaults to 100.

        Returns:
            float: Loss of the model.
        """
        if len(self.loss_history) == 0:
            return np.float128("inf")
        back = self.loss_history[-window:]
        return np.float128(sum(back) / len(back))

    def statistics(self) -> pd.Series:
        return pd.Series(
            {
                "training_time": self.training_time,
                "total_epochs": len(self.loss_history),
                "loss": self.get_loss(),
            }
        )
