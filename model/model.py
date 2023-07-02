from __future__ import annotations
from typing import Callable, Iterator
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import time

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
    def derived_from(cls, 
                     layer: Layer, 
                     optimiser_type: type[Optimiser], 
                     loss: Callable[[V, V], V]
                     ) -> Model:
        """
        Create a model from a layer, optimiser and loss function.
        Layer is copied to avoid any side effects.

        Args:
            layer (Layer): Layer of the model.
            optimiser_type (type[Optimiser]): Optimiser constructor.
            loss (Callable[[V, V], V]): Loss function.
        
        Returns:
            Model: Model derived from the layer, optimiser and loss function.
        """
        return Model(layer.copy(), optimiser_type, loss)

    def __init__(self, 
                layer: Layer,
                optimiser_type: type[Optimiser], 
                loss: Callable[[V, V], V]
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
        self.optimiser_type = optimiser_type
        self.optimiser = optimiser_type(layer.parameters()) 
        self.layer = layer
        self.loss = loss
    
        # Statistics
        self.loss_history: list[float] = []
        self.training_time = 0.0
    
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
    def train(self,
              data_generator: Iterator[tuple[V, V]],
              epoch: int = 1000
              ) -> None:
        """
        Train the model and record the loss.
        
        Args:
            data_generator (Iterator[tuple[V, V]]): Data generator.
            epoch (int, optional): Number of epochs. Defaults to 1000.

        Returns:
            None
        """
        time_start = time.time()
        for e in range(epoch):
            try:
                x, y = next(data_generator)
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
    def get_loss(self, window = 100) -> float:
        """
        Get the loss of the model.

        Args:
            last_n (int, optional): Number of last losses to average. Defaults to 100.

        Returns:
            float: Loss of the model.
        """
        if len(self.loss_history) == 0:
            return np.inf
        back = self.loss_history[-window:]
        return sum(back) / len(back)
    
    def get_training_time(self) -> float:
        """
        Get the training time of the model.

        Returns:
            float: Training time of the model.
        """
        return self.training_time
    
    def get_total_epochs(self) -> int:
        """
        Get the total number of epochs the model has been trained.

        Returns:
            int: Total number of epochs the model has been trained.
        """
        return len(self.loss_history)

    def statistics(self) -> dict[str, float]:
        return {
            "training_time": self.get_training_time(),
            "total_epochs": self.get_total_epochs(),
            "loss": self.get_loss()
        }
    
    # Plotting loss methods
    _loss_plot_queue: list[tuple[Model, str]] = []

    def queue_loss_plot(self, name: str) -> None:
        """
        Queues the model to be plotted for loss history.

        Returns:
            None
        """
        self._loss_plot_queue.append((self, name))

    def reset_loss_plot(self):
        """
        Reset the loss plot.

        Returns:
            None
        """
        plt.clf()
        self._loss_plot_queue.clear()

    @classmethod
    def save_loss_plot(self, path: str):
        """
        Save the current plot to a file.

        Args:
            path (str): Path to save the plot.

        Returns:
            None
        """
        plt.clf()
        plt.title("Loss History")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        for model, name in self._loss_plot_queue:
            plt.plot(model.loss_history, label=name)
        plt.legend()
        plt.savefig(path)
        plt.clf()
        self._loss_plot_queue.clear()