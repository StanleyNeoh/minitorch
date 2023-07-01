from __future__ import annotations
from typing import Optional, Callable, Iterator
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import time

from autograd.variable import V
from autograd.functions import F
from .optimiser import Optimiser

class Layer:
    """
    Base class for all layers.
    """
    def forward(self, input: V) -> V:
        """
        Forward pass of the layer.

        Args:
            input (V): Input to the layer.
        
        Returns:
            V: Output of the layer.
        """
        raise NotImplementedError

    def parameters(self):
        """
        Get the parameters of the layer.

        Returns:
            list[V]: Parameters of the layer.
        """
        raise NotImplementedError
    
    def copy(self, complete: bool = True):
        """
        Copy the layer.
    
        Args:
            complete (bool, optional): Whether to copy the parameters of the layer. Defaults to True.

        Returns:
            Layer: Copy of the layer.
        """
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def __str__(self):
        return self.__repr__()

class Model(Layer):
    """
    Model is a special layer that acts as a wrapper for the layer, optimiser and loss function.
    It is used to train the model and plot the loss and other analysis.
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
                loss: Callable[[V, V], V],
                name: str = "Model" 
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
        self.loss_history: list[float] = []
        self.name = name
    

        # Statistics
        self.training_time = 0.0
    
    def forward(self, input: V) -> V:
        """
        Forward pass of the model.

        Args:
            input (V): Input to the model.

        Returns:
            V: Output of the model.
        """
        return self.layer.forward(input)
    
    def parameters(self) -> list[V]:
        """
        Get the parameters of the model.

        Returns:
            list[V]: Parameters of the model.
        """
        return self.layer.parameters()

    def copy(self, complete: bool=True) -> Layer:
        """
        Copy the model.

        Args:
            complete (bool, optional): Whether to copy the parameters of the model. Defaults to True.

        Returns:
            Layer: Copy of the model.
        """
        return Model(self.layer.copy(complete), self.optimiser_type, self.loss)
    
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
    
    def plot_graph(self,
                   minX: float = -10.0,
                   maxX: float = 10.0,
                   step: float = 0.1,
                   ref: Optional[Callable[[V], V]] = None,
                   show: bool = True,
                   save: Optional[str] = None
                   ) -> None:
        """
        Plot the graph of the model.
        Note that the model must be a model that takes in a single input and outputs a single output.

        Args:
            minX (float, optional): Minimum x value. Defaults to -10.0.
            maxX (float, optional): Maximum x value. Defaults to 10.0.
            step (float, optional): Step size. Defaults to 0.1.
            ref (Callable[[V], V], optional): Reference function to compare with. Defaults to None.
        Returns:
            None
        """
        x = V(np.arange(minX, maxX, step).reshape(-1, 1))
        y = self.layer.forward(x)
        plt.plot(x.data, y.data)
        if ref is not None:
            plt.plot(x.data, ref(x).data)
        plt.xlabel("x")
        plt.ylabel("y")
        if save is not None:
            plt.savefig(save)
        if show:
            plt.show()
    
    def plot_loss(self,
                  save: Optional[str] = None
                  ) -> None:
        """
        Plot the loss history.

        Returns:
            None
        """
        plt.plot(self.loss_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        if save is not None:
            plt.savefig(save)

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
    
class Sequential(Layer):
    """
    Sequential is a layer that combines multiple layers into a single layer.

    Args:
        *layers (Layer): Layers to be combined.
    """
    def __init__(self, *layers: Layer) -> None:
        """
        Initialise the sequential layer with the layers.

        Args:
            *layers (Layer): Layers to be combined.

        Returns:
            None
        """
        self.layers = layers
    
    def forward(self, input: V) -> V:
        """
        Forward pass of the sequential layer.

        Args:
            input (V): Input to the sequential layer.

        Returns:
            V: Output of the sequential layer.
        """
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def parameters(self) -> list[V]:
        """
        Get the parameters of the sequential layer.

        Returns:
            list[V]: Parameters of the sequential layer.
        """
        parameters = []
        for layer in self.layers:
            parameters += layer.parameters()
        return parameters

    def copy(self, complete: bool=True) -> Layer:
        """
        Copy the sequential layer.

        Returns:
            Layer: Copy of the sequential layer.
        """
        return Sequential(*[layer.copy(complete) for layer in self.layers])
    
class Linear(Layer):
    """
    Linear layer.
    """
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            W: Optional[V] = None,
            b: Optional[V] = None
            ) -> None:
        """
        Initialise the linear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            name (str, optional): Name of the layer. Defaults to "Linear".
            W (Optional[V], optional): Weight matrix. Defaults to None.
            b (Optional[V], optional): Bias vector. Defaults to None.

        Returns:
            None
        """
        self.in_features = in_features
        self.out_features = out_features
        if W is None:
            self.W = V.randn((in_features, out_features), requires_grad=True)
        else:
            self.W = W
        if b is None:
            self.b = V.zeros((out_features,), requires_grad=True)
        else:
            self.b = b

    def forward(self, input) -> V:
        """
        Forward pass of the linear layer.

        Args:
            input (V): Input to the linear layer.

        Returns:
            V: Output of the linear layer.
        """
        output = input @ self.W + self.b
        return output

    def parameters(self) -> list[V]:
        """
        Get the parameters of the linear layer.

        Returns:
            list[V]: Parameters of the linear layer.
        """
        return [self.W, self.b]
    
    def copy(self, complete: bool=True) -> Layer:
        """
        Copy the linear layer.

        Args:
            complete (bool, optional): Whether to copy the parameters of the layer. Defaults to True.

        Returns:
            Layer: Copy of the linear layer.
        """
        if complete:
            return Linear(self.in_features, self.out_features, self.W.copy(), self.b.copy())
        else:
            return Linear(self.in_features, self.out_features)
        
class ReLU(Layer):
    def forward(self, input) -> V:
        """
        Forward pass of the ReLU layer.

        Args:
            input (V): Input to the ReLU layer.

        Returns:
            V: Output of the ReLU layer.
        """
        output = F.relu(input)
        return output

    def parameters(self) -> list[V]:
        """
        Get the parameters of the ReLU layer.

        Returns:
            list[V]: Parameters of the ReLU layer.
        """
        return []
    
    def copy(self, complete: bool=True) -> Layer:
        """
        Copy the ReLU layer.

        Args:
            complete (bool, optional): Whether to copy the parameters of the layer. Defaults to True.
            
        Returns:
            Layer: Copy of the ReLU layer.
        """
        return ReLU()
    