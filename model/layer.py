from typing import Optional, Callable
import matplotlib.pyplot as plt # type: ignore
import numpy as np

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
    
    def copy(self):
        """
        Copy the layer.

        Returns:
            Layer: Copy of the layer.
        """
        raise NotImplementedError

class Model(Layer):
    """
    Model is a special layer that acts as a wrapper for the layer, optimiser and loss function.
    It is used to train the model and plot the loss.
    """
    def __init__(self, 
                layer: Layer,
                optimiser_class: Callable[[list[V]], Optimiser],
                loss: Callable[[V, V], V]
                ) -> None:
        """
        Initialise the model with the layer, optimiser and loss function.

        Args:
            layer (Layer): Layer of the model.
            optimiser_class (Callable[[list[V]], Optimiser]): Optimiser constructor.
            loss (Callable[[V, V], V]): Loss function.

        Returns:
            None
        """
        self.optimiser_class = optimiser_class
        self.optimiser = optimiser_class(layer.parameters()) 
        self.layer = layer
        self.loss = loss
        self.loss_history: list[np.float128] = []
    
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

    def copy(self) -> Layer:
        """
        Copy the model.

        Returns:
            Layer: Copy of the model.
        """
        return Model(self.layer.copy(), self.optimiser_class, self.loss)
    
    def train(self, x: V, y: V) -> None:
        """
        Train the model and record the loss.
        
        Args:
            x (V): Input to the model.
            y (V): Target of the model.

        Returns:
            None
        """
        y_hat = self.layer.forward(x)
        loss = self.loss(y_hat, y)
        self.loss_history.append(loss.item())
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
    
    def plot_loss(self) -> None:
        """
        Plot the loss history.

        Returns:
            None
        """
        plt.plot(self.loss_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()
    
    def plot_graph(self,
                   minX: float = -10.0,
                   maxX: float = 10.0,
                   step: float = 0.1,
                   ref: Optional[Callable[[V], V]] = None
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
        plt.show()

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

    def copy(self) -> Layer:
        """
        Copy the sequential layer.

        Returns:
            Layer: Copy of the sequential layer.
        """
        return Sequential(*[layer.copy() for layer in self.layers])
        


class Linear(Layer):
    """
    Linear layer.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        name (str, optional): Name of the layer. Defaults to "Linear".
    """
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            name: str ="Linear", 
            ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.W = V.randn((in_features, out_features), requires_grad=True, name=name + ".W")
        self.b = V.zeros((out_features,), requires_grad=True, name=name + ".b")

    def forward(self, input) -> V:
        output = input @ self.W + self.b
        return output

    def parameters(self) -> list[V]:
        return [self.W, self.b]
    
    def copy(self) -> Layer:
        return Linear(self.in_features, self.out_features, self.name)

class ReLU(Layer):
    def __init__(self, name="ReLU") -> None:
        self.name = name

    def forward(self, input) -> V:
        output = F.relu(input)
        return output

    def parameters(self) -> list[V]:
        return []
    
    def copy(self) -> Layer:
        return ReLU(self.name)
