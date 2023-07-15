from __future__ import annotations
from typing import Optional

from autograd import V, F

# Base Layer
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
    
    def copy(self, reinitialise: bool = False):
        """
        Copy the layer.
    
        Args:
            reinitialise (bool, optional): Whether to reinitialise the parameters of the layer. Defaults to False.

        Returns:
            Layer: Copy of the layer.
        """
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def __str__(self):
        return self.__repr__()

# Concrete Layers
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

    def copy(self, reinitialise: bool=False) -> Layer:
        """
        Copy the sequential layer.

        Args:
            reinitialise (bool, optional): Whether to reinitialise the parameters of the layer. Defaults to False.

        Returns:
            Layer: Copy of the sequential layer.
        """
        return Sequential(*[layer.copy(reinitialise) for layer in self.layers])
    
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
    
    def copy(self, reinitialise: bool=False) -> Layer:
        """
        Copy the linear layer.

        Args:
            reinitialise (bool, optional): Whether to reinitialise the parameters of the layer. Defaults to False.

        Returns:
            Layer: Copy of the linear layer.
        """
        if reinitialise:
            return Linear(self.in_features, self.out_features)
        else:
            return Linear(self.in_features, self.out_features, self.W.copy(), self.b.copy())
        
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
    
    def copy(self, reinitialise: bool=False) -> Layer:
        """
        Copy the ReLU layer.

        Args:
            reinitialise (bool, optional): This parameter is ignored. Defaults to False.
            
        Returns:
            Layer: Copy of the ReLU layer.
        """
        return ReLU()

class Sigmoid(Layer):
    def forward(self, input) -> V:
        """
        Forward pass of the Sigmoid layer.

        Args:
            input (V): Input to the Sigmoid layer.

        Returns:
            V: Output of the Sigmoid layer.
        """
        output = F.sigmoid(input)
        return output

    def parameters(self) -> list[V]:
        """
        Get the parameters of the Sigmoid layer.

        Returns:
            list[V]: Parameters of the Sigmoid layer.
        """
        return []
    
    def copy(self, reinitialise: bool=False) -> Layer:
        """
        Copy the Sigmoid layer.

        Args:
            reinitialise (bool, optional): This parameter is ignored. Defaults to False.
            
        Returns:
            Layer: Copy of the Sigmoid layer.
        """
        return Sigmoid()
    

class Ly:
    """
    Ly is a module that contains all the layers.
    """
    Layer = Layer
    Sequential = Sequential
    Linear = Linear
    ReLU = ReLU
    Sigmoid = Sigmoid