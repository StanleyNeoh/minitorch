import numpy as np

from autograd.variable import V  # type: ignore

# Refer to https://medium.com/nerd-for-tech/optimizers-in-machine-learning-f1a9c549f8b4


class Optimiser:
    """
    Base class for all optimisers.
    """

    def __init__(self):
        """
        Initialise the optimiser with the parameters to optimise.

        Returns:
            None
        """
        self.parameters = None

    def set_parameters(self, parameters: list[V]):
        """
        Add parameters to optimise.

        Args:
            parameters (list[V]): Parameters to optimise.

        Returns:
            None
        """
        self.parameters = parameters

    def zero_grad(self):
        """
        Zero the gradients of all parameters.

        Returns:
            None
        """
        for parameter in self.parameters:
            parameter.grad = np.zeros_like(parameter.data)

    def step(self):
        """
        Update the parameters.

        Returns:
            None

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    def copy(self):
        """
        Copy the optimiser.

        Returns:
            Optimiser: Optimiser derived from the parameters.
        """
        return self.__class__()

    def __repr__(self):
        return "{}(parameters={})".format(self.__class__, self.parameters)

    def __str__(self):
        return str(self.__repr__())


class SGD(Optimiser):
    """
    Stochastic Gradient Descent optimiser.
    """

    def __init__(self, lr: float = 0.01):
        """
        Initialise the optimiser with the parameters to optimise and the learning rate.

        Args:
            lr (float): Learning rate. Defaults to 0.01.
        Returns:
            None
        """
        super().__init__()
        self.lr = lr

    def step(self):
        assert self.parameters is not None, "Parameters not set."
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad


class Momentum(Optimiser):
    """
    Momentum optimiser.
    """

    def __init__(self, lr: float = 0.01, b_v: float = 0.9):
        """
        Initialise the optimiser with the parameters to optimise, the learning rate and the momentum coefficient.

        Args:
            lr (float): Learning rate. Defaults to 0.01.
            b_v (float): Momentum coefficient. Defaults to 0.9.

        Returns:
            None
        """
        super().__init__()
        self.lr = lr
        self.b_v = b_v

    def set_parameters(self, parameters: list[V]):
        self.vs = [np.zeros_like(parameter.data) for parameter in parameters]
        return super().set_parameters(parameters)

    def step(self):
        """
        Update the parameters.

        Returns:
            None
        """
        assert self.parameters is not None, "Parameters not set."
        for i, parameter in enumerate(self.parameters):
            self.vs[i] = self.b_v * self.vs[i] + (1 - self.b_v) * parameter.grad
            parameter.data -= self.lr * self.vs[i]


class Adagrad(Optimiser):
    """
    Adagrad optimiser.
    """

    def __init__(self, lr: float = 0.01, eps: float = 1e-8):
        """
        Initialise the optimiser with the parameters to optimise, the learning rate and the epsilon value.

        Args:
            lr (float): Learning rate. Defaults to 0.01.
            eps (float): Epsilon value to prevent division by zero. Defaults to 1e-8.
        """
        super().__init__()
        self.lr = lr
        self.eps = eps

    def set_parameters(self, parameters: list[V]):
        self.gs = [np.full_like(parameter.data, self.eps) for parameter in parameters]
        return super().set_parameters(parameters)

    def step(self):
        """
        Update the parameters.

        Returns:
            None
        """
        assert self.parameters is not None, "Parameters not set."
        for i, parameter in enumerate(self.parameters):
            self.gs[i] += parameter.grad**2
            parameter.data -= self.lr * parameter.grad / np.sqrt(self.gs[i])


class RMSProp(Optimiser):
    """
    RMSProp optimiser.
    """

    def __init__(self, lr: float = 0.01, b_g: float = 0.9, eps: float = 1e-8):
        """
        Initialise the optimiser with the parameters to optimise, the learning rate, the decay rate and the epsilon value.

        Args:
            lr (float): Learning rate.
            b_g (float): Decay rate.
            eps (float): Epsilon value to prevent division by zero.

        Returns:
            None
        """
        super().__init__()
        self.lr = lr
        self.b_g = b_g
        self.eps = eps

    def set_parameters(self, parameters: list[V]):
        self.gs = [np.full_like(parameter.data, self.eps) for parameter in parameters]
        return super().set_parameters(parameters)

    def step(self):
        """
        Update the parameters.

        Returns:
            None
        """
        assert self.parameters is not None, "Parameters not set."
        for i, parameter in enumerate(self.parameters):
            self.gs[i] = self.b_g * self.gs[i] + (1 - self.b_g) * parameter.grad**2
            parameter.data -= self.lr * parameter.grad / np.sqrt(self.gs[i])


class Adam(Optimiser):
    """
    Adam optimiser.
    """

    def __init__(
        self, lr: float = 0.01, b_v: float = 0.9, b_g: float = 0.9, eps: float = 1e-8
    ):
        """
        Initialise the optimiser with the parameters to optimise, the learning rate, the momentum coefficient, the decay rate and the epsilon value.

        Args:
            parameters (list[V]): Parameters to optimise.
            lr (float): Learning rate.
            b_v (float): Momentum coefficient.
            b_g (float): Decay rate.
            eps (float): Epsilon value to prevent division by zero.

        Returns:
            None
        """
        super().__init__()
        self.lr = lr
        self.b_v = b_v
        self.b_g = b_g
        self.eps = eps

    def set_parameters(self, parameters: list[V]):
        self.gs = [np.full_like(parameter.data, self.eps) for parameter in parameters]
        self.vs = [np.zeros_like(parameter.data) for parameter in parameters]
        return super().set_parameters(parameters)

    def step(self):
        """
        Update the parameters.

        Returns:
            None
        """
        assert self.parameters is not None, "Parameters not set."
        for i, parameter in enumerate(self.parameters):
            self.vs[i] = self.b_v * self.vs[i] + (1 - self.b_v) * parameter.grad
            self.gs[i] = self.b_g * self.gs[i] + (1 - self.b_g) * parameter.grad**2
            parameter.data -= self.lr * self.vs[i] / np.sqrt(self.gs[i])


class O:
    """
    O is a module that contains all the optimisers.
    """

    Optimiser = Optimiser
    SGD = SGD
    Momentum = Momentum
    Adagrad = Adagrad
    RMSProp = RMSProp
    Adam = Adam
