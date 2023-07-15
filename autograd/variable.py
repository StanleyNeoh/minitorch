from __future__ import annotations
from typing import Callable, TypeAlias, Optional

import numpy as np


class V:
    """
    A variable that can be used to build a computation graph and track gradient.

    Attributes:
        - data (npt.NDArray | np.float128): data of this variable
        - requires_grad (bool): whether to track gradient
        - grad (npt.NDArray | np.float128): gradient of this variable
        - _backward (Optional[Callable[[], None]]): backward function of this variable
        - _deps (list[V]): dependencies of this variable
    """

    def __init__(
        self,
        data: DataType,
        requires_grad: bool = False,
    ) -> None:
        """
        Initialize a variable from a numpy array or number.
        Numpy array will be converted to float128.

        Args:
            data (npt.NDArray | np.float128): data of this variable
            requires_grad (bool, optional): whether to track gradient. Defaults to False.
        Returns:
            None
        """
        self.requires_grad = requires_grad
        self.data: DataType
        self.grad: DataType
        if isinstance(data, np.float128):
            self.data = data
            self.grad = np.float128(0.0)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float128)
            self.grad = np.zeros_like(data, dtype=np.float128)
        else:
            raise TypeError(f"{type(data)} is not supported as data type")

        # deps mean dependencies == variables that makes this variable
        self._backward: Optional[Callable[[], None]] = None
        self._deps: list[V] = []

    @classmethod
    def of(cls, x: InputType, requires_grad: bool = False) -> V:
        """
        Create a variable from a data.

        If data is already a variable, return it.

        If data is a float or int, convert it to np.float128 and then convert it to a variable.

        If data is a list, convert it to a numpy array and then convert it to a variable.

        If data is a numpy array, convert it to a variable.

        Args:
            x (InputType): float, int, list, numpy array or variable
            requires_grad (bool, optional): whether to track gradient. Defaults to False.
        Returns:
            V: variable
        """
        if isinstance(x, V):
            return x
        elif isinstance(x, list):
            return V(np.array(x), requires_grad=requires_grad)
        elif isinstance(x, (int, float, np.float128)):
            return V(np.float128(x), requires_grad=requires_grad)
        elif isinstance(x, np.ndarray):
            return V(x, requires_grad=requires_grad)
        else:
            raise TypeError("Invalid data type")

    @classmethod
    def ones(cls, shape: tuple[int, ...], requires_grad: bool = False) -> V:
        """
        Create a variable of ones.

        Args:
            shape (tuple[int, ...]): shape of the variable
            requires_grad (bool, optional): whether to track gradient. Defaults to False.
        Returns:
            V: variable
        """
        return V.of(np.ones(shape), requires_grad=requires_grad)

    @classmethod
    def zeros(cls, shape: tuple[int, ...], requires_grad: bool = False) -> V:
        """
        Create a variable of zeros.

        Args:
            shape (tuple[int, ...]): shape of the variable
            requires_grad (bool, optional): whether to track gradient. Defaults to False.
        Returns:
            V: variable
        """
        return V.of(np.zeros(shape), requires_grad=requires_grad)

    @classmethod
    def randn(cls, shape: tuple[int, ...], requires_grad: bool = False) -> V:
        """
        Create a variable of random values from standard normal distribution.

        Args:
            shape (tuple[int, ...]): shape of the variable
            requires_grad (bool, optional): whether to track gradient. Defaults to False.
        Returns:
            V: variable
        """
        return V.of(np.random.randn(*shape), requires_grad=requires_grad)

    @classmethod
    def uniform(
        cls,
        shape: tuple[int, ...],
        low: float = 0.0,
        high: float = 1.0,
        requires_grad: bool = False,
    ) -> V:
        """
        Create a variable of random values from uniform distribution.

        Args:
            shape (tuple[int, ...]): shape of the variable
            range (tuple[tuple[int,int], ...]): range of the values
            requires_grad (bool, optional): whether to track gradient. Defaults to False.

        Returns:
            V: variable
        """
        return V.of(
            np.random.uniform(low, high, size=shape),
            requires_grad=requires_grad,
        )

    def is_scalar_like(self) -> bool:
        """
        Check if this variable is a scalar like.

        Returns:
            bool: whether this variable is a scalar like
        """
        return isinstance(self.data, np.float128) or self.data.size == 1

    def zero_grad(self) -> None:
        """
        Zero gradient of this variable.

        Returns:
            None
        """
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float128)

    def item(self) -> np.float128:
        """
        Get the value of this variable as a scalar like.

        Returns:
            np.float: value of this variable
        """
        assert (
            self.is_scalar_like()
        ), "variable must be scalar like to be converted to float"
        if isinstance(self.data, np.float128):
            return self.data
        return self.data.item()

    def add_to_grad(self, grad: np.ndarray | np.float128) -> None:
        """
        Add a gradient to this variable independent of require_grad.
        This is used to accumulate gradient from multiple sources.

        A few cases to consider:
        * grad is broadcastable to self.data.shape
        * self.data.shape is broadcastable to grad.shape
            * This can happen when multiple samples are used.
            * grad.shape will be (batch_size, *self.data.shape)
            * To ensure further flexibility, we wil generalise incoming grad to have shape (..., *self.data.shape).
            * We cannot have self.grad broadcast to grad.shape as this will result in self.grad having shape (..., *self.data.shape)
                which is not what we want.
            * The way to solve this is to sum over all dimensions except the last self.data.ndim dimensions.

        Args:
            grad (np.ndarray): gradient to be added
        Returns:
            None
        """
        if grad.ndim > self.data.ndim:
            grad = np.sum(grad, axis=tuple(range(grad.ndim - self.data.ndim)))
        self.grad += grad

    def add_deps(self, vars: list[V]) -> None:
        """
        Add dependencies to this variable.
        Dependencies are variables that makes this variable.
        If this variable does not require gradient, do nothing.

        Args:
            vars (list[V]): list of variables
        Returns:
            None
        """
        if not self.requires_grad:
            return
        for v in vars:
            if v.requires_grad:
                self._deps.append(v)

    def set_backward(self, _backward: Callable[[], None]) -> None:
        """
        Set backward function of this variable.

        The backward function is a callback that updates the gradients of dependencies.

        If this variable does not require gradient, do nothing.

        Args:
            _backward (Callable[[], None]): backward function
        Returns:
            None
        """
        self._backward = _backward

    def backward(self, initial: float = 1.0) -> None:
        """
        Backpropagate gradient to dependencies.
        If this variable does not require gradient, do nothing.

        Returns:
            None
        """
        assert self.is_scalar_like(), "Only scalar like variable can be backpropagated"
        # Gradient of variable must be fully evaluated before it can update gradient of dependencies
        # This can be ensured by performing topological sort on the computation graph
        topoSort = []

        def topo(var):
            for back in var._deps:
                topo(back)
            var.zero_grad()
            topoSort.append(var)

        topo(self)
        self.grad = np.full_like(self.data, initial, dtype=np.float128)
        for var in topoSort[::-1]:
            if callable(var._backward):
                var._backward()

    def shape(self) -> tuple[int, ...]:
        """
        Get shape of data.

        Returns:
            tuple[int, ...]: shape of data
        """
        return self.data.shape

    def copy(self) -> V:
        """
        Create a copy of this variable.

        Returns:
            V: copy of this variable
        """
        return V(self.data.copy(), requires_grad=self.requires_grad)

    def __neg__(self) -> V:
        """
        Negate this variable.

        Dx -x = -1
        """
        requires_grad = self.requires_grad
        data = -self.data
        out = V(data, requires_grad=requires_grad)

        def _backward():
            self.add_to_grad(-out.grad)

        out.set_backward(_backward)
        out.add_deps([self])
        return out

    def __add__(self, other: InputType) -> V:
        """
        Add this variable with another float, int, list, numpy array or variable.

        Dx x+y = 1
        Dy x+y = 1
        """
        v = V.of(other)
        requires_grad = self.requires_grad or v.requires_grad
        data = self.data + v.data
        out = V(data, requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                self.add_to_grad(out.grad)
            if v.requires_grad:
                v.add_to_grad(out.grad)

        out.set_backward(_backward)
        out.add_deps([self, v])
        return out

    def __mul__(self, other: InputType) -> V:
        """
        Multiply this variable with another float, int, list, numpy array or variable.

        Dx x*y = y
        Dy x*y = x
        """
        v = V.of(other)
        requires_grad = self.requires_grad or v.requires_grad
        data = self.data * v.data
        out = V(data, requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                self.add_to_grad(v.data * out.grad)
            if v.requires_grad:
                v.add_to_grad(self.data * out.grad)

        out.set_backward(_backward)
        out.add_deps([self, v])
        return out

    def __sub__(self, other: InputType) -> V:
        """
        Subtract this variable with another float, int, list, numpy array or variable.

        Dx x-y = 1
        Dy x-y = -1
        """
        v = V.of(other)
        requires_grad = self.requires_grad or v.requires_grad
        data = self.data - v.data
        out = V(data, requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                self.add_to_grad(out.grad)
            if v.requires_grad:
                v.add_to_grad(-out.grad)

        out.set_backward(_backward)
        out.add_deps([self, v])
        return out

    def __truediv__(self, other: InputType) -> V:
        """
        Divide this variable with another float, int, list, numpy array or variable.

        Dx x/y = 1/y
        Dy x/y = -x/y^2
        """
        v = V.of(other)
        requires_grad = self.requires_grad or v.requires_grad
        data = self.data / v.data
        out = V(data, requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                self.add_to_grad(out.grad / v.data)
            if v.requires_grad:
                v.add_to_grad(-self.data / (v.data**2.0) * out.grad)

        out.set_backward(_backward)
        out.add_deps([self, v])
        return out

    def __pow__(self, other: InputType) -> V:
        """
        Raise this absolute of this variable to the power of another float, int, list, numpy array or variable.

        Dx |x|^y = (x/|x|) * y * |x|^(y-1)
        Dy |x|^y = log(|x|) * |x|^y
        """
        v = V.of(other)
        requires_grad = self.requires_grad or v.requires_grad
        base = np.abs(self.data)
        sign = np.sign(self.data)
        data = np.power(base, v.data)
        out = V(data, requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                self.add_to_grad(
                    sign * v.data * np.power(base, v.data - 1.0) * out.grad
                )
            if v.requires_grad:
                v.add_to_grad(np.log(base) * data * out.grad)

        out.set_backward(_backward)
        out.add_deps([self, v])
        return out

    def __matmul__(self, other: InputType) -> V:
        """
        Multiply this variable with another variable.

        Matrix multiplication is taken over the last two dimensions of the input arrays.

        See also: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

        Dx x@y = y
        Dy x@y = x
        """
        v = V.of(other)
        requires_grad = self.requires_grad or v.requires_grad
        assert isinstance(
            self.data, np.ndarray
        ), "Matrix multiplication require operands to be numpy arrays"
        assert isinstance(
            v.data, np.ndarray
        ), "Matrix multiplication require operands to be numpy arrays"

        data = self.data @ v.data
        out = V(data, requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                self.add_to_grad(out.grad @ v.data.T)
            if v.requires_grad:
                v.add_to_grad(self.data.T @ out.grad)

        out.set_backward(_backward)
        out.add_deps([self, v])
        return out

    def __lt__(self, other: object) -> np.ndarray | np.bool_:
        assert isinstance(
            other, InputClasses
        ), "Comparison requires operands to be int, float, list, numpy array or variable"
        if not isinstance(other, V):
            return self.data < other
        elif isinstance(self.data, np.ndarray) and isinstance(other.data, np.ndarray):
            assert (
                self.data.shape == other.data.shape
            ), "Comparison requires operands to have the same shape"
            return self.data < other.data
        elif isinstance(self.data, np.float128) and isinstance(other.data, np.float128):
            return self.data < other.data
        raise TypeError("Comparison requires operands to be of same type")

    def __gt__(self, other: object) -> np.ndarray | np.bool_:
        assert isinstance(
            other, InputClasses
        ), "Comparison requires operands to be int, float, list, numpy array or variable"
        if not isinstance(other, V):
            return self.data > other
        elif isinstance(self.data, np.ndarray) and isinstance(other.data, np.ndarray):
            assert (
                self.data.shape == other.data.shape
            ), "Comparison requires operands to have the same shape"
            return self.data > other.data
        elif isinstance(self.data, np.float128) and isinstance(other.data, np.float128):
            return self.data > other.data
        raise TypeError("Comparison requires operands to be of same type")


InputClasses = (int, float, list, np.ndarray, V)
InputType: TypeAlias = int | float | np.float128 | list | np.ndarray | V
DataType: TypeAlias = np.ndarray | np.float128
