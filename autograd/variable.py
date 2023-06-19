from __future__ import annotations
from typing import Callable, TypeAlias, Optional

import numpy as np
import numpy.typing as npt

class V:
    """
    A variable that can be used to build a computation graph and track gradient.

    Attributes:
        - data (npt.NDArray): data of this variable
        - requires_grad (bool): whether to track gradient
        - grad (npt.NDArray): gradient of this variable
        - _backward (Optional[Callable[[], None]]): backward function of this variable
        - _deps (list[V]): dependencies of this variable
    """
    def __init__(
            self, 
            data: npt.NDArray, 
            requires_grad: bool = False
            ) -> None:
        """
        Initialize a variable from a numpy array.
        Numpy array will be converted to float128.

        Args:
            data (npt.NDArray): numpy array
            requires_grad (bool, optional): whether to track gradient. Defaults to False.
        Returns:
            None
        """
        self.data = data.astype(np.float128)
        self.requires_grad = requires_grad
        self.grad: npt.NDArray = np.zeros_like(data, dtype=np.float128)

        # deps mean dependencies == variables that makes this variable
        self._backward: Optional[Callable[[], None]] = None
        self._deps: list[V] = []

    @classmethod
    def of(cls, x: Data, requires_grad: bool =False) -> V:
        """
        Create a variable from a data.

        If data is already a variable, return it.

        If data is a float or int, convert it to numpy array of size 1 and convert it to a variable.

        If data is a list, convert it to a numpy array and then convert it to a variable.

        If data is a numpy array, convert it to a variable.

        Args:
            x (Data): float, int, list, numpy array or variable
            requires_grad (bool, optional): whether to track gradient. Defaults to False.
        Returns:
            V: variable
        """
        if isinstance(x, V):
            return x

        if isinstance(x, float) or isinstance(x, int):
            data = np.array([x])
        elif isinstance(x, list):
            data = np.array(x)
        elif isinstance(x, np.ndarray):
            data = x
        else:
            raise Exception('Invalid data type')
        return V(data, requires_grad=requires_grad) 

    def zero_grad(self) -> None:
        """
        Zero gradient of this variable.
        
        Returns:
            None
        """
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float128)

    def add_to_grad(self, grad: npt.NDArray) -> None:
        """
        Add a gradient to this variable.
        This is used to accumulate gradient from multiple sources.
        If this variable does not require gradient, do nothing.

        Args:
            grad (npt.NDArray): gradient to be added
        Returns:
            None
        """
        if self.requires_grad:
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
        if self.requires_grad:
            self._backward = _backward
    
    def backward(self) -> None:
        """
        Backpropagate gradient to dependencies.
        If this variable does not require gradient, do nothing.

        Returns:
            None
        """
        assert self.data.size == 1, 'Only scalar variable can be backpropagated' 
        # Gradient of variable must be fully evaluated before it can update gradient of dependencies
        # This can be ensured by performing topological sort on the computation graph
        topoSort = []
        def topo(var):
            for back in var._deps:
                topo(back)
            var.zero_grad()
            topoSort.append(var)
        topo(self)
        self.grad = np.ones_like(self.data, dtype=np.float128)
        for var in topoSort[::-1]:
            if callable(var._backward):
                var._backward()

    def item(self, *args):
        """
        Get item from data.

        Args:
            *args: arguments to be passed to numpy.ndarray.item
        Returns:
            float: item from data
        """
        return self.data.item(*args)

    def __repr__(self):
        return 'V(data={}, grad={}, requires_grad={})'.format(
            self.data, self.grad, self.requires_grad)

    def __str__(self) -> str:
        if self.requires_grad:
            return 'var({})'.format(self.data)
        else:
            return 'con({})'.format(self.data)
    
    def __neg__(self) -> V:
        # Dx -x = -1

        requires_grad = self.requires_grad
        data = -self.data
        out = V(data, requires_grad=requires_grad)
        def _backward():
            self.add_to_grad(-out.grad)
        out.set_backward(_backward)
        out.add_deps([self])
        return out

    def __add__(self, other: Data | V) -> V:
        # Dx x+y = 1
        # Dy x+y = 1

        v = V.of(other) 
        requires_grad = self.requires_grad or v.requires_grad
        data = self.data + v.data
        out = V(data, requires_grad=requires_grad)
        def _backward():
            self.add_to_grad(out.grad)
            v.add_to_grad(out.grad)
        out.set_backward(_backward)
        out.add_deps([self, v])
        return out

    def __mul__(self, other: V) -> V:
        # Dx x*y = y
        # Dy x*y = x

        v = V.of(other) 
        requires_grad = self.requires_grad or v.requires_grad
        data = self.data * v.data
        out = V(data, requires_grad=requires_grad)
        def _backward():
            self.add_to_grad(v.data * out.grad)
            v.add_to_grad(self.data * out.grad)
        out.set_backward(_backward)
        out.add_deps([self, v])
        return out
    
    def __sub__(self, other: V) -> V:
        # Dx x-y = 1
        # Dy x-y = -1

        v = V.of(other) 
        requires_grad = self.requires_grad or v.requires_grad
        data = self.data - v.data
        out = V(data, requires_grad=requires_grad)
        def _backward():
            self.add_to_grad(out.grad)
            v.add_to_grad(-out.grad)
        out.set_backward(_backward)
        out.add_deps([self, v])
        return out

    def __truediv__(self, other: V) -> V:
        # Dx x/y = 1/y
        # Dy x/y = -x/y^2

        v = V.of(other) 
        requires_grad = self.requires_grad or v.requires_grad
        data = self.data / v.data
        out = V(data, requires_grad=requires_grad)
        def _backward():
            self.add_to_grad(out.grad / v.data)
            v.add_to_grad(-self.data / (v.data ** 2.0) * out.grad)
        out.set_backward(_backward)
        out.add_deps([self, v])
        return out
    
    def __pow__(self, other: V) -> V:
        # Note: absolute value of self.data will be taken
        # Dx |x|^y = (x/|x|) * y * |x|^(y-1) 
        # Dy |x|^y = log(|x|) * |x|^y

        v = V.of(other) 
        requires_grad = self.requires_grad or v.requires_grad
        base = np.abs(self.data)
        sign = np.sign(self.data)
        data = np.power(base, v.data)
        out = V(data, requires_grad=requires_grad)
        def _backward():
            self.add_to_grad(sign * v.data * np.power(base, v.data - 1.0) * out.grad)
            v.add_to_grad(np.log(base) * data * out.grad)
        out.set_backward(_backward)
        out.add_deps([self, v])
        return out

Data: TypeAlias = int | float | list | npt.NDArray | V