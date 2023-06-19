from __future__ import annotations
from typing import Callable, TypeAlias, Optional

import numpy as np
import numpy.typing as npt

class V:
    def __init__(
            self, 
            data: npt.NDArray, 
            requires_grad: bool =False
            ) -> None:
        self.data = data.astype(np.float128)
        self.requires_grad = requires_grad
        self.grad: npt.NDArray = np.zeros_like(data, dtype=np.float128)

        # deps mean dependencies == variables that makes this variable
        self._backward: Optional[Callable[[], None]] = None
        self._deps: list[V] = []

    @classmethod
    def of(cls, x: Data, requires_grad: bool =False) -> V:
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
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float128)

    def add_to_grad(self, grad: npt.NDArray) -> None:
        if self.requires_grad:
            self.grad += grad
    
    def add_deps(self, vars: list[V]) -> None:
        if not self.requires_grad:
            return
        for v in vars:
            if v.requires_grad:
                self._deps.append(v)
    
    def set_backward(self, _backward: Callable[[], None]) -> None:
        if self.requires_grad:
            self._backward = _backward
    
    def backward(self) -> None:
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