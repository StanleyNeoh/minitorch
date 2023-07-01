from __future__ import annotations
from typing import Callable, TypeAlias, Optional

import numpy as np

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
            data: DataType,
            requires_grad: bool = False,
            name: Optional[str] = None
            ) -> None:
        """
        Initialize a variable from a numpy array.
        Numpy array will be converted to float128.

        Args:
            data (npt.NDArray | np.float128): data of this variable
            requires_grad (bool, optional): whether to track gradient. Defaults to False.
        Returns:
            None
        """
        self.data: DataType
        if isinstance(data, np.float128):
            self.data = data 
        else:
            self.data = data.astype(np.float128)
        self.requires_grad = requires_grad
        self.grad: np.ndarray = np.zeros_like(data, dtype=np.float128)

        # deps mean dependencies == variables that makes this variable
        self._backward: Optional[Callable[[], None]] = None
        self._deps: list[V] = []
        self.name = name

    @classmethod
    def of(cls, x: InputType, requires_grad: bool = False, name: Optional[str] = None) -> V:
        """
        Create a variable from a data.

        If data is already a variable, return it.

        If data is a float or int, convert it to numpy array of size 1 and convert it to a variable.

        If data is a list, convert it to a numpy array and then convert it to a variable.

        If data is a numpy array, convert it to a variable.

        Args:
            x (InputType): float, int, list, numpy array or variable
            requires_grad (bool, optional): whether to track gradient. Defaults to False.
        Returns:
            V: variable
        """
        assert isinstance(x, InputClasses), f'Invalid data type: {type(x)}'
        if isinstance(x, V):
            return x

        data: np.ndarray | np.float128
        if isinstance(x, float) or isinstance(x, int):
            data = np.float128(x)
        elif isinstance(x, list):
            data = np.array(x)
        elif isinstance(x, np.ndarray):
            data = x
        else:
            raise Exception('Invalid data type')
        return V(data, requires_grad=requires_grad, name=name) 

    @classmethod
    def ones(cls, shape: tuple[int, ...], requires_grad: bool =False, name: Optional[str] =None) -> V:
        """
        Create a variable of ones.

        Args:
            shape (tuple[int, ...]): shape of the variable
            requires_grad (bool, optional): whether to track gradient. Defaults to False.
        Returns:
            V: variable
        """
        return V.of(np.ones(shape), requires_grad=requires_grad, name=name)
    
    @classmethod
    def zeros(cls, shape: tuple[int, ...], requires_grad: bool =False, name: Optional[str] =None) -> V:
        """
        Create a variable of zeros.

        Args:
            shape (tuple[int, ...]): shape of the variable
            requires_grad (bool, optional): whether to track gradient. Defaults to False.
        Returns:
            V: variable
        """
        return V.of(np.zeros(shape), requires_grad=requires_grad, name=name)

    @classmethod
    def randn(cls, shape: tuple[int, ...], requires_grad: bool =False, name: Optional[str] =None) -> V:
        """
        Create a variable of random values from standard normal distribution.

        Args:
            shape (tuple[int, ...]): shape of the variable
            requires_grad (bool, optional): whether to track gradient. Defaults to False.
        Returns:
            V: variable
        """
        return V.of(np.random.randn(*shape), requires_grad=requires_grad, name=name)
    
    @classmethod
    def uniform(cls, 
                shape: tuple[int, ...], 
                low: float =0.0,
                high: float =1.0,
                requires_grad: bool =False, 
                name: Optional[str] =None) -> V:
        """
        Create a variable of random values from uniform distribution.

        Args:
            shape (tuple[int, ...]): shape of the variable
            range (tuple[tuple[int,int], ...]): range of the values
            requires_grad (bool, optional): whether to track gradient. Defaults to False.

        Returns:
            V: variable
        """
        return V.of(np.random.uniform(low, high, size=shape), requires_grad=requires_grad, name=name)

    def isscalar(self) -> bool:
        """
        Check if this variable is a scalar.

        Returns:
            bool: whether this variable is a scalar
        """
        return isinstance(self.data, np.float128) 

    def zero_grad(self) -> None:
        """
        Zero gradient of this variable.
        
        Returns:
            None
        """
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float128)

    def item(self) -> float:
        """
        Get the value of this variable as a scalar.

        Returns:
            np.float: value of this variable
        """
        if isinstance(self.data, np.float128):
            return float(self.data)
        return self.data.item()

    def add_to_grad(self, grad: np.ndarray) -> None:
        """
        Add a gradient to this variable.
        This is used to accumulate gradient from multiple sources.
        If this variable does not require gradient, do nothing.
       
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
        if self.requires_grad:
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
        if self.requires_grad:
            self._backward = _backward
    
    def backward(self, initial = 1.0) -> None:
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

    def __repr__(self):
        return 'V(data={}, grad={}, requires_grad={})'.format(
            self.data, self.grad, self.requires_grad)

    def __str__(self) -> str:
        if self.requires_grad:
            return 'var({})'.format(self.data)
        else:
            return 'con({})'.format(self.data)
    
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
            self.add_to_grad(out.grad)
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
            self.add_to_grad(v.data * out.grad)
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
            self.add_to_grad(out.grad)
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
            self.add_to_grad(out.grad / v.data)
            v.add_to_grad(-self.data / (v.data ** 2.0) * out.grad)
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
            self.add_to_grad(sign * v.data * np.power(base, v.data - 1.0) * out.grad)
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
        assert isinstance(self.data, np.ndarray), 'Matrix multiplication require operands to be numpy arrays'
        assert isinstance(v.data, np.ndarray), 'Matrix multiplication require operands to be numpy arrays'

        data = self.data @ v.data
        out = V(data, requires_grad=requires_grad)
        def _backward():
            self.add_to_grad(out.grad @ v.data.T )
            v.add_to_grad(self.data.T @ out.grad)
        out.set_backward(_backward)
        out.add_deps([self, v])
        return out

    def __lt__(self, other: object) -> np.ndarray | np.bool_:
        assert isinstance(other, InputClasses), 'Comparison requires operands to be int, float, list, numpy array or variable'
        if not isinstance(other, V):
            return self.data < other
        elif isinstance(self.data, np.ndarray) and isinstance(other.data, np.ndarray):
            assert self.data.shape == other.data.shape, 'Comparison requires operands to have the same shape'
            return self.data < other.data
        elif isinstance(self.data, np.float128) and isinstance(other.data, np.float128):
            return self.data < other.data
        raise TypeError('Comparison requires operands to be of same type')

    def __gt__(self, other: object) -> np.ndarray | np.bool_:
        assert isinstance(other, InputClasses), 'Comparison requires operands to be int, float, list, numpy array or variable'
        if not isinstance(other, V):
            return self.data > other
        elif isinstance(self.data, np.ndarray) and isinstance(other.data, np.ndarray):
            assert self.data.shape == other.data.shape, 'Comparison requires operands to have the same shape'
            return self.data > other.data
        elif isinstance(self.data, np.float128) and isinstance(other.data, np.float128):
            return self.data > other.data
        raise TypeError('Comparison requires operands to be of same type')

InputClasses = (int, float, list, np.ndarray, V)
InputType: TypeAlias = int | float | list | np.ndarray | V
DataType: TypeAlias = np.ndarray | np.float128