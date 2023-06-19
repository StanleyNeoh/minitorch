import numpy as np
from functools import reduce

from .variable import V # type: ignore

## Matrix operations
def sum(var: V, axis=None, keepdims=True):
    """
    Sum of a variable along an axis.
    
    If axis is None, sum all elements.

    If axis is an integer, sum along that axis.

    If axis is a tuple, sum along all axes in the tuple.
    
    If keepdims is True, keep the dimensions of the original variable.
    
    If keepdims is False, remove the dimensions of the original variable.

    Args:
        var (V): variable
        axis (int, tuple, optional): axis to sum along. Defaults to None.
        keepdims (bool, optional): whether to keep dimensions. Defaults to True.
    Returns:
        V: variable
    """
    require_grad = var.requires_grad
    data = np.sum(var.data, axis=axis, keepdims=keepdims)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(out.grad * np.ones_like(var.data))
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def mean(var: V, axis=None, keepdims=True):
    """
    Mean of a variable along an axis.
    
    If axis is None, mean all elements.

    If axis is an integer, mean along that axis.

    If axis is a tuple, mean along all axes in the tuple.

    If keepdims is True, keep the dimensions of the original variable.

    If keepdims is False, remove the dimensions of the original variable.

    Args:
        var (V): variable
        axis (int, tuple, optional): axis to mean along. Defaults to None.
        keepdims (bool, optional): whether to keep dimensions. Defaults to True.
    Returns:
        V: variable
    """
    require_grad = var.requires_grad
    data = np.mean(var.data, axis=axis, keepdims=keepdims)
    out = V(data, requires_grad=require_grad)
    def _backward():
        if axis is None:
            n = reduce(lambda x, y: x * y, var.data.shape)
        elif type(axis) is int:
            n = var.data.shape[axis]
        elif type(axis) is tuple:
            n = reduce(lambda x, y: x * y, [var.data.shape[i] for i in axis])
        else:
            raise 'Invalid axis'
        var.add_to_grad(out.grad / n)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def softmax(var: V, axis=None):
    """
    Softmax of a variable along an axis.
    
    If axis is None, softmax all elements.

    If axis is an integer, softmax along that axis.

    If axis is a tuple, softmax along all axes in the tuple.

    Args:
        var (V): variable
        axis (int, tuple, optional): axis to softmax along. Defaults to None.
    Returns:
        V: variable
    """
    require_grad = var.requires_grad
    data = np.exp(var.data)
    data /= np.sum(data, axis=axis, keepdims=True)
    out = V(data, requires_grad=require_grad)
    def _backward():
        out.add_to_grad(out.grad * data * (1 - data))
    out.set_backward(_backward)
    out.add_deps([var])
    return out

## Element wise operations
def sin(var: V):
    """
    Sine of a variable.

    Args:
        var (V): variable

    Returns:
        V: variable
    """
    require_grad = var.requires_grad
    data = np.sin(var.data)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(np.cos(var.data) * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def cos(var: V):
    """
    Cosine of a variable.

    Args:
        var (V): variable

    Returns:
        V: variable
    """ 
    require_grad = var.requires_grad
    data = np.cos(var.data)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(-np.sin(var.data) * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def tan(var: V): 
    """
    Tangent of a variable.

    Args:
        var (V): variable

    Returns:
        V: variable
    """
    require_grad = var.requires_grad
    data = np.tan(var.data)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(out.grad / np.cos(var.data) ** 2)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def relu(var: V):
    """
    Rectified linear unit of a variable.

    Args:
        var (V): variable

    Returns:
        V: variable
    """
    require_grad = var.requires_grad
    data = np.maximum(var.data, 0)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad((var.data > 0) * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def sinh(var: V):
    """
    Hyperbolic sine of a variable.

    Args:
        var (V): variable
        
    Returns:
        V: variable
    """
    require_grad = var.requires_grad
    data = np.sinh(var.data)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(np.cosh(var.data) * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def cosh(var: V):
    """
    Hyperbolic cosine of a variable.

    Args:
        var (V): variable

    Returns:
        V: variable
    """
    require_grad = var.requires_grad
    data = np.cosh(var.data)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(np.sinh(var.data) * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def tanh(var: V):
    """
    Hyperbolic tangent of a variable.
    
    Args:
        var (V): variable

    Returns:
        V: variable
    """
    require_grad = var.requires_grad
    data = np.tanh(var.data)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad((1 - data ** 2) * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def log(var: V):
    """
    Natural logarithm of the absolute of the variable.

    Args:
        var (V): variable
    
    Returns:    
        V: variable
    """
    require_grad = var.requires_grad
    data = np.log(np.abs(var.data))
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(out.grad / var.data)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def sigmoid(var: V):
    """
    Sigmoid of a variable.

    Args:
        var (V): variable

    Returns:
        V: variable
    """
    require_grad = var.requires_grad
    data = 1 / (1 + np.exp(-var.data))
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(data * (1 - data) * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

class F:
    """
    Class containing all the functions that can be applied to a variable.

    Attributes:
        sum (function): sum of a variable along an axis.
        mean (function): mean of a variable along an axis.
        softmax (function): softmax of a variable along an axis.
        sin (function): sine of a variable.
        cos (function): cosine of a variable.
        tan (function): tangent of a variable.
        relu (function): rectified linear unit of a variable.
        sinh (function): hyperbolic sine of a variable.
        cosh (function): hyperbolic cosine of a variable.
        tanh (function): hyperbolic tangent of a variable.
        log (function): natural logarithm of the absolute of a variable.
        sigmoid (function): sigmoid of a variable.
    """
    sum = sum
    mean = mean
    softmax = softmax
    sin = sin
    cos = cos
    tan = tan
    relu = relu
    sinh = sinh
    cosh = cosh
    tanh = tanh
    log = log
    sigmoid = sigmoid
