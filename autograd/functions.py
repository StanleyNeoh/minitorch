import numpy as np

from functools import reduce

from .variable import V # type: ignore

## cross axis operations
def sum(var: V, axis=None, keepdims=True) -> V:
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

def mean(var: V, axis=None, keepdims=True) -> V:
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
            n = var.data.size
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

def rms(var: V, axis=None, keepdims=True) -> V:
    """
    Root mean square of a variable along an axis.

    If axis is None, rms all elements.

    If axis is an integer, rms along that axis.

    If axis is a tuple, rms along all axes in the tuple.

    If keepdims is True, keep the dimensions of the original variable.

    If keepdims is False, remove the dimensions of the original variable.

    Args:
        var (V): variable
        axis (int, tuple, optional): axis to rms along. Defaults to None.
        keepdims (bool, optional): whether to keep dimensions. Defaults to True.
    Returns:
        V: variable
    """
    require_grad = var.requires_grad
    data = np.sqrt(np.mean(var.data ** 2, axis=axis, keepdims=keepdims))
    out = V(data, requires_grad=require_grad)
    if axis is None:
        n = var.data.size
    elif type(axis) is int:
        n = var.data.shape[axis]
    elif type(axis) is tuple:
        n = reduce(lambda x, y: x * y, [var.data.shape[i] for i in axis])
    else:
        raise Exception('Invalid axis')
    def _backward():
        var.add_to_grad(out.grad * var.data / (n * out.data))
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def softmax(var: V, axis=None) -> V:
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
def abs(var: V) -> V:
    """
    Absolute value of a variable.

    Args:
        var (V): variable

    Returns:
        V: variable
    """
    require_grad = var.requires_grad
    data = np.abs(var.data)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(out.grad * np.sign(var.data))
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def sin(var: V) -> V:
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

def cos(var: V) -> V:
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

def tan(var: V) -> V: 
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

def relu(var: V) -> V:
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

def sinh(var: V) -> V:
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

def cosh(var: V) -> V:
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

def tanh(var: V) -> V:
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

def log(var: V) -> V:
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

def sigmoid(var: V) -> V:
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

def elu(var: V, a = 1.0) -> V:
    """
    Exponential linear unit of a variable.
    Alpha (a) defaults to 1.

    Args:
        var (V): variable
        a (float): alpha

    Returns:
        V: variable
    """
    require_grad = var.requires_grad
    data = np.where(var.data > 0, var.data, a * (np.exp(var.data) - 1))
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad((var.data > 0) * out.grad + (var.data <= 0) * a * np.exp(var.data) * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def leakyrelu(var: V, a = 0.01) -> V:
    """
    Leaky rectified linear unit of a variable.
    Alpha (a) defaults to 0.01.

    Args:
        var (V): variable
        a (float): alpha

    Returns:
        V: variable
    """
    require_grad = var.requires_grad
    data = np.where(var.data > 0, var.data, a * var.data)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad((var.data > 0) * out.grad + (var.data <= 0) * a * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

## Conditional operations
def where(cond: np.ndarray | np.bool_, x: V, y: V) -> V:
    """
    Conditional variable.

    Args:
        cond (np.ndarray) condition
        x (V): variable
        y (V): variable

    Returns:
        V: variable
    """
    require_grad = x.requires_grad or y.requires_grad
    data = np.where(cond, x.data, y.data)
    out = V(data, requires_grad=require_grad)
    def _backward():
        x.add_to_grad(out.grad * cond)
        y.add_to_grad(out.grad * (1.0 - cond))
    out.set_backward(_backward)
    out.add_deps([x, y])
    return out

class F:
    """
    Class containing all the functions that can be applied to a variable.

    Attributes:
        abs (function): absolute of a variable.
        sum (function): sum of a variable along an axis.
        mean (function): mean of a variable along an axis.
        rms (function): root mean square of a variable along an axis.
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
        elu (function): exponential linear unit of a variable.
        leakyrelu (function): leaky rectified linear unit of a variable.
        where (function): where a condition is met, return a variable, otherwise return another variable.
    """
    abs = abs
    sum = sum
    mean = mean
    rms = rms
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
    elu = elu
    leakyrelu = leakyrelu
    where = where
