import numpy as np
from functools import reduce

from .variable import V 

## Matrix operations
def sum(var: V, axis=None, keepdims=True):
    require_grad = var.requires_grad
    data = np.sum(var.data, axis=axis, keepdims=keepdims)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(out.grad * np.ones_like(var.data))
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def mean(var: V, axis=None, keepdims=True):
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
    require_grad = var.requires_grad
    data = np.sin(var.data)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(np.cos(var.data) * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def cos(var: V):
    require_grad = var.requires_grad
    data = np.cos(var.data)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(-np.sin(var.data) * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def tan(var: V): 
    require_grad = var.requires_grad
    data = np.tan(var.data)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(out.grad / np.cos(var.data) ** 2)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def relu(var: V):
    require_grad = var.requires_grad
    data = np.maximum(var.data, 0)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad((var.data > 0) * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def sinh(var: V):
    require_grad = var.requires_grad
    data = np.sinh(var.data)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(np.cosh(var.data) * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def cosh(var: V):
    require_grad = var.requires_grad
    data = np.cosh(var.data)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(np.sinh(var.data) * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def tanh(var: V):
    require_grad = var.requires_grad
    data = np.tanh(var.data)
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad((1 - data ** 2) * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def log(var: V):
    require_grad = var.requires_grad
    data = np.log(np.abs(var.data))
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(out.grad / var.data)
    out.set_backward(_backward)
    out.add_deps([var])
    return out

def sigmoid(var: V):
    require_grad = var.requires_grad
    data = 1 / (1 + np.exp(-var.data))
    out = V(data, requires_grad=require_grad)
    def _backward():
        var.add_to_grad(data * (1 - data) * out.grad)
    out.set_backward(_backward)
    out.add_deps([var])
    return out
