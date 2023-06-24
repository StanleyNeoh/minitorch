import numpy as np

from .variable import V # type: ignore

def crossentropyloss(output: V, target: V, axis=None, keepdims=True):
    """
    Cross entropy loss for a variable.
    
    Cross entropy loss is used for classification problems 
    where the output is a probability distribution 
    and the target is a one-hot vector.

    A one-hot vector is a vector with all zeros except for one element which is 1.

    Cross entropy loss is a specialised version of Kullback-Leibler divergence where
    the entropy of the target distribution is 0 (i.e. the target distribution is a one-hot vector).

    Zero cross entropy loss is only achieved when the target is a one-hot vector 
    and the output matches exactly to that one-hot vector.

    This is because a one-hot vector is interpreted as a random variable 
    where we are completely certain about the outcome and the minimum number of bits
    required to encode the outcome is 0.

    **Note**: This implementation of cross entropy loss uses the natural logarithm. 

    Args:
        output (V): output variable
        target (V): target variable
    Returns:
        V: loss variable
    """
    require_grad = output.requires_grad or target.requires_grad
    data = -np.sum(target.data * np.log(output.data), axis=axis, keepdims=keepdims)
    out = V(data, requires_grad=require_grad)
    def _backward():
        output.add_to_grad(-target.data / output.data * out.grad)
        target.add_to_grad(-np.log(output.data) * out.grad)
    out.set_backward(_backward)
    out.add_deps([output, target])
    return out

def kulldivergence(output: V, target: V, axis=None, keepdims=True):
    """
    Kullback-Leibler divergence for a variable.

    Kullback-Leibler divergence is used for classification problems where the output is a probability distribution
    and the target is a probability distribution.

    Kullback-Leibler divergence is the difference between the cross entropy loss and the entropy of the target distribution.

    Kullback-Leibler can be thought of as a generalised cross entropy loss that can work with any target distribution
    that does not have zero probability for any outcome.

    **Note**: This implementation of cross entropy loss uses the natural logarithm. 

    Args:
        output (V): output variable
        target (V): target variable
    Returns:
        V: loss variable
    """
    require_grad = output.requires_grad or target.requires_grad
    data = np.sum(target.data * np.log(target.data / output.data), axis=axis, keepdims=keepdims)
    out = V(data, requires_grad=require_grad)
    def _backward():
        output.add_to_grad(-target.data / output.data * out.grad)
        target.add_to_grad((np.log(target.data / output.data) + 1) * out.grad)
    out.set_backward(_backward)
    out.add_deps([output, target])
    return out

def l1loss(output: V, target: V, axis=None, keepdims=True):
    """
    L1 loss for a variable.

    L1 loss is used for regression problems where the output is a real number.

    L1 loss is less sensitive to outliers than L2 loss.

    Args:
        output (V): output variable
        target (V): target variable
    Returns:
        V: loss variable
    """
    require_grad = output.requires_grad or target.requires_grad
    data = np.sum(np.abs(output.data - target.data), axis=axis, keepdims=keepdims)
    out = V(data, requires_grad=require_grad)
    def _backward():
        output.add_to_grad(np.sign(output.data - target.data) * out.grad)
        target.add_to_grad(-np.sign(output.data - target.data) * out.grad)
    out.set_backward(_backward)
    out.add_deps([output, target])
    return out

def l2loss(output: V, target: V, axis=None, keepdims=True):
    """
    L2 loss for a variable.

    L2 loss is used for regression problems where the output is a real number.

    L2 loss is more sensitive to outliers than L1 loss.

    Args:
        output (V): output variable
        target (V): target variable
    Returns:
        V: loss variable
    """
    require_grad = output.requires_grad or target.requires_grad
    data = np.sum((output.data - target.data) ** 2, axis=axis, keepdims=keepdims)
    out = V(data, requires_grad=require_grad)
    def _backward():
        output.add_to_grad(2 * (output.data - target.data) * out.grad)
        target.add_to_grad(-2 * (output.data - target.data) * out.grad)
    out.set_backward(_backward)
    out.add_deps([output, target])
    return out

def huberloss(output: V, target: V, delta: float=1.0):
    """
    Huber loss for a variable.

    Huber loss is used for regression problems where the output is a real number.

    Huber loss is a combination of L1 and L2 loss. Hence, it is less sensitive to outliers than L2 loss.

    Args:
        output (V): output variable
        target (V): target variable
        delta (float): threshold
    Returns:
        V: loss variable
    """
    require_grad = output.requires_grad or target.requires_grad
    diff = output.data - target.data
    data = np.where(np.abs(diff) < delta, 0.5 * diff ** 2, delta * (np.abs(diff) - 0.5 * delta))
    out = V(data, requires_grad=require_grad)
    def _backward():
        grad = np.where(np.abs(diff) < delta, diff, delta * np.sign(diff))
        output.add_to_grad(grad * out.grad)
        target.add_to_grad(-grad * out.grad)
    out.set_backward(_backward)
    out.add_deps([output])
    return out

class L:
    """
    Class for loss functions.

    Attributes:
        crossentropyloss (function): cross entropy loss
        kulldivergence (function): Kullback-Leibler divergence
        l1loss (function): L1 loss
        l2loss (function): L2 loss
        huberloss (function): Huber loss
    """
    crossentropyloss = crossentropyloss
    kulldivergence = kulldivergence
    l1loss = l1loss
    l2loss = l2loss
    huberloss = huberloss
