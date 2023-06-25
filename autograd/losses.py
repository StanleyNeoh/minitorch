import numpy as np

from .variable import V # type: ignore
from .functions import F # type: ignore

def crossentropyloss(output: V, target: V, axis=None, keepdims=False):
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
    return -F.sum(target * F.log(output), axis=axis, keepdims=keepdims)

def kulldivergence(output: V, target: V, axis=None, keepdims=False):
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
    return F.sum(target * F.log(target / output), axis=axis, keepdims=keepdims)

def l1loss(output: V, target: V, axis=None, keepdims=False):
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
    return F.mean(F.abs(output - target), axis=axis, keepdims=keepdims)

def l2loss(output: V, target: V, axis=None, keepdims=False):
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
    return F.mean((output - target) ** 2, axis=axis, keepdims=keepdims)

def huberloss(output: V, target: V, delta: float=1.0, axis=None, keepdims=False):
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
    diff = output - target
    loss = F.where(
        F.abs(diff) < delta,
        (diff ** 2) * 0.5, 
        (F.abs(output - target) - (0.5 * delta)) * delta
        )
    loss = F.mean(loss, axis=axis, keepdims=keepdims)
    return loss

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
