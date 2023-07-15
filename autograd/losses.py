import numpy as np

from .variable import V # type: ignore
from .functions import F # type: ignore

def l1loss(output: V, target: V, axis=None, keepdims=False) -> V:
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

def l2loss(output: V, target: V, axis=None, keepdims=False) -> V:
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

def rmsloss(output: V, target: V, axis=None, keepdims=False) -> V:
    return F.rms(output - target, axis=axis, keepdims=keepdims)

def huberloss(output: V, target: V, delta: float=1.0, axis=None, keepdims=False) -> V:
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

def crossentropyloss(output: V, target_i: list[int], axis=None, keepdims=False) -> V:
    """
    Cross entropy loss for a variable.

    output must be of the format (batch_size, num_classes) 
    and target_i must be of the format (batch_size,) acting as indices for the target classes.

    Args:
        output (V): output variable
        target_i (list[int]): target indices
    
    Returns:
        V: loss variable
    """
    assert output.data.ndim == 2, "output must be of the format (batch_size, num_classes)"
    assert len(target_i) == output.data.shape[0], "target_i must be of the format (batch_size,)"
    num_batches = len(target_i)
    requires_grad = output.requires_grad
    est_prob = output.data[np.arange(num_batches), target_i]
    data = -np.mean(np.log(est_prob))
    loss = V.of(data, requires_grad=requires_grad)
    def _backward() -> None:
        if output.requires_grad:
            slicer = (np.arange(num_batches), target_i)
            masked = np.zeros(output.data.shape)
            masked[*slicer] = (-1.0 / (num_batches * output.data))[*slicer]
            output.add_to_grad(masked * loss.grad)
    loss.set_backward(_backward)
    loss.add_deps([output])
    return loss

class L:
    """
    Class for loss functions.

    Attributes:
        l1loss (function): L1 loss
        l2loss (function): L2 loss
        rmseloss (function): Root mean squared error loss
        huberloss (function): Huber loss
        crossentropyloss (function): cross entropy loss
    """
    l1loss = l1loss
    l2loss = l2loss
    rmsloss = rmsloss
    huberloss = huberloss
    crossentropyloss = crossentropyloss

# Deprecated as not practical

def _crossentropyloss(output: V, target: V, axis=None, keepdims=False) -> V:
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
    return -F.mean(target * F.log(output), axis=axis, keepdims=keepdims)

def _kulldivergence(output: V, target: V, axis=None, keepdims=False) -> V:
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
    return F.mean(target * F.log(target / output), axis=axis, keepdims=keepdims)