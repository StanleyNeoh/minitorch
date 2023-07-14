import numpy as np
from typing import Callable, Iterator

from autograd import V


def uniform_data_g(
    reference: Callable[[V], V],
    shape: tuple[int, ...] = (100, 1),
    start: float = -10.0,
    end: float = 10.0,
) -> Iterator[tuple[V, V]]:
    """
    Generate uniform data for testing.

    Args:
        reference (Callable[[V], V]): reference function to generate output data
        shape (tuple[int, ...]): shape of the input data
        start (float): start of the uniform distribution
        end (float): end of the uniform distribution
    
    Returns:
        Iterator[tuple[V, V]]: Iterator that gives input and output data
    """
    while True:
        x = V.uniform(shape, start, end)
        y = reference(x)
        yield x, y


def uniform_input_g(
    start: float, 
    end: float,
    shape: tuple[int, ...]=(5, 5),
    requires_grad: bool = True,
) -> Iterator[V]:
    """
    Generate uniform input data for testing.

    Args:
        start (float): start of the uniform distribution
        end (float): end of the uniform distribution
        shape (tuple[int, ...]): shape of the input data. Defaults to (5, 5).
        requires_grad (bool): whether the input data requires gradient. Defaults to True.

    Returns:
        Iterator[V]: Iterator that gives input data
    """
    while True:
        yield V.uniform(shape, start, end, requires_grad=requires_grad)

def uniform_ex_input_g(
    start: float,
    end: float,
    exclude: list[float],
    shape: tuple[int, ...]=(5, 5),
    tolerance: float = 1e-2,
    requires_grad: bool = True,
) -> Iterator[V]:
    """
    Generate uniform input data for testing. 
    The generated data is not in the exclude list
    with some tolerance.

    Args:
        start (float): start of the uniform distribution
        end (float): end of the uniform distribution
        exclude (list[float]): list of values to exclude
        shape (tuple[int, ...]): shape of the input data. Defaults to (5, 5).
        tolerance (float): tolerance for excluding values. Defaults to 1e-2.
        requires_grad (bool): whether the input data requires gradient. Defaults to True.

    Returns:
        Iterator[V]: Iterator that gives input data
    """
    while True:
        x = V.uniform(shape, start, end, requires_grad=requires_grad)
        passed = True 
        for e in exclude:
            if np.any(abs(x.data - e) < tolerance):
                passed = False
                break
        if passed:
            yield x
