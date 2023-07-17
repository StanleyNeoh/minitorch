import numpy as np
from typing import Callable, Iterator

from autograd import V


def gen_sample_V_V(
    reference: Callable[[V], V],
    shape: tuple[int, ...] = (100, 1),
    start: float = -10.0,
    end: float = 10.0,
) -> Iterator[tuple[V, V]]:
    """
    Generate uniform data for testing.

    Args:
        reference (Callable[[V], V]): reference function to generate output data
        shape (tuple[int, ...]): shape of the float data
        start (float): start of the uniform distribution
        end (float): end of the uniform distribution

    Returns:
        Iterator[tuple[V, V]]: Iterator that gives float and output data
    """
    while True:
        x = V.uniform(shape, start, end)
        y = reference(x)
        yield x, y


def gen_float_V(
    start: float,
    end: float,
    shape: tuple[int, ...] = (5, 5),
    requires_grad: bool = True,
) -> Iterator[V]:
    """
    Generate uniform float data for testing.

    Args:
        start (float): start of the uniform distribution
        end (float): end of the uniform distribution
        shape (tuple[int, ...]): shape of the float data. Defaults to (5, 5).
        requires_grad (bool): whether the float data requires gradient. Defaults to True.

    Returns:
        Iterator[V]: Iterator that gives float data
    """
    while True:
        yield V.uniform(shape, start, end, requires_grad=requires_grad)


def gen_float_ex_V(
    start: float,
    end: float,
    exclude: list[float],
    shape: tuple[int, ...] = (5, 5),
    tolerance: float = 1e-2,
    requires_grad: bool = True,
) -> Iterator[V]:
    """
    Generate uniform float data for testing.
    The generated data is not in the exclude list
    with some tolerance.

    Args:
        start (float): start of the uniform distribution
        end (float): end of the uniform distribution
        exclude (list[float]): list of values to exclude
        shape (tuple[int, ...]): shape of the float data. Defaults to (5, 5).
        tolerance (float): tolerance for excluding values. Defaults to 1e-2.
        requires_grad (bool): whether the float data requires gradient. Defaults to True.

    Returns:
        Iterator[V]: Iterator that gives float data
    """
    while True:
        x = V.uniform(shape, start, end, requires_grad=requires_grad)
        assert isinstance(x.data, np.ndarray), "x.data must be a numpy array"
        passed = True
        for e in exclude:
            if np.any(abs(x.data - e) < tolerance):
                passed = False
                break
        if passed:
            yield x


def gen_index_NP(
    n: int,
    num_batches: int = 5,
) -> Iterator[np.ndarray]:
    """
    Generate uniform index data for testing.

    Args:
        n (int): number of indices
        num_batches (int): number of batches. Defaults to 5.

    Returns:
        Iterator[np.ndarray]: Iterator that gives index data
    """
    while True:
        yield np.random.randint(0, num_batches, n)

def gen_float_NP(
    start: float,
    end: float,
    shape: tuple[int, ...] = (5, 5),
) -> Iterator[np.ndarray]:
    """
    Generate uniform float data for testing.

    Args:
        start (float): start of the uniform distribution
        end (float): end of the uniform distribution
        shape (tuple[int, ...]): shape of the float data. Defaults to (5, 5).

    Returns:
        Iterator[np.ndarray]: Iterator that gives float data
    """
    while True:
        yield np.random.uniform(start, end, shape)

def gen_float_ex_NP(
    start: float,
    end: float,
    exclude: list[float],
    shape: tuple[int, ...] = (5, 5),
    tolerance: float = 1e-2,
) -> Iterator[np.ndarray]:
    """
    Generate uniform float data for testing.
    The generated data is not in the exclude list
    with some tolerance.

    Args:
        start (float): start of the uniform distribution
        end (float): end of the uniform distribution
        exclude (list[float]): list of values to exclude
        shape (tuple[int, ...]): shape of the float data. Defaults to (5, 5).
        tolerance (float): tolerance for excluding values. Defaults to 1e-2.

    Returns:
        Iterator[np.ndarray]: Iterator that gives float data
    """
    while True:
        x = np.random.uniform(start, end, shape)
        passed = True
        for e in exclude:
            if np.any(abs(x - e) < tolerance):
                passed = False
                break
        if passed:
            yield x