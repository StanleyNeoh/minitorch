from typing import Callable
import numpy as np

def uniform(s: float, e: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a function that generates a uniform distribution between s and e.

    Args:
        s (float): Start of the distribution.
        e (float): End of the distribution.
    
    Returns:
        A function that generates a uniform distribution between s and e.
    """
    def f(x: np.ndarray) -> np.ndarray:
        return s + (e - s) * x
    return f

def uniform_exclude(start: float, end: float, exclude: list[float], tolerance = 1e-2) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a function that generates a uniform distribution between s and e, excluding the values in exclude.

    Args:
        s (float): Start of the distribution.
        e (float): End of the distribution.
        exclude (list[float]): Values to exclude.
        tolerance (float, optional): Tolerance for excluding values. Defaults to 1e-2.

    Returns:
        A function that generates a uniform distribution between s and e, excluding the values in exclude.
    """
    def f(x: np.ndarray) -> np.ndarray:
        while True:
            x = uniform(start, end)(x)
            passed = True 
            for e in exclude:
                if np.any(abs(x - e) < tolerance):
                    passed = False
                    break
            if passed:
                return x
    return f