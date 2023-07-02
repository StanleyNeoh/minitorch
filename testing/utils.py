from typing import Callable, Iterator
import numpy as np
import matplotlib.pyplot as plt # type: ignore

from autograd.variable import V

# Distribution Generators
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

# Data Generators
def uniform_data_generator(
    reference: Callable[[V], V],
    shape: tuple[int, ...] = (100, 1),
    start: float = -10.0,
    end: float = 10.0,
    ) -> Iterator[tuple[V, V]]:
    while True:
        x = V.uniform(shape, start, end)
        y = reference(x)
        yield x, y

    
class GraphPlotter1D:
    _graph_plot_queue: list[tuple[Callable[[V], V], dict]] = []

    @classmethod
    def queue_graph_plot(cls, f: Callable[[V], V], **kwargs) -> None:
        cls._graph_plot_queue.append((f, kwargs))
    
    @classmethod
    def clear_graph_plot(cls) -> None:
        plt.clf()
        cls._graph_plot_queue.clear()
    
    @classmethod
    def save_graph_plot(cls, 
                  title: str,  
                  path: str, 
                  minX: float = -10.0, 
                  maxX: float = 10.0, 
                  step: float = 0.1
                  ) -> None: 
        plt.clf()
        plt.title(title)
        for f, kwargs in cls._graph_plot_queue:
            x = V(np.arange(minX, maxX, step).reshape(-1, 1))
            y = f(x)
            plt.plot(x.data, y.data, **kwargs)
        plt.legend()
        plt.savefig(path)
        plt.clf()
        cls._graph_plot_queue.clear()
    
