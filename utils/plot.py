from typing import Callable, TypeVar
import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from autograd import V
from model import Model

T = TypeVar("T")


class PlotCanvas:
    def __init__(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        transformer: Callable[[T], tuple[np.ndarray, np.ndarray]],
    ):
        self.queue: list[tuple[T, dict]] = []
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.transformer = transformer

    def reset(self):
        self.queue.clear()

    def add(self, plot: T, **kwargs):
        self.queue.append((plot, kwargs))

    def build(self, path: str):
        print("Building plot at '{}'".format(path))
        plt.clf()
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        for plot, kwargs in self.queue:
            plt.plot(*self.transformer(plot), **kwargs)
        plt.legend()
        plt.savefig(path)
        self.reset()
        plt.clf()


class RegressionPlotCanvas(PlotCanvas):
    def __init__(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        minX: float = -10.0,
        maxX: float = 10.0,
        step: float = 0.1,
    ):
        def transformer(f: Callable[[V], V]) -> tuple[np.ndarray, np.ndarray]:
            x = V(np.arange(minX, maxX, step).reshape(-1, 1))
            y = f(x)
            assert isinstance(x.data, np.ndarray) and isinstance(y.data, np.ndarray)
            return x.data, y.data

        super().__init__(
            title=title, xlabel=xlabel, ylabel=ylabel, transformer=transformer
        )


class LossHistoryPlotCanvas(PlotCanvas):
    def __init__(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
    ):
        def transformer(model: Model) -> tuple[np.ndarray, np.ndarray]:
            return np.arange(0, len(model.loss_history)), np.array(model.loss_history)

        super().__init__(
            title=title, xlabel=xlabel, ylabel=ylabel, transformer=transformer
        )
