from typing import Callable, Iterator

from autograd import V

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