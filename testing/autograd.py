from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Optional, Iterator

from autograd import F, V, L

from utils import uniform_ex_input_g, uniform_input_g


class GradChecker:
    """
    A class to check the gradient of a function, given a model.
    """

    def __init__(
        self,
        model: Callable[..., V],
        increment: float = 1e-5,
        rtol: float = 1e-3,
        atol: float = 1e-3,
        bound: float = 1e3,
    ) -> None:
        """
        Initialize a gradient checker with a model.

        Args:
            model (Callable[..., V]): model to check
            increment (float, optional): increment to use for finite difference. Defaults to 1e-6.
            rtol (float, optional): relative tolerance. Defaults to 1e-3.
            atol (float, optional): absolute tolerance. Defaults to 1e-3.
            bound (float, optional): Threshold for gradients. Sets gradients to infinity if \
            their absolute value exceed this value. Defaults to 1e3. 
        Returns:
            None
        """
        self.model = model
        self.increment = increment
        self.rtol = rtol
        self.atol = atol
        self.bound = bound

        # Test Statuses
        self.all_passed: Optional[bool] = None
        self.x: Optional[np.ndarray] = None
        self.calgrads: Optional[np.ndarray] = None
        self.expgrads: Optional[np.ndarray] = None
        self.results: Optional[pd.DataFrame] = None

    def evaluate(self, xs: tuple[V], initial=2.0) -> bool:
        """
        Evaluate the model with the given inputs and check the gradients.
        Small adjustments are made to the inputs to approximate the actual gradients
        which are then compared with the calculated gradients.
        Populates the following attributes of the class.
        * all_passed
        * x
        * calgrads
        * expgrads
        * results

        Args:
            xs (list[V]): inputs to evaluate the model with
            initial (float, optional): initial gradient. Defaults to 2.0.
        Returns:
            bool: whether all tests passed
        """
        out = self.model(*xs)
        out.backward(initial=initial)
        refy = out.item()

        # Obtaining Gradients
        npxs = np.stack([x.data for x in xs])
        shape = npxs.shape
        self.x = npxs.flatten()
        self.calgrads = np.stack([x.grad / initial for x in xs]).flatten()
        grads = []
        assert self.x is not None, "Assignment Error"
        for i in range(self.x.size):
            testx = self.x.copy()
            testx[i] += self.increment
            testx = testx.reshape(shape)
            x = [V.of(r, requires_grad=True) for r in testx]  # Iterating over axis 0
            y = self.model(*x).item()
            grads.append((y - refy) / self.increment)
        self.expgrads = np.array(grads)

        # Comparing Gradients
        assert (
            self.expgrads is not None
            and self.calgrads is not None
            and self.x is not None
        ), "Assignment Error"
        self.results = pd.DataFrame(
            {
                "x": self.x,
                "cal": self.calgrads,
                "exp": self.expgrads,
            },
        )
        self.results["diff"] = self.results["cal"] - self.results["exp"]
        self.results["valid"] = self.results.apply(
            lambda row: abs(row["cal"]) < self.bound and abs(row["exp"]) < self.bound,
            axis=1,
        )
        self.results["passed"] = self.results.apply(
            lambda row: (
                not row["valid"]
                or np.isclose(row["cal"], row["exp"], rtol=self.rtol, atol=self.atol)
            ),
            axis=1,
        )
        self.all_passed = self.results["passed"].all()
        return self.all_passed

    def __str__(self) -> str:
        if self.all_passed is None:
            return "GradChecker: Not evaluated yet"
        elif self.all_passed:
            return "GradChecker: All passed"
        else:
            return "GradChecker: Failed"


class FunctionChecker(GradChecker):
    """
    A class to check the gradient of a function, given a model.
    Subclass of GradChecker and specialises in stress testing the calculation of gradients
    by generating random inputs.
    """

    merger = F.sum

    def __init__(
        self,
        function: Callable[..., V],
        generators: list[Iterator[V]],
        name: str,
        epoch: int = 50,
    ):
        """
        Initialize a function checker with a function. GradChecker parameters are set to default.

        Args:
            function (Callable[..., V]): function to check
            generator (Callable[[], tuple[V, ...]]): function to generate random inputs
            name (str): name of the function
            epoch (int, optional): number of tests to run. Defaults to 50.

        Returns:
            None
        """

        def model(*xs: list[V]) -> V:
            x = function(*xs)
            x = FunctionChecker.merger(x)
            return x

        super().__init__(model=model)
        self.name = name
        self.generators = generators
        self.epoch = epoch

    def stresstest(self) -> bool:
        """
        Run the stress test.

        Returns:
            bool: whether the test passed
        """
        for e in range(self.epoch):
            xs = [next(g) for g in self.generators]
            self.evaluate(xs)
            if not self.all_passed:
                return False
        return True

    def __str__(self) -> str:
        if self.all_passed is None:
            return f"FunctionChecker: {self.name} not evaluated yet"
        elif self.all_passed:
            return f"FunctionChecker: {self.name} passed"
        else:
            return f"FunctionChecker: {self.name} failed"


def autograd_test():
    def piecewise(cond: V, x1: V, x2: V) -> V:
        return F.where(cond > 0.5, F.sin(x1), F.cos(x2))

    functionCheckers = [
        FunctionChecker(F.sum, [uniform_input_g(-10.0, 10.0)], "Sum"),
        FunctionChecker(F.mean, [uniform_input_g(-10.0, 10.0)], "Mean"),
        FunctionChecker(F.rms, [uniform_input_g(-10.0, 10.0)], "RMS"),
        FunctionChecker(F.softmax, [uniform_input_g(-10.0, 10.0)], "Softmax"),
        FunctionChecker(F.abs, [uniform_input_g(-10.0, 10.0)], "Abs"),
        FunctionChecker(F.sin, [uniform_input_g(-10.0, 10.0)], "Sin"),
        FunctionChecker(F.cos, [uniform_input_g(-10.0, 10.0)], "Cos"),
        FunctionChecker(F.tan, [uniform_input_g(-10.0, 10.0)], "Tan"),
        FunctionChecker(F.relu, [uniform_input_g(-10.0, 10.0)], "ReLU"),
        FunctionChecker(F.sinh, [uniform_input_g(-10.0, 10.0)], "Sinh"),
        FunctionChecker(F.cosh, [uniform_input_g(-10.0, 10.0)], "Cosh"),
        FunctionChecker(F.tanh, [uniform_input_g(-10.0, 10.0)], "Tanh"),
        FunctionChecker(F.log, [uniform_ex_input_g(-10.0, 10.0, [0.0])], "Log"),
        FunctionChecker(F.elu, [uniform_input_g(-10.0, 10.0)], "ELU"),
        FunctionChecker(F.leakyrelu, [uniform_input_g(-10.0, 10.0)], "LeakyReLU"),
        FunctionChecker(
            piecewise,
            [
                uniform_input_g(0.0, 1.0),
                uniform_input_g(-10.0, 10.0),
                uniform_input_g(-10.0, 10.0),
            ],
            "Piecewise",
        ),
        FunctionChecker(lambda x: -x, [uniform_input_g(-10.0, 10.0)], "Neg"),
        FunctionChecker(
            lambda x, y: x + y,
            [uniform_input_g(-10.0, 10.0), uniform_input_g(-10.0, 10.0)],
            "x+y",
        ),
        FunctionChecker(
            lambda x, y: x - y,
            [uniform_input_g(-10.0, 10.0), uniform_input_g(-10.0, 10.0)],
            "x-y",
        ),
        FunctionChecker(
            lambda x, y: x * y,
            [uniform_input_g(-10.0, 10.0), uniform_input_g(-10.0, 10.0)],
            "x*y",
        ),
        FunctionChecker(
            lambda x, y: x / y,
            [uniform_input_g(-10.0, 10.0), uniform_ex_input_g(-10.0, 10.0, [0.0])],
            "x/y",
        ),
        FunctionChecker(
            lambda x, y: x**y,
            [uniform_input_g(-10.0, 10.0), uniform_input_g(-10.0, 10.0)],
            "x**y",
        ),
        FunctionChecker(
            lambda x, y: x @ y,
            [uniform_input_g(-10.0, 10.0), uniform_input_g(-10.0, 10.0)],
            "x@y",
        ),
        FunctionChecker(
            L.l1loss,
            [uniform_input_g(-10.0, 10.0), uniform_input_g(-10.0, 10.0)],
            "L1Loss",
        ),
        FunctionChecker(
            L.l2loss,
            [uniform_input_g(-10.0, 10.0), uniform_input_g(-10.0, 10.0)],
            "L2Loss",
        ),
        FunctionChecker(
            L.rmsloss,
            [uniform_input_g(-10.0, 10.0), uniform_input_g(-10.0, 10.0)],
            "RMSLoss",
        ),
        FunctionChecker(
            L.huberloss,
            [uniform_input_g(-10.0, 10.0), uniform_input_g(-10.0, 10.0)],
            "HuberLoss",
        ),
    ]
    failed = []
    for functionChecker in functionCheckers:
        res = functionChecker.stresstest()
        print(functionChecker)
        if not res:
            print(functionChecker.results)
            failed.append(functionChecker.name)

    if len(failed) > 0:
        print("Failed on functions: ", failed)
    else:
        print("All tests passed")


if __name__ == "__main__":
    autograd_test()
