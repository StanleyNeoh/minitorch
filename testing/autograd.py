from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Optional, Iterator

from autograd import F, V, L

from utils import gen_float_V, gen_float_ex_V, gen_index_NP 


class GradChecker:
    """
    A class to check the gradient of a function, given a model.
    """

    def __init__(
        self,
        model: Callable[..., V],
        increment: float = 1e-6,
        rtol: float = 1e-2,
        atol: float = 1e-2,
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
        self.argseti: list[int] = []
        self.argi: list[int] = []
        self.arg: list[float] = []
        self.calgrads: list[np.float128] = []
        self.expgrads: list[np.float128] = []
        self.results: Optional[pd.DataFrame] = None
        self.all_passed: Optional[bool] = None

    def evaluate(self, args: tuple[V, ...], initial=2.0) -> bool:
        """
        Evaluate the model with the given inputs and check the gradients.
        Small adjustments are made to the inputs to approximate the actual gradients
        which are then compared with the calculated gradients.

        Args:
            args (list[V]): inputs to evaluate the model with
            initial (float, optional): initial gradient. Defaults to 2.0.
        Returns:
            bool: whether all tests passed
        """
        out = self.model(*args)
        out.backward(initial=initial)
        refy = out.item()

        # Obtaining Gradients
        self.argseti.clear()
        self.argi.clear()
        self.arg.clear()
        self.calgrads.clear()
        self.expgrads.clear()
        for argi, arg in enumerate(args):
            argseti = 0
            if not isinstance(arg, V) or not arg.requires_grad:
                continue
            nparg = arg.data
            ogarg = nparg.flatten()
            self.calgrads.extend(list(arg.grad.flatten() / initial))
            for i in range(ogarg.size):
                testarg = ogarg.copy()
                testarg[i] += self.increment
                testarg = testarg.reshape(nparg.shape)
                x = args[:argi] + (V.of(testarg, requires_grad=True),) + args[argi + 1 :]
                y = self.model(*x).item()

                self.argseti.append(argseti)
                argseti += 1
                self.argi.append(argi)
                self.arg.append(ogarg[i])
                self.expgrads.append((y - refy) / self.increment)

        # Comparing Gradients
        self.results = pd.DataFrame(
            {
                "argseti": self.argseti,
                "argi": self.argi,
                "arg": self.arg,
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
        epoch: int = 100,
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
            xs = tuple(next(g) for g in self.generators)
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
        FunctionChecker(F.sum, [gen_float_V(-10.0, 10.0)], "Sum"),
        FunctionChecker(F.mean, [gen_float_V(-10.0, 10.0)], "Mean"),
        FunctionChecker(F.rms, [gen_float_V(-10.0, 10.0)], "RMS"),
        FunctionChecker(F.softmax, [gen_float_V(-10.0, 10.0)], "Softmax"),
        FunctionChecker(F.abs, [gen_float_V(-10.0, 10.0)], "Abs"),
        FunctionChecker(F.sin, [gen_float_V(-10.0, 10.0)], "Sin"),
        FunctionChecker(F.cos, [gen_float_V(-10.0, 10.0)], "Cos"),
        FunctionChecker(F.tan, [gen_float_V(-10.0, 10.0)], "Tan"),
        FunctionChecker(F.relu, [gen_float_V(-10.0, 10.0)], "ReLU"),
        FunctionChecker(F.sinh, [gen_float_V(-10.0, 10.0)], "Sinh"),
        FunctionChecker(F.cosh, [gen_float_V(-10.0, 10.0)], "Cosh"),
        FunctionChecker(F.tanh, [gen_float_V(-10.0, 10.0)], "Tanh"),
        FunctionChecker(F.log, [gen_float_ex_V(-10.0, 10.0, [0.0])], "Log"),
        FunctionChecker(F.elu, [gen_float_V(-10.0, 10.0)], "ELU"),
        FunctionChecker(F.leakyrelu, [gen_float_V(-10.0, 10.0)], "LeakyReLU"),
        FunctionChecker(
            piecewise,
            [
                gen_float_V(0.0, 1.0),
                gen_float_V(-10.0, 10.0),
                gen_float_V(-10.0, 10.0),
            ],
            "Piecewise",
        ),
        FunctionChecker(lambda x: -x, [gen_float_V(-10.0, 10.0)], "-x"),
        FunctionChecker(
            lambda x, y: x + y,
            [gen_float_V(-10.0, 10.0), gen_float_V(-10.0, 10.0)],
            "x+y",
        ),
        FunctionChecker(
            lambda x, y: x - y,
            [gen_float_V(-10.0, 10.0), gen_float_V(-10.0, 10.0)],
            "x-y",
        ),
        FunctionChecker(
            lambda x, y: x * y,
            [gen_float_V(-10.0, 10.0), gen_float_V(-10.0, 10.0)],
            "x*y",
        ),
        FunctionChecker(
            lambda x, y: x / y,
            [gen_float_V(-10.0, 10.0), gen_float_ex_V(-10.0, 10.0, [0.0])],
            "x/y",
        ),
        FunctionChecker(
            lambda x, y: x**y,
            [gen_float_V(-10.0, 10.0), gen_float_V(-5.0, 5.0)],
            "x**y",
        ),
        FunctionChecker(
            lambda x, y: x @ y,
            [gen_float_V(-10.0, 10.0), gen_float_V(-10.0, 10.0)],
            "x@y",
        ),
        FunctionChecker(
            L.l1loss,
            [gen_float_V(-10.0, 10.0), gen_float_V(-10.0, 10.0, requires_grad=False)],
            "L1Loss",
        ),
        FunctionChecker(
            L.l2loss,
            [gen_float_V(-10.0, 10.0), gen_float_V(-10.0, 10.0, requires_grad=False)],
            "L2Loss",
        ),
        FunctionChecker(
            L.rmsloss,
            [gen_float_V(-10.0, 10.0), gen_float_V(-10.0, 10.0, requires_grad=False)],
            "RMSLoss",
        ),
        FunctionChecker(
            L.huberloss,
            [gen_float_V(-10.0, 10.0), gen_float_V(-10.0, 10.0, requires_grad=False)],
            "HuberLoss",
        ),
        FunctionChecker(
            L.crossentropyloss,
            [gen_float_V(0.0, 1.0), gen_index_NP(5)],
            "CrossEntropyLoss",
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
