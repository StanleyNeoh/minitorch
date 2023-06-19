from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Callable, Optional

from .functions import F
from .variable import V

class GradResult:
    def __init__(
            self,
            i: int,
            x: float,
            calgrad: float,
            expgrad: float,
            passed: bool,
            remarks: Optional[str] = None,
            ) -> None:
        self.i = i
        self.x = x
        self.calgrad = calgrad
        self.expgrad = expgrad
        self.passed = passed
        if remarks is None:
            remarks = "NA"
        self.remarks = remarks
    
    @classmethod
    def header(cls) -> str:
        return f"{'i':<3} {'x':<15} {'cal':<15} {'exp':<15} {'diff':<15} {'passed':<10} {'remarks':<15}"

    def __str__(self) -> str:
        s = ""
        if not self.passed:
            s = f"{abs(self.calgrad) > 1e6 and abs(self.expgrad) > 1e6}"
        return f"{self.i:<3} {self.x:<15.5f} {self.calgrad:<15.5f} {self.expgrad:<15.5f} {self.calgrad - self.expgrad:<15.5f} {str(self.passed):<10} {self.remarks}"

class GradChecker:
    def __init__(
            self,
            model: Callable[..., V],
            increment: float = 1e-6,
            rtol: float = 1e-3,
            atol: float = 1e-3,
            bound: float = 1e3,
            ) -> None:
        self.model = model
        self.increment = increment
        self.rtol = rtol
        self.atol = atol
        self.bound = bound

        # Test Statuses
        self.all_passed: Optional[bool] = None
        self.x: Optional[npt.NDArray] = None
        self.calgrads: Optional[npt.NDArray] = None
        self.expgrads: Optional[npt.NDArray] = None
        self.results: Optional[list[GradResult]] = None

    def evaluate(self, xs: list[V]) -> bool: 
        out = self.model(*xs)
        out.backward()
        refy = out.item()

        # Obtaining Gradients
        npxs = np.stack([x.data for x in xs])
        shape = npxs.shape
        self.x = npxs.flatten() 
        self.calgrads = np.stack([x.grad for x in xs]).flatten()
        grads = [] 
        for i in range(self.x.size):
            testx = self.x.copy()
            testx[i] += self.increment
            testx = testx.reshape(shape)
            x = [V.of(r, requires_grad=True) for r in testx] # Iterating over axis 0
            y = self.model(*x).item()
            grads.append((y - refy) / self.increment)
        self.expgrads = np.array(grads)

        # Comparing Gradients
        self.results = []
        self.all_passed = True
        assert self.expgrads is not None and self.calgrads is not None and self.x is not None, "Assignment Error"
        for i, (expected, calculated, xf) in enumerate(zip(self.expgrads, self.calgrads, self.x)):
            passed = True
            remark = None
            if abs(expected) > self.bound and abs(calculated) > self.bound:
                passed = True
                remark = "Both gradients are too large"
            elif expected > self.bound or calculated > self.bound:
                passed = False
                remark = "Only one of the gradients are too large"
            else:
                passed = np.isclose(expected, calculated, rtol=self.rtol, atol=self.atol)
            self.all_passed = self.all_passed and passed
            self.results.append(GradResult(i, xf, calculated, expected, passed, remark))
        return self.all_passed
    
    def dump(self) -> str:
        if self.results is None:
            return str(self) 
        return GradResult.header() + "\n" + "\n".join([str(r) for r in self.results])

    def __str__(self) -> str:
        if self.all_passed is None:
            return "GradChecker: Not evaluated yet"
        elif self.all_passed: 
            return "GradChecker: All passed"
        else:
            return "GradChecker: Failed"

class FunctionChecker(GradChecker):
    merger = F.sum
    def __init__(
        self,
        function: Callable[..., V],
        nargs: int,
        epoch: int = 50,
        name: Optional[str] = None,
        dims: tuple = (5, 5),
        
        # Random Options 
        diameter: float = 10.0,
        nonzero: Optional[float] = None,

        # GradChecker Arguments
        increment: float = 1e-6,
        rtol: float = 1e-3,
        atol: float = 1e-3,
        bound: float = 1e3,
    ):
        def model(*xs: list[V]) -> V:
            x = function(*xs)
            x = FunctionChecker.merger(x)
            return x
        super().__init__(
            model=model,
            increment=increment, 
            rtol=rtol, 
            atol=atol,
            bound=bound
            )
        if name is None:
            name = function.__name__
        self.name = name
        self.nargs = nargs
        self.diameter = diameter
        self.dims = dims
        self.nonzero = nonzero
        self.epoch = epoch

    def stresstest(self) -> bool:
        for e in range(self.epoch): 
            args = []
            for i in range(self.nargs):
                nparr = self.diameter * (np.random.random(self.dims) - 0.5)
                if self.nonzero is not None:
                    nparr[nparr == 0] = self.nonzero
                args.append(V.of(nparr, requires_grad=True))
            self.evaluate(args)
            if not self.all_passed:
                return False 
        return True
    
    def __str__(self) -> str:
        if self.all_passed is None:
            return f"Function Checker: {self.name} not evaluated yet"
        elif self.all_passed:
            return f"Function Checker: {self.name} passed"
        else:
            return f"Function Checker: {self.name} failed"

def test_all_functions():
    funcs = [
        #(Name, Function, Number of Arguments, nonzero replacement)
        (None, F.sum, 1, None),
        (None, F.mean, 1, None),
        (None, F.softmax, 1, None),
        (None, F.sin, 1, None),
        (None, F.cos, 1, None),
        (None, F.tan, 1, None),
        (None, F.relu, 1, None),
        (None, F.sinh, 1, None),
        (None, F.cosh, 1, None),
        (None, F.tanh, 1, None),
        (None, F.log, 1, None),
        ("-x", lambda x: -x, 1, None),
        ("x+y", lambda x, y: x + y, 2, None),
        ("x-y", lambda x, y: x - y, 2, None),
        ("x*y", lambda x, y: x * y, 2, None),
        ("x/y", lambda x, y: x / y, 2, 1e-6),
        ("x**y", lambda x, y: x ** y, 2, None),
    ]
    for name, func, nargs, nonzero in funcs:
        name = name or func.__name__
        checker = FunctionChecker(func, nargs, name=name, nonzero=nonzero)
        res = checker.evaluate()
        print(checker)
        if not res:
            print(checker.dump())
            print("Prematurely Terminating... Failed on function: ", name) 
            return

if __name__ == '__main__':
    test_all_functions()
    