from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Callable, Optional

from . import operation as op 
from .variable import V

class GradResult:
    def __init__(
            self,
            i: int,
            x: V,
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
        return f"{'i':<3} {'x':<15} {'ref':<15} {'exp':<15} {'diff':<15} {'passed':<10} {'remarks':<15}"

    def __str__(self) -> str:
        s = ""
        if not self.passed:
            s = f"{abs(self.calgrad) > 1e6 and abs(self.expgrad) > 1e6}"
        return f"{self.i:<3} {self.x:<15.5f} {self.calgrad:<15.5f} {self.expgrad:<15.5f} {self.calgrad - self.expgrad:<15.5f} {str(self.passed):<10} {self.remarks}"

class GradChecker:
    def __init__(
            self,
            model: Callable[[V], V],
            increment: float = 1e-8,
            rtol: float = 1e-5,
            atol: float = 1e-5,
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
        self.results: Optional[GradResult] = None

    def evaluate(self, x) -> bool: 
        out = self.model(x)
        out.backward()
        refy = out.item()

        # Obtaining Gradients
        self.x = x.data.flatten()
        self.calgrads = x.grad.flatten()
        grads = [] 
        for i in range(x.data.size):
            testx = self.x.copy()
            testx[i] += self.increment
            testx= testx.reshape(x.data.shape)
            x = V.of(testx, requires_grad=True)
            y = self.model(x).item()
            grads.append((y - refy) / self.increment)
        self.expgrads = np.array(grads)

        # Comparing Gradients
        self.results = []
        self.all_passed = True
        for i, (expected, calculated, x) in enumerate(zip(self.expgrads, self.calgrads, self.x)):
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
            self.results.append(GradResult(i, x, expected, calculated, passed, remark))
        return self.all_passed
    
    def dump(self) -> GradResult:
        return GradResult.header() + "\n" + "\n".join([str(r) for r in self.results])

    def __str__(self) -> str:
        if self.all_passed is None:
            return "GradChecker: Not evaluated yet"
        elif self.all_passed: 
            return "GradChecker: All passed"
        else:
            return "GradChecker: Failed"

class FunctionChecker(GradChecker):
    merger = op.sum
    def __init__(
        self,
        function: Callable[[V], V],
        diameter: float = 10.0,
        dims: tuple = (10, 10),
        epoch: int = 100,
        # GradChecker Arguments
        increment: float = 1e-8,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        bound: float = 1e3,
    ):
        def model(x):
            x = function(x)
            x = FunctionChecker.merger(x)
            return x
        super().__init__(model, increment, rtol, atol, bound)
        self.name = function.__name__
        self.diameter = diameter
        self.dims = dims
        self.epoch = epoch

    def evaluate(self) -> bool:
        for e in range(self.epoch): 
            x = V.of(self.diameter * (np.random.random(self.dims) - 0.5), requires_grad=True)
            super().evaluate(x)
            if not self.all_passed:
                return False 
        return True
    
    def __str__(self) -> str:
        if self.all_passed is None:
            return "Function Checker: {self.name} not evaluated yet"
        elif self.all_passed:
            return f"Function Checker: {self.name} passed"
        else:
            return f"Function Checker: {self.name} failed"

def test_all_funcs():
    funcs = [
        op.sum,
        op.mean,
        op.softmax,
        op.sin,
        op.cos,
        op.tan,
        op.relu,
        op.sinh,
        op.cosh,
        op.tanh,
        op.log,
    ]
    for func in funcs:
        checker = FunctionChecker(func)
        res = checker.evaluate()
        print(checker)
        if not res:
            print(checker.dump())
            print("Prematurely Terminating... Failed on function: ", func.__name__)
            return

if __name__ == '__main__':
    test_all_funcs()
    # test_func(sinh, epoch=1, dims=10, increment=1e-4, rtol=1e-3, atol=1e-4, verbose=True)

    # x = V.of(-15.89179, requires_grad=True)
    # y = sinh(x)
    # y.backward()
    # print(x.grad)

    # grad_checker(sinh, x,verbose=True)