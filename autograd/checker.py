from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Callable, Optional

from .functions import F
from .variable import V

class GradResult:
    """
    A class to store the result of gradient checking.

    Attributes:
        - i (int): index of result 
        - x (float): value of input used
        - calgrad (float): calculated gradient
        - expgrad (float): expected gradient
        - passed (bool): whether the test passed
        - remarks (str): remarks of the test 
    """
    def __init__(
            self,
            i: int,
            x: float,
            calgrad: float,
            expgrad: float,
            passed: bool,
            remarks: Optional[str] = None,
            ) -> None:
        """
        Initialize a result.

        Args:
            i (int): index of result
            x (float): value of input used
            calgrad (float): calculated gradient
            expgrad (float): expected gradient
            passed (bool): whether the test passed
            remarks (str, optional): remarks of the test. Defaults to None.
        Returns:
            None
        """
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
        """
        Return the header of the result table.

        Returns:
            str: header of the result table
        """
        return f"{'i':<3} {'x':<15} {'cal':<15} {'exp':<15} {'diff':<15} {'passed':<10} {'remarks':<15}"

    def __str__(self) -> str:
        s = ""
        if not self.passed:
            s = f"{abs(self.calgrad) > 1e6 and abs(self.expgrad) > 1e6}"
        return f"{self.i:<3} {self.x:<15.5f} {self.calgrad:<15.5f} {self.expgrad:<15.5f} {self.calgrad - self.expgrad:<15.5f} {str(self.passed):<10} {self.remarks}"

class GradChecker:
    """
    A class to check the gradient of a function, given a model.

    Attributes:
        - model (Callable[..., V]): model to check
        - increment (float): increment to use for finite difference
        - rtol (float): relative tolerance
        - atol (float): absolute tolerance
        - bound (float): bound for finite difference
        - all_passed (Optional[bool]): whether all tests passed
        - x (Optional[npt.NDArray]): input used
        - calgrads (Optional[npt.NDArray]): calculated gradients
        - expgrads (Optional[npt.NDArray]): expected gradients
        - results (Optional[list[GradResult]]): results of the test
    """
    def __init__(
            self,
            model: Callable[..., V],
            increment: float = 1e-6,
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
            bound (float, optional): bound for finite difference. Defaults to 1e3.
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
        self.x: Optional[npt.NDArray] = None
        self.calgrads: Optional[npt.NDArray] = None
        self.expgrads: Optional[npt.NDArray] = None
        self.results: Optional[list[GradResult]] = None

    def evaluate(self, xs: list[V]) -> bool: 
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
        Returns:    
            bool: whether all tests passed
        """
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
        """
        Return the results of the last test in a table.

        Returns:
            str: table of the results
        """
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
    """
    A class to check the gradient of a function, given a model.
    Subclass of GradChecker and specialises in stress testing the calculation of gradients
    by generating random inputs.

    Attributes:
        - function (Callable[..., V]): function to check
        - nargs (int): number of arguments to the function
        - epoch (int): number of tests to run
        - name (Optional[str]): name of the function
        - dims (tuple): dimensions of the random inputs
        - diameter (float): diameter of the random inputs
        - nonzero (Optional[float]): minimum value of the random inputs
        - increment (float): increment to use for finite difference
        - rtol (float): relative tolerance
        - atol (float): absolute tolerance
        - bound (float): bound for finite difference
    """
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
        """
        Initialise the FunctionChecker with the given parameters.

        Args:
            function (Callable[..., V]): function to check
            nargs (int): number of arguments to the function
            epoch (int, optional): number of tests to run. Defaults to 50.
            name (Optional[str], optional): name of the function. Defaults to None.
            dims (tuple, optional): dimensions of the random inputs. Defaults to (5, 5).
            diameter (float, optional): diameter of the random inputs. Defaults to 10.0.
            nonzero (Optional[float], optional): value to replace zero values of test inputs. Defaults to None.
            increment (float, optional): increment to use for finite difference. Defaults to 1e-6.
            rtol (float, optional): relative tolerance. Defaults to 1e-3.
            atol (float, optional): absolute tolerance. Defaults to 1e-3.
            bound (float, optional): bound for finite difference. Defaults to 1e3.
        
        Returns:
            None
        """
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
        """
        Run the stress test.

        Returns:
            bool: whether the test passed
        """
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
    """
    Test all functions
    """
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
        res = checker.stresstest()
        print(checker)
        if not res:
            print(checker.dump())
            print("Prematurely Terminating... Failed on function: ", name) 
            return

if __name__ == '__main__':
    test_all_functions()
    