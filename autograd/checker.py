from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from .functions import F
from .losses import L
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
        - bound (float): threshold for gradients. Sets gradients to infinity if \
            their absolute value exceed this value. Defaults to 1e3. 
        - all_passed (Optional[bool]): whether all tests passed
        - x (Optional[np.ndarray]): input used
        - calgrads (Optional[np.ndarray]): calculated gradients
        - expgrads (Optional[np.ndarray]): expected gradients
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
        assert self.x is not None, "Assignment Error"
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
        - increment (float): increment to use for finite difference
        - rtol (float): relative tolerance
        - atol (float): absolute tolerance
        - bound (float): threshold for gradients. Sets gradients to infinity if \
            their absolute value exceed this value. Defaults to 1e3. 
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
        randmaps: Optional[list[Callable[[np.ndarray], np.ndarray]]] = None,

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

            randmaps (list[Callable[[float], float]], optional): list of functions to map \
                random numbers from 0 to 1. Defaults to lambda x: x.

            increment (float, optional): increment to use for finite difference. Defaults to 1e-6.
            rtol (float, optional): relative tolerance. Defaults to 1e-3.
            atol (float, optional): absolute tolerance. Defaults to 1e-3.
            bound (float, optional): Threshold for gradients. Sets gradients to infinity if \
            their absolute value exceed this value. Defaults to 1e3. 
        
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
        if randmaps is None:
            randmaps = [lambda x: x] * nargs
        assert len(randmaps) == nargs, "Number of random maps must be equal to number of arguments"

        self.nargs = nargs
        self.epoch = epoch
        self.name = name
        self.dims = dims
        self.randmaps = randmaps

    def stresstest(self) -> bool:
        """
        Run the stress test.

        Returns:
            bool: whether the test passed
        """
        assert self.randmaps is not None, "Random maps not initialised"
        for e in range(self.epoch): 
            args = []
            for i in range(self.nargs):
                nparr = self.randmaps[i](np.random.random(self.dims))
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
    def uniform(s: float, e: float) -> Callable[[np.ndarray], np.ndarray]:
        def f(x: np.ndarray) -> np.ndarray:
            return s + (e - s) * x
        return f
    
    def uniform_exclude(start: float, end: float, exclude: list[float], tolerance = 1e-2) -> Callable[[np.ndarray], np.ndarray]:
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
    
    def where_sin_cos(cond: V, x1: V, x2: V) -> V:
        return F.where(cond > 0.5, F.sin(x1), F.cos(x2))

    funcs = [
        #(Name, Function, Number of Arguments, randmaps) 
        (None,      F.sum,                      1, [uniform(-10.0, 10.0)]),
        (None,      F.mean,                     1, [uniform(-10.0, 10.0)]),
        (None,      F.rms,                      1, [uniform(-10.0, 10.0)]),
        (None,      F.softmax,                  1, [uniform(-10.0, 10.0)]),
        (None,      F.abs,                      1, [uniform(-10.0, 10.0)]),
        (None,      F.sin,                      1, [uniform(-10.0, 10.0)]),
        (None,      F.cos,                      1, [uniform(-10.0, 10.0)]),
        (None,      F.tan,                      1, [uniform(-10.0, 10.0)]),
        (None,      F.relu,                     1, [uniform(-10.0, 10.0)]),
        (None,      F.sinh,                     1, [uniform(-10.0, 10.0)]),
        (None,      F.cosh,                     1, [uniform(-10.0, 10.0)]),
        (None,      F.tanh,                     1, [uniform(-10.0, 10.0)]),
        (None,      F.log,                      1, [uniform_exclude(-10.0, 10.0, [0.0])]),
        (None,      F.elu,                      1, [uniform(-10.0, 10.0)]),
        (None,      F.leakyrelu,                1, [uniform(-10.0, 10.0)]),
        (None,      where_sin_cos,              3, [uniform(0.0, 1.0), uniform(-10.0, 10.0), uniform(-10.0, 10.0)]),
        ("-x",      lambda x: -x,               1, [uniform(-10.0, 10.0)]), 
        ("x+y",     lambda x, y: x + y,         2, [uniform(-10.0, 10.0), uniform(-10.0, 10.0)]),
        ("x-y",     lambda x, y: x - y,         2, [uniform(-10.0, 10.0), uniform(-10.0, 10.0)]),
        ("x*y",     lambda x, y: x * y,         2, [uniform(-10.0, 10.0), uniform(-10.0, 10.0)]),
        ("x/y",     lambda x, y: x / y,         2, [uniform(-10.0, 10.0), uniform_exclude(-10.0, 10.0, [0.0])]),
        ("x**y",    lambda x, y: x ** y,        2, [uniform(-10.0, 10.0), uniform(-5.0, 5.0)]),
        ("x@y",     lambda x, y: x @ y,         2, [uniform(-10.0, 10.0), uniform(-10.0, 10.0)]),
        (None,      L.crossentropyloss,         2, [uniform(0.0, 1.0), uniform(0.0, 1.0)]),
        (None,      L.kulldivergence,           2, [uniform(0.0, 1.0), uniform(0.0, 1.0)]),
        (None,      L.l1loss,                   2, [uniform(-10.0, 10.0), uniform(-10.0, 10.0)]),
        (None,      L.l2loss,                   2, [uniform(-10.0, 10.0), uniform(-10.0, 10.0)]),
        (None,      L.rmsloss,                  2, [uniform(-10.0, 10.0), uniform(-10.0, 10.0)]),
        (None,      L.huberloss,                2, [uniform(-10.0, 10.0), uniform(-10.0, 10.0)]),
    ]
    for name, func, nargs, randmaps in funcs:
        name = name or func.__name__
        checker = FunctionChecker(func, nargs, name=name, randmaps=randmaps)
        res = checker.stresstest()
        print(checker)
        if not res:
            print(checker.dump())
            print("Prematurely Terminating... Failed on function: ", name) 
            return

if __name__ == '__main__':
    test_all_functions()
    