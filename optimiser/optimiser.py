import numpy as np

from autograd.variable import V # type: ignore

class Optimiser:
    def __init__(self, *parameters: V):
        self.parameters = parameters

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = np.zeros_like(parameter.data)  
    
    def step(self, loss: V):
        raise NotImplementedError
    
    def __repr__(self):
        return '{}(parameters={})'.format(self.__class__, self.parameters)
    
    def __str__(self):
        return str(self.__repr__())

class SGD(Optimiser):
    def __init__(self, *parameters: V, lr: float = 0.01):
        super().__init__(*parameters)
        self.lr = lr
    
    def step(self, loss: V):
        for parameter in self.parameters:
            print(parameter.name, "===>") 
            print("Grad:", parameter.grad)
            print("Data:", parameter.data)
            print("Loss:", loss.data)
            parameter.data = parameter.data - parameter.grad * self.lr * loss.item()
    
