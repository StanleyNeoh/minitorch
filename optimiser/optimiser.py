import numpy as np

from autograd.variable import V # type: ignore

class Optimiser:
    def __init__(self, *parameters: V):
        self.parameters = parameters

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = np.zeros_like(parameter.data)  
    
    def step(self):
        raise NotImplementedError
    
    def __repr__(self):
        return '{}(parameters={})'.format(self.__class__, self.parameters)
    
    def __str__(self):
        return str(self.__repr__())

class SGD(Optimiser):
    def __init__(self, *parameters: V, lr: float = 0.01):
        super().__init__(*parameters)
        self.lr = lr
    
    def step(self):
        for parameter in self.parameters:
            parameter.data = parameter.data - (parameter.grad * self.lr)

class Adagrad(Optimiser):
    def __init__(self, *parameters: V, lr: float = 0.01, eps: float = 1e-8):
        super().__init__(*parameters)
        self.lr = lr
        self.eps = eps
        self.cache = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def step(self):
        for parameter, cache in zip(self.parameters, self.cache):
            cache = cache + parameter.grad ** 2
            parameter.data = parameter.data - (self.lr * parameter.grad / (np.sqrt(cache) + self.eps))

class Adam(Optimiser):
    def __init__(self, *parameters: V, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(*parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(parameter.data) for parameter in self.parameters]
        self.v = [np.zeros_like(parameter.data) for parameter in self.parameters]
        self.t = 0
    
    def step(self):
        self.t += 1
        for parameter, m, v in zip(self.parameters, self.m, self.v):
            m = self.beta1 * m + (1 - self.beta1) * parameter.grad
            v = self.beta2 * v + (1 - self.beta2) * parameter.grad ** 2
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            parameter.data = parameter.data - (self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
