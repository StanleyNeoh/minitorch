from typing import Optional, Callable, Iterator

from autograd.variable import V
from autograd.functions import F
from autograd.losses import L
from model.optimiser import Optimiser, Adam, SGD, RMSProp
from model.layer import Layer, Model, Linear, ReLU, Sequential 
from .utils import uniform_data_generator

def model_test():
    baseModel = Sequential(
        Linear(1, 100),
        ReLU(),
        Linear(100, 1)
    )

    def f(x):
        return F.sin(x) + x * 0.5
    
    generator = uniform_data_generator(f, (100, 1))
    model = Model.derived_from(baseModel, Adam, L.huberloss)
    model.train(generator, 1000)
    print(model.statistics())

    # model = Model.derived_from(baseModel, SGD, L.huberloss)
    # model.train(generator)
    # print(model.statistics())
# 
    # model = Model.derived_from(baseModel, RMSProp, L.huberloss)
    # model.train(generator)
    # print(model.statistics())

if __name__ == "__main__":
    model_test()