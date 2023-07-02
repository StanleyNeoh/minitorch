from typing import Optional, Callable, Iterator
import numpy as np
import matplotlib.pyplot as plt # type: ignore

from autograd import F, V, L
from model import Model, O, Ly

from .utils import uniform_data_generator, GraphPlotter1D

# Regression Test Functions
def regression0(x):
    return x * 5.0 + 3.0

def regression1(x):
    return x * x - x * 3.0 + 2.0

def regression2(x):
    return x * x * x - x * x * 10.0 - x * 50.0 + 10.0

def regression3(x):
    return F.sin(x) * 2.0 + x * 0.5 + 1.0

def regression4(x):
    return x * F.sin(x) * 2.0 + x

def regression5(x):
    return (V.of(2) ** ((x ** 2) * -0.01)) * 10.0 * F.sin(x) + x

def regression6(x):
    return F.where(
        x < -10.0,
        (x ** 2) * 0.1 + x - 10.0, 
        F.where(
            x < 10.0, 
            F.sin(x) * x,
            F.cos(x) * 5.0 - x
        ))

regression_model = Ly.Sequential(
    Ly.Linear(1, 100),
    Ly.ReLU(),
    Ly.Linear(100, 1)
)

regression_loss = [
    ("Adam_l1loss", O.Adam, L.l1loss),
    ("Adam_l2loss", O.Adam, L.l2loss),
    ("Adam_rmsloss", O.Adam, L.rmsloss),
    ("Adam_huberloss", O.Adam, L.huberloss),
]

optimisers = [
    ("SGD_huberloss", O.SGD, L.huberloss),
    ("Momentum_huberloss", O.Momentum, L.huberloss),
    ("Adagrad_huberloss", O.Adagrad, L.huberloss),
    ("RMSProp_huberloss", O.RMSProp, L.huberloss),
    ("Adam_huberloss", O.Adam, L.huberloss),
]

regressions_reference = [
    ("regression0", regression0), 
    ("regression1", regression1), 
    ("regression2", regression2), 
    ("regression3", regression3), 
    ("regression4", regression4), 
    ("regression5", regression5), 
    ("regression6", regression6)
]

def regression_model_test():
    num_epoch = 1000

    print("Regression Model Test: Loss")
    for refname, ref in regressions_reference:
        generator = uniform_data_generator(ref, (100, 1))
        title = f"{refname}_epoch-{num_epoch}"
        GraphPlotter1D.queue_graph_plot(ref, label="reference")
        for name, optimiser, loss in regression_loss:
            full_name = f"{name}_{refname}"
            model = Model.derived_from(regression_model, optimiser, loss)
            model.train(generator, num_epoch)
            print(full_name, model.statistics())

            GraphPlotter1D.queue_graph_plot(model, label=full_name, ls="--")
        GraphPlotter1D.save_graph_plot(title, f"blob/regression/loss/{refname}.png")
        GraphPlotter1D.clear_graph_plot()
    
    print("Regression Model Test: Optimiser")
    for refname, ref in regressions_reference:
        generator = uniform_data_generator(ref, (100, 1))
        title = f"{refname}_epoch-{num_epoch}"
        GraphPlotter1D.queue_graph_plot(ref, label="reference")
        for name, optimiser, loss in optimisers:
            full_name = f"{name}_{refname}"
            model = Model.derived_from(regression_model, optimiser, loss)
            model.train(generator, num_epoch)
            print(full_name, model.statistics())

            GraphPlotter1D.queue_graph_plot(model, label=full_name, ls="--")
        GraphPlotter1D.save_graph_plot(title, f"blob/regression/optimiser/{refname}.png")
        GraphPlotter1D.clear_graph_plot()

    print("Regression Model Test: Loss History")
    ref = regression6
    generator = uniform_data_generator(ref, (100, 1))
    title = f"Loss_epoch-{num_epoch}"
    for name, optimiser, loss in optimisers:
        full_name = f"{name}"
        model = Model.derived_from(regression_model, optimiser, loss)
        model.train(generator, num_epoch)
        model.queue_loss_plot(full_name)
        print(full_name, model.statistics())
    Model.save_loss_plot(f"blob/regression/loss_history/{title}.png")

if __name__ == "__main__":
    regression_model_test()