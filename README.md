# Minitorch

> "Hmm, I wonder how PyTorch works? I should write a toy version of it to find out."

Minitoch is a minimal recreation of Pytorch. It is intended for me to learn how a Machine Learning
framework works, and to serve as a teaching tool for others. It is not intended to be fast
or to be used for anything serious (at least not yet).

Stay tuned for updates!

## Installation

Currently Minitorch is not available on PyPI. To install it, clone the repository and run

```bash
git clone git@github.com:StanleyNeoh/minitorch.git
cd minitorch
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Features implemented
- [x] Variable wrapper around numpy
- [x] Operations between Variable
- [x] Autograd
- [x] Autograd Checker
- [x] Losses
- [x] Optimizers
- [x] Layers
- [ ] Datasets
- [ ] Dataloaders
- [ ] Training loop
- [ ] Model saving/loading
- [ ] GPU support
- [ ] Distributed training
- [ ] ONNX export
- [ ] JIT compilation
- [ ] C++ backend
