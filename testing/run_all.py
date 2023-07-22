import os
import sys
sys.path[0] = os.path.abspath(os.path.join(__file__, "..", ".."))
from testing.autograd import autograd_test
from testing.model import model_test
autograd_test()
model_test()