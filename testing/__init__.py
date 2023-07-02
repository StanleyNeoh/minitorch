if __name__ == "__main__":
    from .autograd import autograd_test
    from .model import model_test
    autograd_test()
    model_test()