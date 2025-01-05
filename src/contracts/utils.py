import numpy as np


def tensor_product(a, b):
    """Compute tensor product of two states."""
    return np.kron(a, b)


def Y(f):
    """Y combinator for finding fixed points of functions."""

    def Y_inner(x):
        return f(lambda *args: x(x)(*args))

    return Y_inner(Y_inner) 