import numpy as np
class _errfn(object):
    """ Abstract factory base class for any generalized error function."""
    def calc(cost, h, y):
        h = np.matrix(h)
        y = np.matrix(y)
        return cost._calc(h, y)
    @classmethod
    def grad(cost, h, y):
        h = np.matrix(h)
        y = np.matrix(y)
        return cost._grad(h, y)

    def _calc(cost, h, y):
        raise Exception("_calc not implemented")

    def _grad(cost, h, y):
        raise Exception("_grad not implmented")

class squareError(_errfn):
    """
    squared error function: (1/2)(h - y)^2
    """
    def _calc(cost, h, y):
        return 0.5 * np.power(h-y, 2)

    def _grad(cost, h, y):
        return h - y


class logitError(_errfn):
    """
    logit error function: -ylog(h) - (1-y)log(1-h)
    """
    def _calc(cost, h, y):
        return -np.multiply(y, np.log(h)) - np.multiply(1-y, np.log(1-h))

    def _grad(cost, h, y):
        return np.divide(h-y, np.multiply(h, 1-h))