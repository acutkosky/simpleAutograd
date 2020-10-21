import numpy as np


class Variable(object):
    '''variable class used in autograd.
    Groups two numpy ndarrays together, self.data which holds the 
    value of this variable, and self.grad, which holds the gradient
    of some other variable with respect to this one.
    Also holds pointers to the operations that created this Variable,
    if any, for use in computing gradients.'''

    def __init__(self, data, parent=None, stop_grad=False):
        self.data = np.array(data)

        # some things are a little easier if scalars are actually
        # rank 1 vectors.
        if self.data.shape == ():
            self.data = np.expand_dims(self.data, 0)

        self.grad = None
        self.stop_grad = stop_grad
        self.parent = parent

    def backward(self, downstream_grad=None):
        if downstream_grad is None:
            downstream_grad = np.ones_like(self.data)

        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += downstream_grad

        if self.parent is not None:
            self.parent.backward(downstream_grad)
