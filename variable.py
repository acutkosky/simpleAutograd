


class Variable(Object):
    '''variable class used in autograd.
    Simple wrapper that groups two numpy ndarrays together along with a
    pointer to the operation that created this Variable, if any.'''
    def __init__(self, data, parent=None):
        self.data = data
        self.grad = None
        self.parent = parent


    def backward(self):
        parent.backward(grad)
