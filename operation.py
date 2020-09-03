

class Operation(object):
    '''Base class for operations'''

    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.forward_called = False


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        '''wrapper around forward_call to set flag.'''
        self.forward_called = True
        return self.forward_call(*args, **kwargs)

    def backward(self, downstream_grad):
        '''wrapper around backward_call to check assertion.'''

        assert(self.forward_called,
               "backward called before forward on {} operation!".format(self.name))

        self.backward_call(downstream_grad)

        for input in self.inputs:
            input.backward()

    def backward_call(self, downstream_grad):
        '''Performs backward pass.

        This function should also set self.gradients in such a way that
        self.gradients[i] is the gradient of the final output of the computation
        graph with respect to self.parameters[i].

        Args:
            downstream_grad: gradient from downstream operation in the
                computation graph. This package will only consider
                computation graphs that result in scalar outputs at the final
                node (e.g. loss function computations). As a result,
                the dimension of downstream_grad should match the dimension of the
                output of this operation class.

                Formally, if this operation computes F(x), and the final
                computation computes a scalar, G(F(x)), then input_grad is
                dG/dF.
        returns nothing
            list of gradients to pass to upstream operations. The size of this
                list equals the number of inputs to the operation.

                Example:
                If there are N inputs, and the output is F(x_1,...,x_N), then
                the ith element of this list is equal to
                dF/dx_i(x_1,..,x_n) * input_grad, where note that dF/dx_i is a
                matrix of dimension n * m where n is the dimension of x_i
                and m is the dimension of the output F as well as the dimension
                of input_grad, so that dF/dx_i(x_1,..,x_n) * downstream_grad has
                dimension n.
        '''
        raise NotImplementedError

        def forward_call(self, *args, **kwargs):
            '''forward pass. Should compute operation and save relevant state
            needed for backward pass.
            Args:
                inputs: inputs to this operation.
            returns output of operation.
            '''
            raise NotImplementedError
