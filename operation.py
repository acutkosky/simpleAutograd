from variable import Variable
import numpy as np

class Operation(object):
    '''Base class for operations'''

    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.forward_called = False
        self.parents = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        '''wrapper around forward_call to set flag and return Variable.'''
        self.forward_called = True
        output = self.forward_call(*args, **kwargs)
        assert self.parents is not None, "forward did not set self.parents on {} operation!".format(
            self.name)
        return Variable(data=output, parent=self, stop_grad=np.all([p.stop_grad for p in self.parents]))

    def backward(self, downstream_grad):
        '''wrapper around backward_call to check assertion.'''

        assert self.forward_called, "backward called before forward on {} operation!".format(
            self.name)
        upstream_grads = self.backward_call(downstream_grad)
        for var, grad in zip(self.parents, upstream_grads):
            if not var.stop_grad:
                var.backward(grad)

    def backward_call(self, downstream_grad):
        '''Performs backward pass.

        This function has no return values. Instead for each input variable
        V, this function must compute the gradient with respect to V and
        then pass this gradient on to V.backward.

        Specifically, if the output of this operation is F(V,W,U) for some inputs
        V, W, and U, downstream_grad will be dL/dF for some final output L. 
        The forward function should set self.parents = [V, W, U].
        The backward function should then compute and return [dL/dV, dL/dW, dL/dU]

        Args:
            downstream_grad: np.array
                Gradient from downstream operation in the
                computation graph. This package will only consider
                computation graphs that result in scalar outputs at the final
                node (e.g. loss function computations). As a result,
                the dimension of downstream_grad should match the dimension of the
                output of this operation class.

                Formally, if this operation computes F(x), and the final
                computation computes a scalar, G(F(x)), then input_grad is
                dG/dF.
        returns:
            list of np.arrays
        '''
        raise NotImplementedError

    def forward_call(self, *args, **kwargs):
        '''forward pass. Should compute operation and save relevant state
        needed for backward pass.

        Also, must set the self.parents list to be a list of input Variable objects
        provided as input in *args or **kwargs for which this operation can compute
        the derivative with respect to.
        Args:
            inputs: inputs to this operation.

        returns output of operation. This should be an np.array object which will
            eventually be set as the .data attribute of a Variable in forward.
            This is just to save a little boilerplate: the return value of forward
            is Variable(data=ret, parent=self) where ret is the return value of 
            forward_call.
        '''
        raise NotImplementedError
