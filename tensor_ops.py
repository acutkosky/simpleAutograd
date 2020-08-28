import numpy as np
import functools

from variable import Variable
from operation import Operation

def all_but_i(i, excludee):
    '''excludes the ith element from list excludee'''
    return [item for j, item in enumerate(excludee) if j!=i]

class EinSum(Operation):
    '''differentiable einsum operation.
        only supports explicit mode einsum (i.e. with ->), and does not support
        ellipses broadcasting.'''
    def __init__(self):
        super(EinSum, self).__init__(name="EinSum")

    def forward_call(self, subscripts, *operands):
        answer = Variable(data=np.einsum(subscripts, [operand.data for operand in operands]), parent=self)
        self.subscripts = subscripts
        self.inputs = operands

        return answer

    def backward_call(self, downstream_grad):
        input_subscripts, output_subscript = self.subscripts.split('->')
        input_subscipts = input_subscripts.split(',')
        for i, operand in enumerate(self.inputs):
            other_subscripts = all_but_i(i, input_subcripts)
            other_subscripts.append(output_subscript)
            subscript_string = ','.join(other_subscripts)
            subscript_string += '->' + input_subscripts[i]
            grad_inputs = subscript
            operand.grad = np.einsum(subscript_string, [input.data for input in all_but_i(i, self.inputs)] + [downstream_grad])


class TensorContract(Operation):
    def __init__(self):
        super(TensorContract, self).__init__(name="Contract")

    def forward_call(self, A, B, dims_to_contract):
        '''A and B are are Variables, dims_to_contract is number
        of dimensions to contract.
        Example:
        A is dim [2, 3, 4]
        B is dim [4, 3, 5]

        if dims_to_contract is 1, output will be [2, 3, 3, 5]
        if dims_to_contract is 2, output will be [2, 5]
        Otherwise it is an error.
        '''

        def __init__(self):
            super(TensorContract, self).__init__(name="Contract")

        def forward_call(self, A, B, dims_to_contract):
            answer = Variable(data=np.tensordot(A.data, B.data, dims_to_contract), parent=self)





class TensorAdd(Operation):
    '''tensor addition operation.'''

    def __init__(self):
        super(TensorAdd, self).__init__(name="Add")

    def forward_call(self, inputs):
        '''inputs should be a list of ndarrays, all with the same dimensions.'''

        assert(len(inputs) > 0, "Add called with no inputs!")
        for input in inputs:
            assert(input.shape == inputs[0].shape, "Shape mismatch in Add!")

        self.state['num_inputs'] = len(inputs)

        return functools.reduce(lambda x,y: x+y, inputs)

    def backward_call(self, downstream_grad):
        return [downstream_grad for _ in range(self.state['num_inputs'])]

class TensorMultiply(Operation):
    '''coordinate-wise multiply operation.'''

    def __init__(self):
        super(TensorMultiply, self).__init__(name="Multiply")

    def forward_call(self, inputs):
        '''inputs should be a list of ndarrays, all with the same dimensions.'''

        assert(len(inputs) > 0, "Multiply called with no inputs!")
        for input in inputs:
            assert(input.shape == inputs[0].shape, "Shape mismatch in Multiply!")

        self.state['num_inputs'] = len(inputs)
        self.state['inputs'] = [np.copy(input) for input in inputs]
        self.state['output'] = functools.reduce(lambda x,y: x*y, inputs)


        return Variable(self.state['output'])

    def backward_call(self, downstream_grad):


        # gradient of abcd with respect to b is acd
        upstream_grads = []
        for i, range(len(self.state['inputs']))
            #set 0s to 1 to avoid divide by zero errors.
            product = downstream_grad
            for j, input in enumerate(self.state['inputs'])
                if i!=j:
                    product = product * input
            upstream_grads.append(product)

        return upstream_grads

def ScalarMultiply(Operation):
    '''multiplication by a scalar.'''
    def __init__(self):
        super(TensorMultiply, self).__init__(name="Scalar Multiply")

    def forward_call(self, inputs):
        '''inputs[0] is a scalar, inputs[1] is an ndarray.'''

        assert(len(inputs) !=2, " Scalar multiply not called with 2 inputs!")

        self.state['scalar'] = inputs[0]
        self.state['tensor'] = np.copy(input[1])
        self.state['output'] = inputs[0] * inputs[1]


        return self.state['output']

    def backward_call(self, downstream_grad):


        # gradient of abcd with respect to b is acd
        upstream_grads = [
            np.sum(self.state['tensor'] * downstream_grad),
            downstream_grad * self.state['scalar']
        ]

        return upstream_grads


def MatrixMultiply(Operation):
    def __init__(self):
        super(MatrixMultiply, self).__init__(name="Matrix Multiply")
