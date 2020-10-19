import numpy as np
import functools

from variable import Variable
from operation import Operation

def all_but_i(i, exclude_from):
    '''excludes the ith element from list exclude_from'''
    return [item for j, item in enumerate(exclude_from) if j!=i]


def einsum_subscripts_ops(ops, subscripts, subscriptout, output_shape):
    '''
    wrapper around np.einsum that takes inputs in a different order
    and allows us to specify the output_shape.

    args:
        ops: list of numpy arrays
        subscripts: list of subscript indices. Each element is a list of
            integers. len(subscripts[i]) = len(np.shape(ops[i])).
            The i^th element is the subscript string corresponding to
            ops[i] in np.einsum.
        subscriptout: list of integers corresponding to subscripts for the
            output, len(subscriptout) should be the number of dimensions
            in the output.
            Note that *unlike np.einsum*, it is ok to have values in this
            list that do not show up in the input subscripts.
        output_shape: shape of output array.
    '''
    #remove excess indices from subscriptout
    index_set = set([index for subscript in subscripts for index in subscript])

    dimensions = {}
    for op, subscript in zip(ops, subscripts):
        for i, index in enumerate(subscript):
            if index not in dimensions:
                dimensions[index] = np.shape(op)[i]
            else:
                dimensions[index] = max(dimensions[index], np.shape(op)[i])

    # np.einsum only believes in output subscripts that are subsets of the
    # input subscripts. So we need to filter out the output indices that
    # are not present in the input, and then run einsum.
    # The output of einsum will then be the wrong shape, so we insert extra
    # dimensions and then broadcast back to the correct shape.
    filtered_subscriptout = [_ for _ in subscriptout if _ in index_set]
    output_prebroadcast_shape = [dimensions[_] if _ in index_set else 1 for _ in subscriptout]
    einsum_inputs = [_ for pair in zip(ops, subscripts) for _ in pair]
    einsum_inputs.append(filtered_subscriptout)

    return np.broadcast_to(np.reshape(np.einsum(*einsum_inputs), output_prebroadcast_shape), output_shape)

def get_subscript_lists(subscripts):
    '''convert list of subscript strings into a list of lists of integers.'''
    counter = 0
    chars_to_index = {}
    subscripts_lists = []
    for subscript in subscripts:
        subscript_list = []
        if subscript == '':
            # Special case for empty string so that einsum
            # doesn't complain.
            subscript_list = [0]
        for char in subscript:
            if char not in chars_to_index:
                chars_to_index[char] = counter
                counter += 1
            subscript_list.append(chars_to_index[char])
        subscripts_lists.append(subscript_list)
    return subscripts_lists

class EinSum(Operation):
    '''differentiable einsum operation.
        only supports explicit mode einsum (i.e. with ->), and does not support
        ellipses broadcasting.'''
    def __init__(self, subscripts, name="EinSum"):
        super(EinSum, self).__init__(name=name)
        self.subscripts = subscripts

    def forward_call(self, *operands):
        answer = Variable(data=np.einsum(self.subscripts, *[operand.data for operand in operands]), parent=self)
        self.inputs = list(operands)

        return answer

    def backward_call(self, downstream_grad):
        '''backwards operation for einsum

        Most of the time, the derivative can be expressed as another einsum.
        If we differentiate with respect to the first argument, and the input
        is 
        I1, I2,...,In -> J
        where I1, I2, etc are subscript strings and J is the subscript string
        of the output, then as long as
        1. J has rank > 1
        2. all indices in I1 show up in the other I's or in J
        then the derivative is
        J, I2,..., In -> I1

        If 1 is false, then the derivative is just given by multiplying the 
        scalar downstream_grad by:
        I2,...,In -> I1
        
        if 2 is false, then if we let I1' be the indices that are in common,
        we compute 
        J,I2,...,In -> I1'
        and then we need to broadcast I1' to the correct shape I1.

        '''

        input_subscripts, output_subscript = self.subscripts.split('->')
        input_subscripts = input_subscripts.split(',')
        output_subscript = output_subscript.strip()
        subscript_lists = get_subscript_lists(input_subscripts + [output_subscript])

        for i, operand in enumerate(self.inputs):
            other_subscripts = all_but_i(i, subscript_lists)
            operands_for_grad = [input.data for input in all_but_i(i, self.inputs + [downstream_grad])]
            operand.grad = einsum_subscripts_ops(operands_for_grad, other_subscripts, subscript_lists[i], np.shape(operand.data))


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

        assert len(inputs) > 0, "Add called with no inputs!"
        for input in inputs:
            assert input.shape == inputs[0].shape, "Shape mismatch in Add!"

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

        assert len(inputs) > 0, "Multiply called with no inputs!"
        for input in inputs:
            assert input.shape == inputs[0].shape, "Shape mismatch in Multiply!"

        self.state['num_inputs'] = len(inputs)
        self.state['inputs'] = [np.copy(input) for input in inputs]
        self.state['output'] = functools.reduce(lambda x,y: x*y, inputs)


        return Variable(self.state['output'])

    def backward_call(self, downstream_grad):


        # gradient of abcd with respect to b is acd
        upstream_grads = []
        for i in range(len(self.state['inputs'])):
            #set 0s to 1 to avoid divide by zero errors.
            product = downstream_grad
            for j, input in enumerate(self.state['inputs']):
                if i!=j:
                    product = product * input
            upstream_grads.append(product)

        return upstream_grads

def ScalarMultiply(Operation):
    '''multiplication by a scalar.'''
    def __init__(self):
        super(ScalarMultiply, self).__init__(name="Scalar Multiply")

    def forward_call(self, inputs):
        '''inputs[0] is a scalar, inputs[1] is an ndarray.'''

        assert len(inputs) !=2, " Scalar multiply not called with 2 inputs!"

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


def MatrixMultiply(EinSum):
    def __init__(self):
        super(MatrixMultiply, self).__init__("ij,jk->ik", name="Matrix Multiply")

