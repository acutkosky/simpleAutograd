import numpy as np

from operation import Operation


def all_but_i(i, exclude_from):
    '''excludes the ith element from list exclude_from'''
    return [item for j, item in enumerate(exclude_from) if j != i]


def einsum_subscripts_ops(ops, subscripts, subscriptout, output_shape):
    '''
    wrapper around np.einsum that enhances functionality in two ways:
    allows to specify the output shape
    if the output subscripts have repeated indices, will set all entries of the output
    for which those repeated indices are not equal to zero.

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
    # remove excess indices from subscriptout
    index_set = set([index for subscript in subscripts for index in subscript])

    dimensions = {}
    for op, subscript in zip(ops, subscripts):
        for i, index in enumerate(subscript):
            if index not in dimensions:
                dimensions[index] = np.shape(op)[i]
            else:
                dimensions[index] = max(dimensions[index], np.shape(op)[i])

    # np.einsum does not work with repeated indices in the output, so we will need to
    # extract these repeated indices. In order to put expand back properly, we will
    # use np.put_along_axis. We will make one call to this function for every repeated
    # index. We'll keep track of a kind of "path" descriping how we should structure
    # these calls.
    # I strongly suspect there is a better way to do this, but we have reached the limits of
    # my poor numpy-fu.
    # uniqued_subscriptout = np.unique(filtered_subscriptout)
    # unique_map = {s: [] for s in uniqued_subscriptout}
    # for i, s in enumerate(filtered_subscriptout):
    #     unique_map[s].append(i)
    output_expanded_uniqued_shape = []
    output_uniqued_shape = []
    uniqued_subscriptout = []
    unique_subscripts = {}
    source_to_dest_indices = {}
    i = 0
    needs_expanding = False
    for s in subscriptout:
        if s not in unique_subscripts:
            output_expanded_uniqued_shape.append(output_shape[i])
            output_uniqued_shape.append(output_shape[i])
            uniqued_subscriptout.append(s)
            unique_subscripts[s] = i
            source_to_dest_indices[i] = []
        else:
            needs_expanding = True
            output_expanded_uniqued_shape.append(1)
            source_to_dest_indices[unique_subscripts[s]].append(i)
        i += 1

    def expand_index(source_index, dest_index, data):

        dest_shape = list(data.shape)
        dest_shape[dest_index] = data.shape[source_index]
        dest_data = np.zeros(dest_shape)

        put_indices_shape = np.ones_like(data.shape)
        put_indices_shape[source_index] = data.shape[source_index]
        put_indices = np.reshape(
            np.arange(data.shape[source_index]), put_indices_shape)

        np.put_along_axis(dest_data, put_indices, data, dest_index)

        return dest_data

    def expand_indices(source_to_dest, data):
        if not needs_expanding:
            return data
        for source_index in source_to_dest:
            for dest_index in source_to_dest[source_index]:
                data = expand_index(source_index, dest_index, data)

        return data

    # np.einsum also only believes in output subscripts that are subsets of the
    # input subscripts. So we need to filter out the output indices that
    # are not present in the input, and then run einsum.
    # The output of einsum will then be the wrong shape, so we insert extra
    # dimensions and then broadcast back to the correct shape.
    filtered_subscriptout = [_ for _ in uniqued_subscriptout if _ in index_set]

    einsum_inputs = [_ for pair in zip(ops, subscripts) for _ in pair]
    einsum_inputs.append(filtered_subscriptout)
    output = np.einsum(*einsum_inputs)

    output_prebroadcast_shape = [dimensions[_]
                                 if _ in index_set else 1 for _ in uniqued_subscriptout]
    output = np.reshape(output, output_prebroadcast_shape)
    output = np.broadcast_to(output, output_uniqued_shape)
    output = np.reshape(output, output_expanded_uniqued_shape)
    output = expand_indices(source_to_dest_indices, output)
    return output


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

    def __init__(self, subscripts, name=None):
        if name is None:
            name = "EinSum: {}".format(subscripts)
        super(EinSum, self).__init__(name=name)
        self.subscripts = subscripts

    def forward_call(self, *operands):
        self.parents = list(operands)
        return np.einsum(self.subscripts, *[operand.data for operand in operands])

    def backward_call(self, downstream_grad):
        '''backwards operation for einsum

        Most of the time, the derivative can be expressed as another einsum.
        If we differentiate with respect to the first argument, and the input
        is 
        I1, I2,...,In -> J
        where I1, I2, etc are subscript strings and J is the subscript string
        of the output, then so long as
        1. all indices in I1 show up in the other I's or in J.
        2. no index in I1 repeats.
        then the derivative is
        J, I2,..., In -> I1

        similar statements hold for the 2nd, 3rd, 4th etc arguments.

        If 1 is false, we remove the extra indices from I1, compute the einsum, and then
        broadcast back to the correct shape.

        if 2 is false, then we remove the duplicate indices and compute the einsum, then
        populate a tensor of the desired output shape with all zeros except for indices in which
        the duplicated indices are identical. On this indices, we take values from the output of einsum.

        Most of this operation is captured in the einsum_subscript_ops function, which augments einsum with
        the ability to perform these broadcasting and sparse-populating operations.
        '''

        input_subscripts, output_subscript = self.subscripts.split('->')
        input_subscripts = [_.strip() for _ in input_subscripts.split(',')]
        output_subscript = output_subscript.strip()
        subscript_lists = get_subscript_lists(
            input_subscripts + [output_subscript])

        grads = []

        for i, operand in enumerate(self.parents):
            other_subscripts = all_but_i(i, subscript_lists)
            operands_for_grad = [x.data for x in all_but_i(
                i, self.parents + [downstream_grad])]
            grads.append(einsum_subscripts_ops(
                operands_for_grad, other_subscripts, subscript_lists[i], np.shape(operand.data)))
        return grads
