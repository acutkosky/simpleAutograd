from variable import Variable
from tensor_ops import EinSum
import numpy as np

def numerical_grad(inputs, downstream_grad, evaluate):
    '''computes numerical gradients.
    Args:
        inputs: list of np.arrays
        downstream_grad: np.array
        evaluate: taks a list of input arrays and produces an output array
            of same shape as downstream_grad
    returns a list of np.arrays such that the ith element of the return value
        is the gradient of np.sum(evaluate(inputs) * downstream_grad)
        with respect to the ith element of inputs.
    '''
    delta = 1e-8

    base_function_value = np.sum(downstream_grad * evaluate(inputs))

    gradients = []

    for i in range(len(inputs)):
        grad = np.zeros(inputs[i].size)
        perturbation = np.zeros(inputs[i].size)
        for j in range(inputs[i].size):
            perturbation[j] = delta
            inputs[i] = inputs[i] + np.reshape(perturbation, inputs[i].shape)
            perturbed_value = np.sum(downstream_grad * evaluate(inputs))
            inputs[i] = inputs[i] - np.reshape(perturbation, inputs[i].shape)
            perturbation[j] = 0.0
            grad[j] = (perturbed_value - base_function_value) / delta
        gradients.append(np.reshape(grad, inputs[i].shape))

    return gradients


def test_operation_random(input_shapes, output_shape, operation_fn):
    inputs = [np.random.normal(size=shape) for shape in input_shapes]
    downstream_grad = downstream_grad = np.random.normal(size=output_shape)
    evaluation = lambda inputs_: operation_fn([Variable(input) for input in inputs_]).data

    evaluation(inputs)

    numeric = numerical_grad(inputs, downstream_grad, evaluation)

    variables = [Variable(input) for input in inputs]
    op = operation_fn(variables)
    op.backward(downstream_grad)

    analytic = [var.grad for var in variables]

    diff = np.sum([np.linalg.norm(a-n)/np.linalg.norm(a+n) for a,n in zip(numeric,analytic)])
    print("inputs: ", inputs)
    print("downstream grad: ", downstream_grad)
    print("numeric: ", numeric)
    print("analytic: ", analytic)
    return diff


operation_fn = lambda inputs: EinSum('i,j,k->i').forward(*inputs)
input_shapes = [(3),(3), (2)]
output_shape = (3)

diff = test_operation_random(input_shapes, output_shape, operation_fn)

print("diff: ", diff)
