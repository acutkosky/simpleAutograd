from variable import Variable
import tensor_ops as ops
from einsum import EinSum
import numpy as np
import functools

import unittest


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


def test_backward_random(input_shapes, output_shape, reference_fn, operation_fn, positive=False):
    args = [np.random.normal(size=shape) for shape in input_shapes]
    if positive:
        args = [np.abs(arg) for arg in args]
    # np.random.normal(size=output_shape)
    downstream_grad = np.ones(output_shape)

    numeric = numerical_grad(args, downstream_grad, reference_fn)

    variables = [Variable(arg) for arg in args]
    output = operation_fn(variables)
    output.backward(downstream_grad)

    analytic = [var.grad for var in variables]
    diff = np.sum([np.linalg.norm(a-n)/(1e-10 + np.linalg.norm(a+n))
                   for a, n in zip(numeric, analytic)])
    return diff


def test_forward_random(input_shapes, reference_fn, operation_fn, positive=False):
    args = [np.random.normal(size=shape) for shape in input_shapes]
    if positive:
        args = [np.abs(arg) for arg in args]
    variables = [Variable(arg) for arg in args]
    analytic = operation_fn(variables).data

    reference = reference_fn(args)

    diff = np.linalg.norm(analytic-reference) / \
        (1e-10 + np.linalg.norm(analytic+reference))
    return diff


einsum_tests = [
    ('i,i,i->i', [(4), (4), (4)], (4)),
    ('ij,jk->ik', [(3, 4), (4, 5)], (3, 5)),
    ('ijk,klk->il', [(2, 3, 4), (4, 6, 4)], (2, 6)),
    ('i->', [(4)], (1)),
    ('iii->', [(6, 6, 6)], (1)),
    ('iii->i', [(7, 7, 7)], (7)),
    ('hihk,okh,oo->ko', [(4, 5, 4, 6), (3, 6, 4), (3, 3)], (6, 3)),
    ('hihh,okh,oho->i', [(4, 5, 4, 4), (3, 6, 4), (3, 4, 3)], (5)),
    ('hihh,okh,oho->', [(4, 5, 4, 4), (3, 6, 4), (3, 4, 3)], (1)),
    ('ij, ikk, il -> kj', [(2, 3), (2, 3, 3), (2, 4)], (3, 3)),
]


class TestAutograd(unittest.TestCase):

    def test_add(self):
        input_shapes = [(2, 3), (2, 3), (2, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            return sum(args)

        def operation_fn(args):
            add = ops.TensorAdd()
            return add(args)
        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_mul(self):
        input_shapes = [(2, 3), (2, 3), (2, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            return functools.reduce(lambda x, y: x*y, args)

        def operation_fn(args):
            mul = ops.TensorMultiply()
            return mul(args)

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_scalar_multiply(self):
        input_shapes = [(1), (2, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            return args[0]*args[1]

        def operation_fn(args):
            mul = ops.ScalarMultiply()
            return mul(*args)
        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_matrix_multiply(self):
        input_shapes = [(4, 2), (2, 3)]
        output_shape = (4, 3)

        def reference_fn(args):
            return np.dot(args[0], args[1])

        def operation_fn(args):
            mul = ops.MatrixMultiply()
            return mul(*args)
        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_tensordot(self):
        input_shapes = [(2, 4, 2, 6), (2, 6, 3)]
        output_shape = (2, 4, 3)

        def reference_fn(args):
            return np.tensordot(args[0], args[1], 2)

        def operation_fn(args):
            dot = ops.TensorDot()
            return dot(*args, dims_to_contract=2)
        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_exponent(self):
        input_shapes = [(2, 3, 4)]
        output_shape = (2, 3, 4)

        def reference_fn(args):
            return np.exp(args[0])

        def operation_fn(args):
            exp = ops.Exp()
            return exp(args[0])

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_power(self):
        input_shapes = [(2, 3, 4)]
        output_shape = (2, 3, 4)

        def reference_fn(args):
            return np.power(args[0], 0.7)

        def operation_fn(args):
            power = ops.Power(exponent=0.7)
            return power(args[0])

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=True)

    def test_maximum(self):
        input_shapes = [(2, 3), (2, 3), (2, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            return functools.reduce(lambda x, y: np.maximum(x, y), args)

        def operation_fn(args):
            maximum = ops.Maximum()
            return maximum(args)

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_reduce_max(self):
        input_shapes = [(2, 3, 4)]
        output_shape = (1)

        def reference_fn(args):
            return np.max(args[0])

        def operation_fn(args):
            reduce_max = ops.ReduceMax()
            return reduce_max(args[0])

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_einsum(self):
        for einsum_config in einsum_tests:
            with self.subTest():
                input_shapes = einsum_config[1]
                output_shape = einsum_config[2]

                def reference_fn(args):
                    args = [einsum_config[0]] + args
                    return np.einsum(*args)

                def operation_fn(args):
                    einsum_op = EinSum(einsum_config[0])
                    return einsum_op(*args)

                self._test_op(input_shapes, output_shape,
                              reference_fn, operation_fn, positive=False)

    def test_chained_ops(self):
        input_shapes = [(2, 3), (3, 4), (2, 4)]
        output_shape = (1)

        def reference_fn(args):
            return np.max(args[2]+np.exp(np.dot(args[0], args[1])))

        def operation_fn(args):
            matmul = ops.MatrixMultiply()
            exp = ops.Exp()
            add = ops.TensorAdd()
            reduce_max = ops.ReduceMax()

            return reduce_max(add([args[2], exp(matmul(args[0], args[1]))]))

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_reuse_vars(self):
        input_shapes = [(2, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            return args[0]*args[0]

        def operation_fn(args):
            mul = ops.TensorMultiply()
            return mul([args[0], args[0]])

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_graph(self):
        input_shapes = [(1), (1), (1)]
        output_shape = (1)

        def reference_fn(args):
            x = args[0] * args[1]
            y = x + args[2]
            z = y*args[1]
            w = z + x
            return w

        def operation_fn(args):
            mul1 = ops.TensorMultiply()
            add1 = ops.TensorAdd()
            mul2 = ops.TensorMultiply()
            add3 = ops.TensorAdd()

            x = mul1([args[0], args[1]])
            y = add1([x, args[2]])
            z = mul2([y, args[1]])
            w = add3([z, x])

            return w

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def test_large_graph(self):
        input_shapes = [(2, 3), (3, 4), (2, 4), (2, 3, 3), (2, 3)]
        output_shape = (2, 3)

        def reference_fn(args):
            x = np.maximum(args[2], np.dot(args[0], args[1]))
            y = x*x
            z = np.power(y, 1.3)
            w = np.einsum('ij, ikk, il -> kj', args[0], args[3], z)
            a = np.dot(args[4], w)
            b = args[0] + a
            return b

        def operation_fn(args):
            matmul1 = ops.MatrixMultiply()
            maximum = ops.Maximum()
            mul = ops.TensorMultiply()
            power = ops.Power(1.3)
            einsum = EinSum('ij, ikk, il -> kj')
            matmul2 = ops.MatrixMultiply()
            add = ops.TensorAdd()

            x = maximum([args[2], matmul1(args[0], args[1])])
            y = mul([x, x])
            z = power(y)
            w = einsum(args[0], args[3], z)
            a = matmul2(args[4], w)
            b = add([args[0], a])

            return b

        self._test_op(input_shapes, output_shape, reference_fn,
                      operation_fn, positive=False)

    def _test_op(self, input_shapes, output_shape, reference_fn, operation_fn, positive=False):
        forward_diff = test_forward_random(
            input_shapes, reference_fn, operation_fn, positive=positive)
        self.assertLessEqual(forward_diff, 1e-6)

        backward_diff = test_backward_random(
            input_shapes, output_shape, reference_fn, operation_fn, positive=positive)
        self.assertLessEqual(backward_diff, 1e-6)


if __name__ == '__main__':
    unittest.main()
