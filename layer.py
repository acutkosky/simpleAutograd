'''backprop implementation with layer abstraction.
This could be made more complicated by keeping track of an actual DAG of
operations, but this way is not too hard to implement.
'''
import numpy as np


class Layer:
    '''A layer in a network.

    A layer is simply a function from R^n to R^d for some specified n and d.
    A neural network can usually be written as a sequence of layers:
    if the original input x is in R^d, a 3 layer neural network might be:

    L3(L2(L1(x)))

    We can also view the loss function as itself a layer, so that the loss
    of the network is:

    Loss(L3(L2(L1(x))))

    This class is a base class used to represent different kinds of layer
    functions. We will eventually specify a neural network and its loss function
    with a list:

    [L1, L2, L3, Loss]

    where L1, L2, L3, Loss are all Layer objects.

    Each Layer object implements a function called 'forward'. forward simply
    computes the output of a layer given its input. So instead of
    Loss(L3(L2(L1(x))), we write
    Loss.forward(L3.forward(L2.forward(L1.forward(x)))).
    Doing this computation finishes the forward pass of backprop.

    Each layer also implements a function called 'backward'. Backward is
    responsible for the backward pass of backprop. After we have computed the
    forward pass, we compute
    L1.backward(L2.backward(L3.backward(Loss.backward(1))))
    We give 1 as the input to Loss.backward because backward is implementing
    the chain rule - it multiplies gradients together and so giving 1 as an
    input makes the multiplication an identity operation.

    The outputs of backward are a little subtle. Some layers may have a
    parameter that specifies the function being computed by the layer. For
    example, a Linear layer maintains a weight matrix, so that
    Linear(x) = xW
    for some matrix W.
    The input to backward should be the gradient of the final loss with respect
    to the output of the current layer. The output of backprop should be the
    gradient of the final loss with respect to the input of the current layer,
    which is just the output of the previous layer. This is why it is correct
    to chain the outputs of backprop together. However, backward should ALSO
    compute the gradient of the loss with respect to the current layer's
    parameter and store this internally to be used in training.
    '''
    def __init__(self, parameter=None, name=None):
        self.name = name
        self.forward_called = False
        self.parameter = parameter
        self.grad = None

    def zero_grad(self):
        self.grad = None

    def forward(self, input):
        '''forward pass. Should compute layer and save relevant state
        needed for backward pass.
        Args:
            input: input to this layer.
        returns output of operation.
        '''
        raise NotImplementedError

    def backward(self, downstream_grad):
        '''Performs backward pass.

        This function should also set self.grad to be the gradient of the final
        output of the computation with respect to the parameter.

        Args:
            downstream_grad: gradient from downstream operation in the
                computation graph. This package will only consider
                computation graphs that result in scalar outputs at the final
                node (e.g. loss function computations). As a result,
                the dimension of downstream_grad should match the dimension of
                the output of this layer.

                Formally, if this operation computes F(x), and the final
                computation computes a scalar, G(F(x)), then input_grad is
                dG/dF.
        returns:
            gradient to pass to upstream layers. If the layer computes F(x, w),
            where x is the input and w is the parameter of the layer, then
            the return value should be dF(x,w)/dx * downstream_grad. Here,
            x is in R^n, F(x, w) is in R^m, dF(x, w)/dx is a matrix in R^(n x m)
            downstream_grad is in R^m and * indicates matrix multiplication.

        We should also compute the gradient with respect to the parameter w.
        Again by chain rule, this is dF(x, w)/dw * downstream_grad
        '''
        raise NotImplementedError


class Linear(Layer):
    '''Linear layer. Parameter is NxM matrix L, input is matrix v of size B x N
    where B is batch size, output is vL.'''

    def __init__(self, weights, name="Linear"):
        super(Linear, self).__init__(weights, name)

    def forward(self, input):
        self.input = input
        # print('input: ', input)
        return np.dot(input, self.parameter)

    def backward(self, downstream_grad):
        '''downstream_grad should be NxB.'''

        #Promote vectors to matrices
        if len(downstream_grad.shape) != 2:
            downstream_grad = np.reshape(
                downstream_grad, (len(downstream_grad), 1))

        self.grad = np.dot(self.input.T, downstream_grad)
        return np.dot(downstream_grad, self.parameter.transpose())


class Bias(Layer):
    '''adds a constant bias.'''

    def __init__(self, bias, name="bias"):
        super(Bias, self).__init__(bias, name)

    def forward(self, input):
        self.input = input
        return self.parameter + self.input

    def backward(self, downstream_grad):
        self.grad = np.sum(downstream_grad)
        return downstream_grad

class ReLU(Layer):
    '''ReLU layer. No parameters.'''

    def __init__(self, name="ReLU"):
        super(ReLU, self).__init__(name=name)

    def forward(self, input):
        self.non_negative = input > 0
        return np.maximum(input, 0.0)

    def backward(self, downstream_grad):
        return self.non_negative * downstream_grad


class Sigmoid(Layer):
    '''Sigmoid layer. No parameters.'''

    def __init__(self, name="Sigmoid"):
        super(Sigmoid, self).__init__(name=name)

    def forward(self, input):
        self.output = np.exp(input) / (1.0 + np.exp(input))
        return self.output

    def backward(self, downstream_grad):
        return (self.output - self.output**2) * downstream_grad


class SoftMax(Layer):
    '''SoftMax Layer, no parameters.'''

    def __init__(self, name="SoftMax"):
        super(SoftMax, self).__init__(name="softmax")

    def forward(self, input):
        '''input is BxN array, B is batch dimension.'''
        self.input = input

        self.exponents = np.exp(input)
        self.normalization_constant = np.sum(self.exponents, axis=1)

        return (self.exponents.T / self.normalization_constant).T

    def backward(self, downstream_grad):
        '''downstream grad should be BxN array.'''
        #The gradient expression is depressingly complicated... maybe someone
        #better at vectorizing could do something nicer.
        return ((self.exponents * downstream_grad).T / self.normalization_constant).T \
            - (np.einsum('ij,ji,ik->ki', downstream_grad, self.exponents.T,
                         self.exponents) / self.normalization_constant**2).T


class CrossEntropy(Layer):
    '''cross entropy loss.'''

    def __init__(self, labels, name="Cross Entropy"):
        '''labels is BxC 1-hot vector for correct label.'''
        super(CrossEntropy, self).__init__(name="Cross Entropy")
        self.labels = labels

    def forward(self, input):
        '''input is BxN, output is B'''
        self.input = input
        return -np.sum(self.labels * np.log(self.input)) / self.input.shape[0]

    def backward(self, downstream_grad):
        grad = -self.labels / (self.input * self.input.shape[0])

        return grad * downstream_grad


def forward_layers(layers, input):
    '''Forward pass on all the layers. Must be called before backwards pass.'''
    output = input
    for layer in layers:
        output = layer.forward(output)
    assert output.size == 1, "only supports computations that output a scalar!"
    return output


def backward_layers(layers):
    '''runs a backward pass on all the layers.
    after this function is finished, look at layer.grad to find the
    gradient with respect to that layer's parameter.'''
    downstream_grad = np.array([1])
    for layer in reversed(layers):
        downstream_grad = layer.backward(downstream_grad)


def zero_grad(layers):
    for layer in layers:
        layer.zero_grad()


def numerical_derivative(layers, input):
    base_output = forward_layers(layers, input)
    delta = 1e-7
    for layer in layers:
        if layer.parameter is None:
            continue
        size = layer.parameter.size
        shape = layer.parameter.shape
        base_param = np.copy(layer.parameter)
        perturb = np.zeros(size)
        grad = np.zeros(size)
        for i in range(size):
            perturb[i] = delta
            layer.parameter = base_param + np.reshape(perturb, shape)
            perturb_output = forward_layers(layers, input)
            grad[i] = (perturb_output - base_output) / delta
            perturb[i] = 0.0
        layer.parameter = base_param

        layer.grad = np.reshape(np.copy(grad), shape)


def test_autograd():
    h = 20
    b = 50
    input = np.random.normal(np.zeros((b, h)))
    labels = np.zeros((b, h))
    for i in range(b):
        labels[i, np.random.choice(range(h))] = 1.0

    layers = [
        Linear(np.random.normal(size=(h, 2 * h))),
        Sigmoid(),
        Bias(np.array([np.random.normal()])),
        Linear(np.random.normal(size=(2 * h, 3 * h))),
        ReLU(),
        Linear(np.random.normal(size=(3 * h, h))),
        SoftMax(),
        CrossEntropy(labels)
    ]
    output = forward_layers(layers, input)
    backward_layers(layers)
    analytics = [np.copy(layer.grad)
                 for layer in layers if layer.grad is not None]
    zero_grad(layers)
    numerical_derivative(layers, input)
    numerics = [np.copy(layer.grad)
                for layer in layers if layer.grad is not None]

    diff = np.sum([np.linalg.norm(analytic - numeric)/np.linalg.norm(numeric)
                   for analytic, numeric in zip(analytics, numerics)])
    assert diff < 1e-5, "autograd differs by {} from numeric grad!".format(diff)


if __name__ == "__main__":
    test_autograd()
    print("looking good!")
