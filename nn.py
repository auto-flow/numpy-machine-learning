#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-14
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np


class Module(object):
    def __init__(self):
        pass

    def forward(self, *x):
        pass

    def parameters(self):
        pass

    def backward(self, grad):
        pass

    def __call__(self, *x):
        return self.forward(*x)


class Variable(object):
    def __init__(self, weight, wgrad, bias, bgrad):
        self.weight = weight
        self.wgrad = wgrad
        self.v_weight = np.zeros(self.weight.shape)
        self.bias = bias
        self.bgrad = bgrad


class Linear(Module):
    def __init__(self, inplanes, outplanes, preweight=None):
        super(Linear, self).__init__()
        if preweight is None:
            self.weight = np.random.randn(inplanes, outplanes) * 0.5
            self.bias = np.random.randn(outplanes) * 0.5
        else:
            self.weight, self.bias = preweight
        self.input = None
        self.output = None
        self.wgrad = np.zeros(self.weight.shape)
        self.bgrad = np.zeros(self.bias.shape)
        self.variable = Variable(self.weight, self.wgrad, self.bias, self.bgrad)

    def parameters(self):
        return self.variable

    def forward(self, *x):
        x = x[0]
        self.input = x
        # Z = XW + b
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output

    def backward(self, grad):
        self.bgrad = grad
        # dL/dW =  dL/dZ * dZ/dW = X * dL/dZ
        self.wgrad += np.dot(self.input.T, grad)
        # dL/dX = dL/dZ * W
        # 相当于用权重W对梯度grad做线性变换，将梯度传到上一层的神经元
        grad = np.dot(grad, self.weight.T)
        return grad


class Relu(Module):
    def __init__(self):
        super(Relu, self).__init__()
        self.input = None
        self.output = None

    def forward(self, *x):
        x = x[0]
        self.input = x
        x[self.input <= 0] *= 0
        self.output = x
        return self.output

    def backward(self, grad):
        grad[self.input > 0] *= 1
        grad[self.input <= 0] *= 0
        return grad


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.input = None
        self.output = None

    def forward(self, *x):
        x = x[0]
        self.input = x
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backward(self, grad):
        grad *= self.output * (1 - self.output)
        return grad


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.input = None
        self.output = None

    def forward(self, *x):
        x = x[0]
        self.input = x
        self.output = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.output

    def backward(self, grad):
        grad *= (1 - np.square(self.output))
        return grad


class Sequence(Module):
    def __init__(self, *layer):
        super(Sequence, self).__init__()
        self.layers = []
        self.parameter = []
        for item in layer:
            self.layers.append(item)

        for layer in self.layers:
            if isinstance(layer, Linear):
                self.parameter.append(layer.parameters())

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, *x):
        x = x[0]
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        return self.parameter


class Mynet(Module):
    def __init__(self, inplanes, outplanes, n_hidden):
        super(Mynet, self).__init__()
        self.layers = Sequence(
            Linear(inplanes, n_hidden),
            Relu(),
            Linear(n_hidden, outplanes)
        )
        self.criterion = MSE()

    def parameters(self):
        return self.layers.parameters()

    def forward(self, *x):
        x = x[0]
        return self.layers.forward(x)

    def backward(self, grad=None):
        grad = self.criterion.backward(grad)
        self.layers.backward(grad)


class MSE(object):
    def __init__(self):
        self.label = None
        self.pred = None
        self.grad = None
        self.loss = None

    def __call__(self, pred, label):
        return self.forward(pred, label)

    def forward(self, pred, label):
        self.pred, self.label = pred, label
        self.loss = np.sum(0.5 * np.square(self.pred - self.label))
        return self.loss

    def backward(self, grad=None):
        self.grad = (self.pred - self.label)
        ret_grad = np.sum(self.grad, axis=0)
        return np.expand_dims(ret_grad, axis=0)


class SGD(object):
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum

    def zero_grad(self):
        for parameters in self.parameters:
            parameters.wgrad *= 0
            parameters.bgrad *= 0

    def step(self):
        for parameters in self.parameters:
            # parameters.v_weight=parameters.v_weight*self.momentum-self.lr*parameters.wgrad
            parameters.weight -= self.lr * parameters.wgrad
            parameters.bias -= self.lr * parameters.bgrad
