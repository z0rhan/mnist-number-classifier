from linear import Linear
from tanh import Tanh

import numpy as np


class MLP:
    def __init__(self, input_dim: int, hidden_dims: [int], output_dim: int):
        """
        input_dim = dimension of input
        hidden_dim = number of nodes in each hidden layer
        output_dim = dimension of output
        """
        self.__layers = []

        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]

            self.__layers.append(Linear(in_dim, out_dim))

            if i < len(dims) - 2:
                self.__layers.append(Tanh())

    def forward(self, x_input: np.ndarray):
        """
        x_input = input data
        """
        self.__input = x_input

        y_output = x_input
        for layer in self.__layers:
            y_output = layer.forward(y_output)

        return y_output

    def backward(self, grad_loss: np.ndarray):
        """
        grad_loss = gradient of loss w.r.t output
        """
        grad = grad_loss
        for layer in reversed(self.__layers):
            grad = layer.backward(grad)

    def update_params(self, learning_rate: float):
        """
        update the weights and biases
        """
        for layer in self.__layers:
            if hasattr(layer, "update_params"):
                layer.update_params(learning_rate)
