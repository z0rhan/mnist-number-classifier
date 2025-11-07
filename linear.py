import numpy as np


class Linear:
    def __init__(self,
                 input_dim: int, output_dim: int,
                 intial_weights=None, initial_biases=None):
        """
        input_dim = dimension of input data
        output_dim = number of neurons in next layer
        weights = np.array of size (output_dim, input_dim)
        biases = np.array of size (output_dim, )
        """
        if intial_weights is None:
            intial_weights = np.random.randn(output_dim, input_dim)

        if initial_biases is None:
            initial_biases = np.random.randn(output_dim)

        self.__weights = intial_weights
        self.__biases = initial_biases
        self.__input_dim = input_dim
        self.__output_dim = output_dim

    def forward(self, input_data: np.ndarray):
        """
        input_data = np.array of size (number of input, input_dim)
        returns output = np.array of size (number of input, output_dim)
        """
        if np.size(input_data, axis=1) != self.__input_dim:
            raise RuntimeError("Dimensions do not match")

        self.__input = input_data

        self.__output = input_data @ self.__weights.T + self.__biases
        return self.__output

    def backward(self, grad_output: np.ndarray):
        """
        grad_output = gradient of loss w.r.t output
        returns gradient of loss w.r.t input =
            np.array of size (number of input, input_dim)
        """
        self.__grad_weights = grad_output.T @ self.__input
        self.__grad_biases = grad_output.sum(axis=0)
        self.__grad_input = grad_output @ self.__weights

        return self.__grad_input

    def update_paramers(self, learning_rate: float):
        """
        learning_rate = it is what it is
        """
        self.__weights = self.__weights - (learning_rate * self.__grad_weights)
        self.__biases = self.__biases - (learning_rate * self.__grad_biases)
