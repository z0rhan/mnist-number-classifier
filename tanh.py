import numpy as np


class Tanh:
    def forward(self, input_data: np.ndarray):
        """
        input_data = np.array of size of output of previous layer
        returns output = np.array same size as input
        """
        self.__input = input_data
        self.__output = np.tanh(input_data)

        return self.__output

    def backward(self, grad_output: np.ndarray):
        """
        grad_output = gradient of loss w.r.t output
        return gradient of loss w.r.t input
        """
        self.__grad_input = (1 - (self.__output)**2) * grad_output

        return self.__grad_input
