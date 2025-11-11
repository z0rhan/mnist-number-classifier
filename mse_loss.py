import numpy as np


class MSELoss:
    def __call__(self, y_output: np.ndarray, target: np.ndarray):
        diff = y_output - target
        return (diff ** 2).mean()

    def backward(self, y_output: np.ndarray, target: np.ndarray):
        row, col = y_output.shape
        return (2 * (y_output - target)) / (row * col)
