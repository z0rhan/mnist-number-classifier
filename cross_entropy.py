import numpy as np
import math


class CrossEntropyLoss:
    def __call__(self, softmax_output: np.ndarray, target: np.ndarray):
        losses = np.zeros(np.size(softmax_output, axis=0))
        for i, (prob_row, label_row) in enumerate(zip(softmax_output, target), start=0):
            loss = 0.0
            for prob, label in zip(prob_row, label_row):
                if label == 1:
                    loss -= math.log(prob + 1e-15)
            losses[i] = loss

        return losses

    def backward(self, softmax_output: np.ndarray, target: np.ndarray):
        row, col = softmax_output.shape
        return (softmax_output - target) / row
