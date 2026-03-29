import numpy as np
import math


class SoftMax:
    def __call__(self, y_output: np.ndarray):
        for row in y_output:
            denominator = 0.0
            for val in row:
                denominator += math.exp(val)

            for i, val in enumerate(row, start=0):
                row[i] = math.exp(val) / denominator

        return y_output
