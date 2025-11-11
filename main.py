from mlp import MLP
from mse_loss import MSELoss

import scipy.io
import numpy as np


mnist_data = scipy.io.loadmat("mnist-original.mat")

images = mnist_data["data"].T       # shape (N, 784)
labels = mnist_data["label"][0]     # shape (N,)


def accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    correct = 0
    total = X.shape[0]

    for x, label in zip(X, y):
        x = x.reshape(1, -1)
        y_pred = model.forward(x)
        if round(y_pred[0][0]) == label:
            correct += 1

    return correct / total


def main():
    model = MLP(input_dim=784, hidden_dims=[10, 10], output_dim=1)

    loss_fn = MSELoss()
    learning_rate = 0.01
    epochs = 1

    min_loss = float("inf")

    for epoch in range(epochs):
        for step, (x, y_true) in enumerate(zip(images, labels), start=1):
            x = x.reshape(1, -1)
            y_true = y_true.reshape(1, -1)

            y_pred = model.forward(x)           # (1, 10)
            loss = loss_fn(y_pred, y_true)      # scalar or (1,)

            grad_loss = loss_fn.backward(y_pred, y_true)  # (1, 10)
            model.backward(grad_loss)
            model.update_params(learning_rate)

            if loss < min_loss:
                min_loss = loss

            if step % 1000 == 0:
                print(f"y predicted: {y_pred}")
                print(f"{step}/70000 loss: {loss} min_loss: {min_loss}")

    final_acc = accuracy(model, images, labels)
    print(f"Training accuracy: {final_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
