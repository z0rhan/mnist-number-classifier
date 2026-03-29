from mlp import MLP
from mse_loss import MSELoss
from softmax import SoftMax
from cross_entropy import CrossEntropyLoss

import scipy.io
import numpy as np


mnist_data = scipy.io.loadmat("mnist-original.mat")

images = mnist_data["data"].T       # shape (N, 784)
labels = mnist_data["label"][0]     # shape (N,)

indices = np.arange(images.shape[0])
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]
images = images.astype("float32") /  255.0


def accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    temperature = 0.2
    correct = 0
    total = X.shape[0]

    for x, label in zip(X, y):
        x = x.reshape(1, -1)
        y_pred = model.forward(x)
        y_pred = y_pred * temperature
        if np.argmax(y_pred[0]) == label:
            correct += 1

    return correct / total


def one_hot_encoding(y: np.floating, target: int):
    label = np.zeros(target)
    label[int(y)] = 1
    return label


def main():
    model = MLP(input_dim=784, hidden_dims=[100, 100, 50], output_dim=10)

    loss_fn = CrossEntropyLoss()
    learning_rate = 0.001
    epochs = 2

    min_loss = float("inf")

    for epoch in range(epochs):
        for step, (x, y_true) in enumerate(zip(images, labels), start=1):
            x = x.reshape(1, -1)  # shape (1, 784)
            y_true_onehot = one_hot_encoding(y_true, 10)  # shape (10,)
            y_true_onehot = y_true_onehot.reshape(1, -1)
            y_pred = model.forward(x)  # shape (1, 10)
            softmax_output = SoftMax()(y_pred)  # shape (1, 10)
            loss = loss_fn(softmax_output, y_true_onehot)  # scalar

            grad_loss = loss_fn.backward(softmax_output, y_true_onehot)  # (1, 10)
            model.backward(grad_loss)
            model.update_params(learning_rate)

            if loss < min_loss:
                min_loss = loss

            if step % 1000 == 0:
                print(f"y predicted: {softmax_output}")
                print(f"y true: {y_true_onehot}")
                print(f"{step}/70000 loss: {loss} min_loss: {min_loss}")

    model.save("model.pkl")

    final_acc = accuracy(model, images, labels)
    print(f"Training accuracy: {final_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
