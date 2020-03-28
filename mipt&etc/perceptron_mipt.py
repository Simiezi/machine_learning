from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class Neuron:
    def __init__(self, w=None, b=0):
        self.w = w
        self.b = b

    def activate(self, x):
        return sigmoid(x)

    def forward_pass(self, X):
        n = X.shape[0]
        y_pred = np.zeros((n, 1))
        y_pred = self.activate(X @ self.w.reshape(X.shape[1], 1) + self.b)
        return y_pred.reshape(-1, 1)

    def backward_pass(self, X, y, y_pred, learning_rate=0.1):
        n = len(y)
        y = np.array(y).reshape(-1, 1)
        sigma = self.activate(X @ self.w + self.b)
        self.w = self.w - learning_rate * (X.T @ ((sigma - y) * sigma * (1 - sigma))) / n
        self.b = self.b - learning_rate * np.mean((sigma - y) * sigma * (1 - sigma))

    def fit(self, X, y, num_epochs=5000):
        self.w = np.zeros((X.shape[1], 1))
        self.b = 0
        # значения функции потерь на различных итерациях обновления весов
        Loss_values = []

        for i in range(num_epochs):
            # предсказания с текущими весами
            y_pred = self.forward_pass(X)
            # считаем функцию потерь с текущими весами
            Loss_values.append(loss(y_pred, y))
            # обновляем веса в соответсвие с тем, где ошиблись раньше
            self.backward_pass(X, y, y_pred)

        return Loss_values


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Производная сигмоиды
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def loss(y_pred, y):
    y_pred = y_pred.reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    return 0.5 * np.mean((y_pred - y) ** 2)


if __name__ == '__main__':
    data = pd.read_csv("datasets/apples_pears.csv")

    X = data.iloc[:, :2].values
    y = data['target'].values.reshape((-1, 1))

    neuron = Neuron()
    loss_values = neuron.fit(X, y, num_epochs=50000)

    plt.figure(figsize=(10, 8))
    plt.plot(loss_values)
    plt.title('Функция потерь', fontsize=15)
    plt.xlabel('номер итерации', fontsize=14)
    plt.ylabel('$Loss(\hat{y}, y)$', fontsize=14)
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=np.array(neuron.forward_pass(X) > 0.5).ravel(), cmap='spring')
    plt.title('Яблоки и груши', fontsize=15)
    plt.xlabel('симметричность', fontsize=14)
    plt.ylabel('желтизна', fontsize=14)
    plt.show();
