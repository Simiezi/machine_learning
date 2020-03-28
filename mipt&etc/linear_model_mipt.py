import numpy as np
import matplotlib.pyplot as plt


# Веса
W = None
# Байес
b = None


def mean_squared_error(preds, y):
    return ((preds - y)**2).mean()


def weights_init(X, y):
    global W, b

    N = X.shape[0]
    bias = np.ones((N, 1))
    features = np.append(bias, X, axis=1)

    weights = np.linalg.inv(features.T @ features) @ features.T @ y
    # Разделяем байес
    W = weights[1:]
    b = np.array([weights[0]])


def predict(X):
    global W, b
    return np.squeeze(X @ W + b.reshape(-1, 1))


def grad_descent(X, y, lr, iterations):
    global W, b
    W = np.random.rand(X.shape[1])
    b = np.array(np.random.rand(1))

    losses = []

    N = X.shape[0]
    for iter_num in range(iterations):
        preds = predict(X)
        losses.append(mean_squared_error(preds, y))

        w_grad = np.zeros_like(W)
        b_grad = 0
        for sample, prediction, label in zip(X, preds, y):
            w_grad += 2 * (prediction - label) * sample
            b_grad += 2 * (prediction - label)

        W -= lr * w_grad
        b -= lr * b_grad
    return losses


def generate_data(range_, a, b, std, num_points=100):
    X_train = np.random.random(num_points) * (range_[1] - range_[0]) + range_[0]
    y_train = a * X_train + b + np.random.normal(0, std, size=X_train.shape)

    return X_train, y_train


def mse_solution(X_train, y_train):
    print(W, b)
    weights_init(X_train.reshape(-1, 1), y_train)
    plt.scatter(X_train, y_train, c='r')
    plt.plot(X_train, 0.34 * X_train + 13.7)
    plt.plot(X_train, np.squeeze(X_train.reshape(-1, 1) @ W + b.reshape(-1, 1)))
    plt.show()


def grad_descent_solution(X_train, y_train):
    losses = grad_descent(X_train.reshape(-1, 1), y_train, 1e-9, 15000)
    plt.plot(losses)
    plt.scatter(X_train, y_train, c='r')
    plt.plot(X_train, real_a * X_train + real_b)
    plt.plot(X_train, np.squeeze(X_train.reshape(-1, 1) @ W + b.reshape(-1, 1)))
    plt.show()


if __name__ == '__main__':
    real_a = 0.34
    real_b = 13.7
    real_std = 7

    X_train, y_train = generate_data([0, 150], real_a, real_b, real_std)
    mse_solution(X_train, y_train)
