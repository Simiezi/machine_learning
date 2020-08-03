import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(int(time.time()))

eps = 0.001
eps0 = 0.0001
lambdas = [0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 5, 10, 20, 30, 40, 50, 60, 70]
lambda_reg = lambdas[4]
step = 0.007

points = 100
poly_deg = 15


def dataset_preprocess(dataset, train_size=0.8, val_size=0.1, test_size=0.1):
    np.random.shuffle(dataset)
    train = int(len(dataset) * train_size)
    val = int(len(dataset) * val_size)
    train_tensor = dataset[:train]
    val_tensor = dataset[train: train + val]
    test_tensor = dataset[train + val:]
    return train_tensor, val_tensor, dataset


points = 100
data = np.linspace(0, 1, points)
gt_train = 30 * data * data
err = 2 * np.random.randn(len(data))
t = err + gt_train
meme = np.concatenate((data[:, np.newaxis], t[:, np.newaxis]), axis=1)

train, val, test = dataset_preprocess(meme.copy())
x_train, y_train = train[:, 0], train[:, 1]
x_val, y_val = val[:, 0], val[:, 1]
x_test, y_test = test[:, 0], test[:, 1]
y_test += 10 * np.random.randn(len(y_test))
y_val += 2 * np.random.randn(len(y_val))


def return_phi(X, n):
    phi_n = np.empty((len(X), n + 1))
    phi_n[:, 0] = 1
    phi_n[:, 1] = X
    for i in range(2, n + 1):
        phi_n[:, i] = phi_n[:, i - 1] * phi_n[:, 1]
    return phi_n


def loss(X, t, w, lamb, n):
    return 1 / 2 * np.sum((t - w @ return_phi(X, n).T) ** 2) + (lamb / 2) * np.sum(w ** 2)


def mse_loss(X, t, w, n):
    return 1 / 2 * np.sum((t - w @ return_phi(X, n).T) ** 2)


def gradient(X, t, w, lamb, n):
    return -np.dot(t - np.dot(w, return_phi(X, n).T), return_phi(X, n)) + lamb * w


def gradient_descent(X, t, n, step, lamb):
    loss_vals = []
    w_next = np.random.rand(n + 1).reshape((1, n + 1)) / 100
    cant_stop = True
    while cant_stop:
        w_old = w_next
        w_next = w_old - step * gradient(X, t, w_old, lamb, n)
        loss_vals.append(loss(X, t, w_next, lamb, n))
        if np.linalg.norm(w_next - w_old) <= eps * np.linalg.norm(w_next) * eps0:
            cant_stop = False
    return loss_vals, w_next


def train_val(x_train, t_train, x_val, t_val, n, step, lambda_reg):
    train_vals = []
    valid_vals = []
    w_next = np.random.rand(n + 1).reshape((1, n + 1)) / 100
    cant_stop = True
    while cant_stop:
        w_old = w_next
        w_next = w_old - step * gradient(x_train, t_train, w_old, lambda_reg, n)
        train_vals.append(loss(x_train, t_train, w_next, lambda_reg, n))
        valid_vals.append(loss(x_val, t_val, w_next, lambda_reg, n))
        if np.linalg.norm(w_next - w_old) <= eps * np.linalg.norm(w_next) * eps0:
            cant_stop = False
    return w_next, train_vals[-1], valid_vals[-1]


models = []
for lamb in lambdas:
    weights, train_loss, val_loss = train_val(x_train, y_train, x_val, y_val, poly_deg, step, lamb)
    results = dict()
    results["lambda"] = lamb
    results["train_loss"] = train_loss
    results["validation_loss"] = val_loss
    results["weights"] = weights
    models.append(results)

width = 0.4
bin_pos = np.array(list(range(len(models))))
train_losses = [k["train_loss"] for k in models]
val_losses = [k["validation_loss"] for k in models]
fig, ax = plt.subplots(figsize=(20, 10))
train_for_bar = ax.bar(bin_pos, train_losses, width, align='center', label='Train')
val_for_bar = ax.bar(bin_pos, val_losses, width, align='center', label='Validation')
plt.xlabel('Regularization value')
plt.ylabel('Loss')
plt.xticks(bin_pos, lambdas)
plt.show()

# Best model
sorted_models = sorted(models, key=lambda d: d["validation_loss"])
best_model = sorted_models[0]
print('\n')
print(f'Best model lambda = {best_model["lambda"]}')
print(f'Best model weights = {best_model["weights"]}')
print(f'Best model test loss = {mse_loss(x_test, y_test, best_model["weights"], poly_deg)}')