import numpy as np
import matplotlib.pyplot as plt

np.random.seed(665)

eps = 0.001
eps0 = 0.0001
lambda_reg = 0.0001
step = 0.007

points = 100
poly_deg = 15
x_train = np.linspace(0, 1, points)
gt_train = 30 * x_train * x_train
err = 2 * np.random.randn(points)
err[3] += 200
err[77] += 100
err[50] -= 100
t_train = gt_train + err


def return_phi(X, n):
    phi_n = np.empty((len(X), n + 1))
    phi_n[:, 0] = 1
    phi_n[:, 1] = X
    for i in range(2, n + 1):
        phi_n[:, i] = phi_n[:, i - 1] * phi_n[:, 1]
    return phi_n


def loss(X, t, w, lamb, n):
    return 1 / 2 * np.sum((t - w @ return_phi(X, n).T)**2)


def gradient(X, t, w, lamb, n):
    return -np.dot(t - np.dot(w, return_phi(X, n).T), return_phi(X, n)) + lamb * w


def gradient_descent(X, t, n, step, lamb):
    loss_vals = []
    w_next = np.random.rand(n + 1).reshape((1, n + 1)) / 100
    cant_stop = True
    while cant_stop:
        w_old = w_next
        w_next = w_old - step * gradient(X, t, w_old, lamb, n)
        "TO DO"
        loss_vals.append(loss)
        if np.linalg.norm(w_next - w_old) <= eps * np.linalg.norm(w_next) * eps0:
            cant_stop = False
    return loss_vals, w_next


loss_vals, w_itog = gradient_descent(x_train, t_train, poly_deg, step, lambda_reg)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x_train, t_train, 'ro', markersize=3)
ax1.plot(x_train, w_itog.dot(return_phi(x_train, poly_deg).T).flatten())
ax2.plot(list(range(1, len(loss_vals) + 1)), loss_vals)
ax2.set_xlabel("Ð˜Ñ‚ÐµÑ€Ð°Ñ†Ð¸Ñ")
ax2.set_ylabel("ÐžÑˆÐ¸Ð±ÐºÐ°")
plt.show()