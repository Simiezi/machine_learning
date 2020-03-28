import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    x = np.linspace(0, 2 * np.pi, 1000)
    gt = 100 * np.sin(x) + 0.5 * np.exp(x) + 300
    err = np.random.normal(0, 10, size=1000)
    y = gt + err
    basis = np.ones((1000, 1)).reshape(1000, -1)
    fig, plots = plt.subplots(4, 5, sharex=True, sharey=True, figsize=(30, 15))
    losses = []
    coefficient = 1
    for i in range(4):
        for j in range(5):
            basis = np.concatenate((basis, x.reshape(1000, -1)**coefficient), axis=1)
            weights = (np.linalg.inv(basis.T@basis)@basis.T)@y
            y_out = (weights*basis).sum(axis=1)
            plots[i, j].plot(x, y, 'bo', color='blue')
            plots[i, j].plot(x, y_out, color='red')
            losses.append(((y - y_out)**2).mean())
            plots[i, j].set_title(f'power = {coefficient}, loss = {losses[coefficient-1]:.2f}')
            coefficient += 1

    print(losses)
    plt.ylim((0, 700))
    # plt.xlim(-1, 7)
    plt.show()
    """
    Выведем график изменения лосса
    """
    plt.plot(losses)
    plt.ylim(0, 7000)
    plt.title('Loss change')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.show()