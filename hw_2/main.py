import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations


def train_split(data, data_len):
    return data[: int(0.8 * data_len)]


def eval_split(data, data_len):
    return data[int(0.8 * data_len):int(0.8 * data_len) + int(0.1 * data_len)]


def test_split(data, data_len):
    return data[int(0.8 * data_len) + int(0.1 * data_len) : int(0.8 * data_len) + int(0.1 * data_len) * 2]


def magic_function(function, value):
    return eval(function)


def create_basis(mode):
    return np.ones((len(mode), 1)).reshape(len(mode), -1)

def create_model(basis, mode, combination):
    pass


if __name__ == '__main__':
    size = 2000
    x = np.linspace(0.01, 2 * np.pi, size)
    gt = 100 * np.sin(x) + 0.5 * np.exp(x) + 300
    err = np.random.normal(0, 10, size=size)
    y = gt + err

    models = []
    functions = ['x**1', 'x**2', 'x**3', 'np.cos(x)', 'np.sin(x)', 'np.exp(x)', 'np.log(x)']
    train_x, train_y = train_split(x, size), train_split(y, size)
    eval_x, eval_y = eval_split(x, size), eval_split(y, size)
    test_x, test_y = test_split(x, size), test_split(y, size)

    for quantity in range(1, 8):
        for combination in combinations(range(len(functions)), quantity):
            basis = create_basis(train_x)
            for index in combination:
                basis = np.concatenate((basis, (magic_function(functions[index], train_x[:, np.newaxis])[:, np.newaxis])[:len(train_x)]), axis=1)
            weights = (np.linalg.inv(basis.T @ basis) @ basis.T) @ train_y
            train_y_out = (weights*basis).sum(axis=1)
            train_loss = ((train_y - train_y_out)**2).mean()

            eval_basis = create_basis(eval_x)
            for index in combination:
                eval_basis = np.concatenate((eval_basis, (magic_function(functions[index], eval_x[:, np.newaxis])[:, np.newaxis])[:len(eval_x)]), axis=1)
            eval_y_out = (weights * eval_basis).sum(axis=1)
            eval_loss = ((eval_y - eval_y_out)**2).mean()

            model = dict()
            model['weights'] = weights
            model['functions'] = list(combination)
            model['train_loss'] = train_loss
            model['eval_loss'] = eval_loss
            models.append(model)

    models.sort(key=lambda arr: arr['eval_loss'])

    for i in range(3):
        funcs = models[i]['functions']
        print(funcs)
        weights = models[i]['weights']
        test_basis = create_basis(test_x)
        for index in funcs:
            test_basis = np.concatenate((test_basis, (magic_function(functions[index], test_x[:, np.newaxis])[:, np.newaxis])[:len(test_x)]), axis=1)
        test_y_out = (weights * test_basis).sum(axis=1)
        test_loss = ((test_y - test_y_out)**2).mean()
        models[i]['test_loss'] = test_loss

        basis = create_basis(x)
        for index in funcs:
            basis = np.concatenate((basis, (magic_function(functions[index], x[:, np.newaxis])[:, np.newaxis])[:len(x)]), axis=1)
        y_out = (weights * basis).sum(axis=1)
        label = ''.join([f'{w:.2f} * {functions[name]}+' for w, name in zip(weights, funcs)]) + f'{weights[-1]:.2f}'
        plt.plot(x, y, 'bo',color='blue', label='dataset')
        plt.plot(x, y_out, color='red', marker='o', label=label)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

    labels = []
    best_train = [model['train_loss'] for model in models[:3]]
    best_eval = [model['eval_loss'] for model in models[:3]]
    for i in range(3):
        weights = models[i]['weights']
        funcs = models[i]['functions']
        reg = ''.join([f'{w:.2f} * {functions[name]}+' for w, name in zip(weights, funcs)]) + f'{weights[-1]:.2f}'
        labels.append(reg)
    width = 0.3
    bin_positions = np.array(list(range(3)))
    bin1 = plt.bar(bin_positions - width / 3, best_train, width, label='Train dataset')
    bin2 = plt.bar(bin_positions + width / 3, best_eval, width, label='Validation dataset')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()