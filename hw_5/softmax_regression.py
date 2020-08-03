from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import pickle
digits = datasets.load_digits()


class SoftmaxRegression:

    def __init__(self, classes_count=10, lr=0.01):
        self.classes_count = classes_count
        self.lr = lr

    def encode(self, gt):
        one_hot_vec = np.zeros((len(gt), self.classes_count))
        for i in range(len(one_hot_vec)):
            one_hot_vec[i, int(gt[i])] = 1.0
        return one_hot_vec

    @staticmethod
    def decode(one_hot_vec):
        return np.argmax(one_hot_vec, axis=1)

    @staticmethod
    def softmax(vec):
        z = vec - np.max(vec, axis=-1, keepdims=True)
        exponent = np.exp(z)
        sum_exp = np.sum(exponent, axis=-1, keepdims=True)
        predictions_vec = exponent / sum_exp
        return predictions_vec

    @staticmethod
    def normalize(data):
        temp = 2 * ((data[:, :-1] - np.min(data[:, :-1])) / np.max(data[:, :-1]) - np.min(data[:, :-1])) - 1
        data[:, :-1] = temp
        return data

    @staticmethod
    def loss(y_pred, y_true, batch_size):
        loss = 0.0
        for i in range(y_pred.shape[0]):
            loss += -(y_true[i]) * (
                        y_pred[i] - np.max(y_pred[i]) - np.log(np.sum(np.exp(y_pred[i] - np.max(y_pred[i])))))
        return loss.mean() / batch_size

    def train_val_test_split(self, data):
        dataset_length = len(data)
        train_length = int(dataset_length * 0.8)
        valid_test_length = int(dataset_length * 0.1)

        train_data = data[: train_length]
        validation_data = data[train_length: train_length + valid_test_length]
        test_data = data[train_length + valid_test_length:]
        train_labels = train_data[:, -1]
        validation_labels = validation_data[:, -1]
        test_labels = test_data[:, -1]

        train_data = train_data[:, :-1]
        validation_data = validation_data[:, :-1]
        test_data = test_data[:, :-1]

        return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

    def preprocess(self, data, labels):
        data = np.concatenate((data, labels[:, np.newaxis]), axis=1)
        np.random.shuffle(data)
        data = self.normalize(data)
        train_data, train_labels, validation_data, validation_labels, test_data, test_labels = self.train_val_test_split(
            data)
        return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

    @staticmethod
    def batch_loader(data, labels, batch_size, iteration):
        if iteration != 0:
            if (int(data.shape[0] / batch_size)) < iteration:
                iteration = iteration % (int(data.shape[0] / batch_size))
        frst = iteration * batch_size
        scnd = frst + batch_size
        return data[frst:scnd], labels[frst:scnd]

    def init_weights(self, features_num):
        w = np.zeros((features_num, self.classes_count))
        b = np.zeros(shape=(self.classes_count), dtype=np.float64)
        return w, b

    def save_weights(self, w, b, filename):
        f = open(filename, 'wb')
        dic = {"w": w, "b": b}
        pickle.dump(dic, f)
        f.close()

    @staticmethod
    def predict(X, w, b):
        return X @ w + b

    def train(self, data, labels, batch_size, iters, val_data, val_labels):
        iteration = 0
        print_val = list(range(0, iters + 20, 20))
        val_losses = []
        losses = []
        features_num = data.shape[1]
        w, b = self.init_weights(features_num)
        while iters > iteration:
            if iteration in print_val:
                val_losses.append(
                    self.loss(self.predict(val_data, w, b), self.encode(val_labels[0:]), val_data.shape[0]))
            data_batch, labels_batch = self.batch_loader(data, labels, batch_size, iteration)
            labels_batch = self.encode(labels_batch)
            predictions = self.predict(data_batch, w, b)
            softmax_preds = self.softmax(predictions)
            losses.append(self.loss(predictions, labels_batch, batch_size))
            gradient = data_batch.T @ (softmax_preds - labels_batch) / batch_size
            w -= self.lr * gradient + self.lr * w
            b -= self.lr * np.sum(softmax_preds - labels_batch, axis=0) / batch_size
            iteration += 1
        print("train loss")
        self.print_loss(losses)
        print("validation loss")
        self.print_loss(val_losses, print_val[:-1])
        return w, b

    def make_confusion_matrix(self, y_pred, y_true):
        self.confusion_matrix = np.zeros((10, 10))
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                self.confusion_matrix[int(y_pred[i]), int(y_pred[i])] += 1
            else:
                self.confusion_matrix[int(y_pred[i]), int(y_true[i])] += 1
        return self.confusion_matrix


    def make_precisions_recalls(self):
        self.precisions = []
        self.recalls = []
        for i in range(self.classes_count):
            precision = self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[i, :])
            self.precisions.append(precision)
            recall = self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[:, i])
            self.recalls.append(recall)
        self.precisions = np.array(self.precisions)
        self.recalls = np.array(self.recalls)
        return self.precisions, self.recalls

    @staticmethod
    def print_loss(losses, losses_range=None):
        if not losses_range:
            plt.plot(range(len(losses)), losses)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.show()
        else:
            plt.plot(losses_range, losses)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.show()

    def test_classifier(self, data, labels, batch_size, iters, val_data, val_labels, train_data, train_labels):
        weights, biases = self.train(train_data, train_labels, batch_size, iters, val_data, val_labels)
        pred = self.predict(data, weights, biases)
        classification = self.decode(self.softmax(pred))
        pd_confusion_matrix = pd.DataFrame(data=self.make_confusion_matrix(classification, labels),
                                           index=list(range(self.classes_count)),
                                           columns=list(range(self.classes_count)), dtype=np.int32)
        self.save_weights(weights, biases, 'weights.pickle')
        print("Confusion matrix")
        print(pd_confusion_matrix)
        precisions, recalls = self.make_precisions_recalls()
        print('Precisions')
        print(precisions)
        print('Recalls')
        print(recalls)


classifier = SoftmaxRegression(10)
data = digits.data
labels = digits.target
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = classifier.preprocess(data,
                                                                                                             labels)
classifier.test_classifier(test_data, test_labels, 16, 2000, validation_data, validation_labels, train_data,
                           train_labels)