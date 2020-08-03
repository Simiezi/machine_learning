from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка датасета
digits = datasets.load_digits()

# #Показать случайные картинки
# print(digits.data.reshape(1797, 8, 8)[890])
# print(digits.target)
# fig, axes = plt.subplots(8,8)
# axes=axes.flatten()
# for i, ax in enumerate(axes):
#     dig_ind=np.random.randint(0,len(digits.images))
#     ax.imshow(digits.images[dig_ind].reshape(8,8))
#     ax.set_title(digits.target[dig_ind])
# plt.show()


# Посчитать картинок какого класса сколько
dic = {x: 0 for x in range(10)}
for dig in digits.target:
    dic[dig] += 1
print(dic)


def prepare_data(data, avg):
    """
    Подготавливает данные для кореляционного классификатора
    :param data: np.array, данные (размер выборки, количество пикселей
    :return: data: np.array, данные (размер выборки, количество пикселей
    """
    data = data - avg
    return data.reshape(len(data), avg, -1)


def train_val_test_split(data, labels):
    """
    Делит выборку на обучающий и тестовый датасет
    :param data: np.array, данные (размер выборки, количество пикселей)
    :param labels: np.array, метки (размер выборки,)
    :return: train_data, train_labels, validation_data, validation_labels, test_data, test_labels
    """
    dataset_length = len(data)
    train_length = int(dataset_length * 0.8)
    valid_test_length = int(dataset_length * 0.1)

    train_data = data[: train_length]
    train_labels = labels[: train_length]
    validation_data = data[train_length: train_length + valid_test_length]
    validation_labels = labels[train_length: train_length + valid_test_length]
    test_data = data[train_length + valid_test_length:]
    test_labels = labels[train_length + valid_test_length:]

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def softmax(vec):
    z = vec - np.max(vec, axis=-1, keepdims=True)
    exponent = np.exp(z)
    sum_exp = np.sum(exponent, axis=-1, keepdims=True)
    predictions_vec = exponent / sum_exp
    return predictions_vec


def images_to_classes(data, labels):
    classes_dict = dict()
    for i in range(10):
        classes_dict[i] = []

    for image in range(len(data)):
        classes_dict[labels[image]].append(data[image])
    return classes_dict


class CorelationClassifier:

    def __init__(self, classes_count=10):
        self.classes_count = classes_count

    def fit(self, data, labels):
        """
        Производит обучение алгоритма на заданном датасете
        :param data: np.array, данные (размер выборки, количество пикселей)
        :param labels: np.array, метки (размер выборки,)
        :return:
        """
        self.averages = []
        class_dict = images_to_classes(data, labels)
        for images in range(len(class_dict)):
            self.averages.append(sum(class_dict[images]) / len(class_dict[images]))
        self.averages = np.array(self.averages)

    def predict(self, data):
        """
        Предсказывает вектор вероятностей для каждого наблюдения в выборке
        :param data: np.array, данные (размер выборки, количество пикселей)
        :return: np.array, результаты (len(data), count_of_classes)
        """
        predictions = []
        for image in range(len(data)):
            image_predicts = []
            for avg in range(self.classes_count):
                image_predicts.append(np.sum(np.multiply(data[image], self.averages[avg])))
            predictions.append(image_predicts)
        predictions = np.array(predictions)
        print(predictions.shape)
        return softmax(predictions)

    def accuracy(self, data, labels):
        """
        Оценивает точность (accuracy) алгоритма по выборке
        :param data: np.array, данные (размер выборки, количество пикселей)
        :param labels: np.array, метки (размер выборки,)
        :return:
        """
        y_pred = np.argmax(self.predict(data), axis=1)
        self.make_confusion_matrix(y_pred, labels)
        self.make_precisions_recalls()
        accs = []
        for i in range(self.classes_count):
            accs.append(self.confusion_matrix[i, i])
        self.total_accuracy = np.array(accs).mean()

    def make_confusion_matrix(self, y_pred, y_true):
        self.confusion_matrix = np.zeros((10, 10))
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                self.confusion_matrix[y_pred[i], y_pred[i]] += 1
            else:
                self.confusion_matrix[y_pred[i], y_true[i]] += 1

    def make_precisions_recalls(self):
        self.precisions = []
        self.recalls = []
        for i in range(self.classes_count):
            print(np.sum(self.confusion_matrix[i, :]))
            precision = self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[i, :])
            self.precisions.append(precision)
            recall = self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[:, i])
            self.recalls.append(recall)
        self.precisions = np.array(self.precisions)
        self.recalls = np.array(self.recalls)


train_data, train_labels, validation_data, validation_labels, test_data, test_labels = train_val_test_split(digits.data,
                                                                                                            digits.target)

train_data = prepare_data(train_data, 8)
validation_data = prepare_data(validation_data, 8)
test_data = prepare_data(test_data, 8)

# Посчитать картинок какого класса сколько в обучающем датасете
dic = {x: 0 for x in range(10)}
for dig in train_labels:
    dic[dig] += 1
print(dic)
classifier = CorelationClassifier()
classifier.fit(train_data, train_labels)
classifier.accuracy(validation_data, validation_labels)
pd_confusion_matrix = pd.DataFrame(data=classifier.confusion_matrix, index=list(range(10)), columns=list(range(10)),
                                   dtype=np.int32)
print(f'Validation Confusion matrix\n{pd_confusion_matrix}\n')
pd_precisions = pd.DataFrame(data=classifier.precisions)
print(f'Validation Precisions {pd_precisions}\n')
pd_recalls = pd.DataFrame(data=classifier.recalls)
print(f'Validation Recalls {pd_recalls} \n')
print(f'Validation Mean True Positives {classifier.total_accuracy}')

classifier = CorelationClassifier()
classifier.fit(train_data, train_labels)
classifier.accuracy(test_data, test_labels)
pd_confusion_matrix = pd.DataFrame(data=classifier.confusion_matrix, index=list(range(10)), columns=list(range(10)),
                                   dtype=np.int32)
print(f'Test Confusion matrix\n{pd_confusion_matrix} \n')
pd_precisions = pd.DataFrame(data=classifier.precisions)
print(f'Test Precisions {pd_precisions} \n')
pd_recalls = pd.DataFrame(data=classifier.recalls)
print(f'Test Recalls {pd_recalls} \n')
print(f'Test Mean True Positives {classifier.total_accuracy}')