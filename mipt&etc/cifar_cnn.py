import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms


input_size = 3*32*32
num_classes = 10
n_epochs = 2
batch_size = 4
lr = 0.001


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar_trainset = datasets.CIFAR10(root='C:/Users/adelk/PycharmProjects/machine_learning/mipt&etc/datasets',
                                  train=True,
                                  download=True,
                                  transform=transform)

cifar_testset = datasets.CIFAR10(root='C:/Users/adelk/PycharmProjects/machine_learning/mipt&etc/datasets',
                                 train=False,
                                 download=True,
                                 transform=transform)

train_loader = DataLoader(dataset=cifar_trainset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=cifar_testset,
                         batch_size=batch_size,
                         shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CifarModel(nn.Module):
    def __init__(self):
        super(CifarModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step


if __name__ == '__main__':
    model = CifarModel()
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_step = make_train_step(model, loss_fn, optimizer)
    # обучение
    for epoch in range(n_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            loss = train_step(images, labels)
    # тест
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Точность: {} %'.format(100 * correct / total))
    # Валидация
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Точность для %5s : %2d %%' % (c[i], 100 * class_correct[i] / class_total[i]))