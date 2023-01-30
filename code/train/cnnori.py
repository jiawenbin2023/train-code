import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models  # add models to the list
from torchvision.utils import make_grid
import os

from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#% matplotlib inline
from torchsummary import summary
# ignore harmless warnings
import warnings


warnings.filterwarnings("ignore")
train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # rotate +/- 10 degrees
    transforms.RandomHorizontalFlip(),  # reverse 50% of images
    transforms.Resize(224),  # resize shortest side to 224 pixels
    transforms.CenterCrop(224),  # crop longest side to 224 pixels at center
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
# train_data=datasets.ImageFolder(root=("../input/smallcracked/haitao10.31/train"),transform=train_transform)
# test_data=datasets.ImageFolder(root=("../input/smallcracked/haitao10.31/val"),transform=test_transform)
# train_data=datasets.ImageFolder(root=("../input/smallcracked/haitao10.31bw/train"),transform=train_transform)
# test_data=datasets.ImageFolder(root=("../input/smallcracked/haitao10.31bw/val"),transform=test_transform)
train_data = datasets.ImageFolder(root=("../data_set/500/train"), transform=train_transform)
test_data = datasets.ImageFolder(root=("../data_set/500/val"), transform=test_transform)
class_names = train_data.classes
class_names
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

test_loader = DataLoader(test_data, batch_size=10)
len(train_data)
len(test_data)
for images, labels in train_loader:
    break
# print the labels
print('Label:', labels.numpy())
print('Class:', *np.array([class_names[i] for i in labels]))

im = make_grid(images, nrow=5)
plt.figure(figsize=(10, 10))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                     std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
im = inv_normalize(im)
plt.figure(figsize=(10, 10))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))


class ConvolutionalNetwork(nn.Module):  # Building our own convolutional neural network
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 54 * 54, 120)  # building with 120 neurons
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 54 * 54)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)

        return F.log_softmax(X, dim=1)


CNNmodel = ConvolutionalNetwork()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CNNmodel.to(device);
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)  # We would be using adam as an optimizer


def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>8}')
    print(f'________\n{sum(params):>8}')


count_parameters(CNNmodel)
import time

start_time = time.time()
train_losses = []
test_losses = []
train_correct = []
test_correct = []

epochs = 1
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    for b, (X_train, y_train) in enumerate(
            train_loader):  # Training our model on images and then testing our model with
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        b += 1  # test batch sizes.
        y_pred = CNNmodel(X_train)
        loss = criterion(y_pred, y_train)
        # true predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        # update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print interim results
        if b % 10 == 0:
            print(f"epoch: {i} batch: {b} accuracy: {trn_corr.item() * 100 / (10 * b):7.3f}%")
    loss = loss.cpu().detach().numpy()
    train_losses.append(loss)
    train_correct.append(trn_corr)

    y_predict = torch.tensor([]).to(device)  # 创建tensor(empty)
    y_true = torch.tensor([]).to(device)

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            y_val = CNNmodel(X_test)
            loss = criterion(y_val, y_test)

            predicted = torch.max(y_val.data, 1)[1]
            btach_corr = (predicted == y_test).sum()
            tst_corr += btach_corr

            y_predict=torch.cat([y_predict,predicted],0)
            y_true=torch.cat([y_true,y_test],0)


            # print(predicted)
            # print(y_test)
        loss = loss.cpu().detach().numpy()
        test_losses.append(loss)
        test_correct.append(tst_corr)
        y_predict = np.array(torch.tensor(y_predict, device='cpu'))
        y_true = np.array(torch.tensor(y_true, device='cpu'))
        print(len(y_true))
        print(len(y_predict))
        print(f"test accuracy: {tst_corr.item() * 100 / (10 * b):7.3f}%")#item取出tensor值
        print(classification_report(y_true, y_predict))
test_correct = np.array(torch.tensor(test_correct, device='cpu'))
test_correct=np.array(test_correct)
test_correct=test_correct/len(test_data)
print(test_correct)
summary(CNNmodel,(3,224,224))
# netron.start('F:/deep-learning-for-image-processing-master/haiwang/Alexnet_11.29rgb2500.pth')
# print(f'\nDuration: {time.time() - start_time:.0f} seconds')


alexnetmodel = models.alexnet(pretrained=False)
count_parameters(alexnetmodel)
# for param in alexnetmodel.parameters():
#     param.requires_grad=False
torch.manual_seed(42)

alexnetmodel.classifier = nn.Sequential(nn.Linear(9216, 1024),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(1024, 2),
                                        nn.LogSoftmax(dim=1))
alexnetmodel
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(alexnetmodel.classifier.parameters(), lr=0.001)
import time

start_time = time.time()
train_losses = []
test_losses = []
trn_correct = []
tst_correct = []
train_correct = []
test_correct = []
device = torch.device("cpu")

epochs = 1
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1

        y_pred = alexnetmodel(X_train)
        loss = criterion(y_pred, y_train)
        # Update parameters
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 10 == 0:
            print(f'epoch: {i:2}  batch: {b:4} [{10 * b:6}]  loss: {loss.item():10.8f}  \
accuracy: {trn_corr.item() * 100 / (10 * b):7.3f}%')

    loss = loss.detach().numpy()
    train_losses.append(loss)
    train_correct.append(trn_corr)

    y_predict = torch.tensor([]).to(device)  # 创建tensor(empty)
    y_true = torch.tensor([]).to(device)

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            b += 1

            y_val = alexnetmodel(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            btach_corr = (predicted == y_test).sum()
            tst_corr += btach_corr

            y_predict = torch.cat([y_predict, predicted], 0)
            y_true = torch.cat([y_true, y_test], 0)

    #
    y_predict = np.array(torch.tensor(y_predict, device='cpu'))
    y_true = np.array(torch.tensor(y_true, device='cpu'))

    print(len(y_predict), len(y_true))
    loss = criterion(y_val, y_test)
    loss = loss.detach().numpy()
    test_losses.append(loss)
    test_correct.append(tst_corr)
    print(f"test accuracy: {tst_corr.item() * 100 / (10 * b):7.3f}%")
    print(classification_report(y_true, y_predict))
test_correct = np.array(torch.tensor(test_correct, device='cpu'))
test_correct=np.array(test_correct)
test_correct=test_correct/len(test_data)
print(test_correct)
summary(alexnetmodel,(3,224,224),)
# print(f'\nDuration: {time.time() - start_time:.0f} seconds')