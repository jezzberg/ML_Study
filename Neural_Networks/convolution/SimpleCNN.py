import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def read_data():
    datasetDir = 'images/'

    firstImg = True
    images, labels = None, None
    for classDir in os.listdir(datasetDir):
        label = int(classDir)
        imgDir = datasetDir + classDir + '/'
        for imgFile in os.listdir(imgDir):
            img = mpimg.imread(imgDir + imgFile)
            if firstImg == True:
                images = np.array([img])
                labels = np.array([label])
                firstImg = False
            else:
                images = np.vstack([images, [img]])
                labels = np.append(labels, label)

    # shuffle data
    randomIdx = np.random.permutation(len(images))
    images = images[randomIdx]
    labels = labels[randomIdx]

    images = torch.Tensor(images)
    images = images.view([images.shape[0], 1, images.shape[2], images.shape[1]])
    labels = torch.Tensor(labels).long()
    return images, labels


class SimpleCNN(nn.Module):
    def __init__(self, imgWidth, imgHeight):
        super(SimpleCNN, self).__init__()

        inputWidth = imgWidth
        nrConvFilters = 3
        convFilterSize = 5
        poolSize = 2
        outputSize = 10

        self.convLayer = nn.Conv2d(1, nrConvFilters, convFilterSize)
        self.poolLayer = nn.MaxPool2d(poolSize)
        fcInputSize = (inputWidth - 2 * (convFilterSize // 2)) * (
                    inputWidth - 2 * (convFilterSize // 2)) * nrConvFilters // (2 * poolSize)
        self.fcLayer = nn.Linear(fcInputSize, outputSize)

    def forward(self, input):
        output = self.convLayer(input)
        output = self.poolLayer(output)
        output = F.relu(output)
        output = output.view([1, -1])
        output = self.fcLayer(output)
        return output

    def predict(self, input):
        return np.argmax(self.forward(input).cpu().detach().numpy())

    def train(self, X_train, y_train, X_test, y_test):
        lossFunc = nn.CrossEntropyLoss()
        nrEpochs = 10
        learnRate = 0.01
        optimizer = torch.optim.SGD(self.parameters(), learnRate)

        for epoch in range(nrEpochs):
            print(f"Epoch {epoch + 1}")
            loss_train = 0
            loss_test = 0
            y_pred = []
            # Training
            for image, label in zip(X_train, y_train):
                optimizer.zero_grad()
                predicted = self.forward(image.unsqueeze(0))
                y_pred.append(np.argmax(predicted.cpu().detach().numpy()))
                loss_train = lossFunc(predicted, label.unsqueeze(0))
                loss_train.backward()
                optimizer.step()
            acc = len(np.where((y_pred - y_train.cpu().detach().numpy()) == 0)[0]) / len(y_pred)
            # Test
            y_pred = []
            for image, label in zip(X_test, y_test):
                predicted = self.forward(image.unsqueeze(0))
                y_pred.append(np.argmax(predicted.cpu().detach().numpy()))
                loss_test = lossFunc(predicted, label.unsqueeze(0))

            acc_test = len(np.where((y_pred - y_test.cpu().detach().numpy()) == 0)[0]) / len(y_pred)

            print(
                f"loss: {round(loss_train.item(), 3)}, acc:{round(acc, 3)} \t test_loss: {round(loss_test.item(), 3)}, test_acc: {round(acc_test, 3)}\n")

def read_my_data():
    path = "mydata"
    images = []
    labels = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            labels.append(int(name.split(".")[0]))
            images.append(cv2.imread(os.path.join(root, name), cv2.IMREAD_GRAYSCALE))

    images = np.array(images)
    labels = np.array(labels)

    images = torch.Tensor(images)
    resh_images = images.view([images.shape[0], 1, images.shape[2], images.shape[1]])
    labels = torch.Tensor(labels).long()
    return resh_images, labels


split_proc = 0.7
images, labels = read_data()
index = int(split_proc * len(images))
X_train, X_test, y_train, y_test = images[:index], images[index:], labels[:index], labels[index:]
imgWidth, imgHeight = 28, 28

myCNN = SimpleCNN(imgWidth, imgHeight)
myCNN.train(X_train, y_train, X_test, y_test)

myimgs, mylbs = read_my_data()
for img, lb in zip(myimgs, mylbs):
    pred = myCNN.predict(img.unsqueeze(0))
    print(f"Predicted: {pred}, True: {lb}")



    print("done")
