import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv


class SimpleNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(SimpleNN, self).__init__()
        
        self.fc1 = torch.nn.Linear(inputSize, hiddenSize)
        self.fc2 = torch.nn.Linear(hiddenSize, outputSize)
        self.accuracy_epochs = []
        self.loss_epochs = []

        self.valid_accuracy_epochs = []
        self.valid_loss_epochs = []

    def forward(self, input):
        output = self.fc1(input)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.softmax(output)
        return output

    def predict(self, input):
        input_tensor = torch.Tensor(input)
        result = self.forward(input_tensor)
        idx = np.argmax(result)
        print(f"Apratine clasei {idx} cu probabilitatea {result}")
        return idx, result[idx]

    def getbest_epoch(self):
        return np.argmax(np.array(self.valid_accuracy_epochs))

    def trian_cross(self, X, y, nrEpochs=10, learnRate=0.01, p_valid=0.1):
        lossFunc = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), learnRate)
        len_data = X.shape[0]
        n_taken = int(p_valid * len_data)

        for epoch in range(nrEpochs):
            idx = (epoch * n_taken) % len_data
            X_valid = X[idx: n_taken*(epoch+1)]
            y_valid = y[idx: n_taken*(epoch+1)]

            X_train = np.concatenate((X[0:idx], X[idx + n_taken:]))
            y_train = np.concatenate((y[0:idx], y[idx + n_taken:]))

            X_trainT = torch.Tensor(X_train)
            y_trainT = torch.Tensor(y_train).long()
            X_validT = torch.Tensor(X_valid)
            y_validT = torch.Tensor(y_valid).long()

            acc_epoch = []
            loss_epoch = []
            for input, target in zip(X_trainT, y_trainT):
                optimizer.zero_grad()
                predicted = self.forward(input.unsqueeze(0))
                acc_epoch.append((np.argmax(to_np(predicted)) == target.data.tolist()) * 1)
                loss = lossFunc(predicted, target.unsqueeze(0))
                loss_epoch.append(loss.data.tolist())
                loss.backward()
                optimizer.step()

            self.loss_epochs.append(np.mean(np.array(loss_epoch)))
            self.accuracy_epochs.append(np.count_nonzero(np.array(acc_epoch)) / len(acc_epoch))

            valid_acc_epoch = []
            valid_loss_epoch = []
            for x_v, y_v in zip(X_validT, y_validT):
                pred_valid = self.forward(x_v.unsqueeze(0))
                valid_acc_epoch.append((np.argmax(to_np(pred_valid)) == y_v.data.tolist()) * 1)
                loss = lossFunc(pred_valid, y_v.unsqueeze(0))
                valid_loss_epoch.append(loss.data.tolist())

            self.valid_loss_epochs.append(np.mean(np.array(valid_loss_epoch)))
            self.valid_accuracy_epochs.append(np.count_nonzero(np.array(valid_acc_epoch)) / len(valid_acc_epoch))

            print(
                f"EPOCH {epoch} ---- TRAINING_DATA -------- ACC = {self.accuracy_epochs[epoch]} -------- LOSS = {self.loss_epochs[epoch]}")
            print(
                f"EPOCH {epoch} ---- VALID_DATA -------- ACC = {self.valid_accuracy_epochs[epoch]} -------- LOSS = {self.valid_loss_epochs[epoch]}")
            print("\n\n")

    def train(self, X_train, y_train, X_valid, y_valid, nrEpochs=10, learnRate=0.01):
        lossFunc = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), learnRate)

        for epoch in range(nrEpochs):
            acc_epoch = []
            loss_epoch = []
            for input, target in zip(X_train, y_train):
                optimizer.zero_grad()
                predicted = self.forward(input.unsqueeze(0))
                acc_epoch.append((np.argmax(to_np(predicted)) == target.data.tolist())*1)
                loss = lossFunc(predicted, target.unsqueeze(0))
                loss_epoch.append(loss.data.tolist())
                loss.backward()
                optimizer.step()

            self.loss_epochs.append(np.mean(np.array(loss_epoch)))
            self.accuracy_epochs.append(np.count_nonzero(np.array(acc_epoch)) / len(acc_epoch))

            valid_acc_epoch = []
            valid_loss_epoch = []
            for x_v, y_v in zip(X_valid, y_valid):
                pred_valid = self.forward(x_v.unsqueeze(0))
                valid_acc_epoch.append((np.argmax(to_np(pred_valid)) == y_v.data.tolist()) * 1)
                loss = lossFunc(pred_valid, y_v.unsqueeze(0))
                valid_loss_epoch.append(loss.data.tolist())

            self.valid_loss_epochs.append(np.mean(np.array(valid_loss_epoch)))
            self.valid_accuracy_epochs.append(np.count_nonzero(np.array(valid_acc_epoch)) / len(valid_acc_epoch))

            print(f"EPOCH {epoch} \t TRAINING_DATA \t ACC = {self.accuracy_epochs[epoch]} \t LOSS = {self.loss_epochs[epoch]}")
            print(f"EPOCH {epoch} \t VALID_DATA \t ACC = {self.valid_accuracy_epochs[epoch]} \t LOSS = {self.valid_loss_epochs[epoch]}")
            print("\n\n")

    def plot_loss(self):
        plt.plot(range(len(self.loss_epochs)), self.loss_epochs, 'y-')
        plt.plot(range(len(self.loss_epochs)), self.valid_loss_epochs, 'b-')
        plt.title("LOSS")
        plt.legend(["training", "validation"], loc="lower right")
        plt.xlabel("Epochs")
        plt.show()

    def plot_acc(self):
        plt.plot(range(len(self.accuracy_epochs)), self.accuracy_epochs, 'y-')
        plt.plot(range(len(self.valid_accuracy_epochs)), self.valid_accuracy_epochs, 'b-')
        plt.title("ACCURACY")
        plt.legend(["training", "validation"], loc="lower right")
        plt.xlabel("Epochs")
        plt.show()


def to_np(tensor):
    return tensor.cpu().detach().numpy()


def read_data(path='iris.csv'):

    dataFile = open(path, 'r')
    dataset = csv.reader(dataFile)
    # skip first row which contains csv header
    nrAttributes = len(next(dataset)) - 1
    dataset = list(dataset)
    nrInstances = len(dataset)

    instances = np.empty([nrInstances, nrAttributes])
    labelStrings = [None] * nrInstances
    labels = np.empty(nrInstances)

    idx = 0
    for row in dataset:
        instances[idx] = np.array(row[:nrAttributes])
        labelStrings[idx] = row[-1]
        idx += 1

    uniqueLabelStrings = sorted(set(labelStrings))
    labelDict = {}
    labelIdx = 0
    for label in uniqueLabelStrings:
        labelDict[label] = labelIdx
        labelIdx += 1

    for i in range(len(labelStrings)):
        labels[i] = labelDict[labelStrings[i]]

    # shuffle data
    randomIdx = np.random.permutation(len(instances))
    instances = instances[randomIdx]
    labels = labels[randomIdx]

    return instances, labels

def split_half(X, y, p=0.9):
    end_index = X.shape[0]
    taken_set = int(end_index * p)
    return X[:taken_set], y[:taken_set], X[taken_set:], y[taken_set:]


if __name__ == '__main__':
    X, y = read_data()

    X_train, y_train, X_valid, y_valid = split_half(X, y)

    X_trainT = torch.Tensor(X_train)
    y_trainT = torch.Tensor(y_train).long()
    X_validT = torch.Tensor(X_valid)
    y_validT = torch.Tensor(y_valid).long()

    myNet = SimpleNN(4, 5, 3)
    myNet.train(X_trainT, y_trainT, X_validT, y_validT)
    
    # myNet.trian_cross(X, y)
    myNet.plot_loss()
    myNet.plot_acc()

