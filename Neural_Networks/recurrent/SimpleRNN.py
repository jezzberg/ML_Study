import numpy as np
import torch as tr
import torch.nn as nn
import random

# build training data
sequenceLength = 1
noSequences = 120


class SimpleRNN(nn.Module):
    def __init__(self, inputSize, outputSize, lstmLayerSize, noLSTMLayers):
        super(SimpleRNN, self).__init__()
        self.inputSize = inputSize
        self.lstmLayerSize = lstmLayerSize
        self.outputSize = outputSize
        self.noLSTMLayers = noLSTMLayers

        self.lstmLayer = nn.LSTM(self.inputSize, self.lstmLayerSize, self.noLSTMLayers)
        self.outLayer = nn.Linear(self.lstmLayerSize, self.outputSize)

    def forward(self, input):
        input = input.view(-1, 1, 1)
        lstmOut, hidden = self.lstmLayer(input)
        outLayerInput = lstmOut[-1, 0, :]
        predictedOut = self.outLayer(outLayerInput)
        return predictedOut

    def train(self, inputs, targets):
        noEpochs = 200
        learnRate = 0.002
        optimizer = tr.optim.Adam(self.parameters(), learnRate)
        lossFunc = nn.CrossEntropyLoss()

        for epoch in range(noEpochs):
            correct_counter = 0
            for input, target in zip(inputs, targets):
                optimizer.zero_grad()
                predicted = self.forward(input)
                loss = lossFunc(predicted.unsqueeze(0), target.unsqueeze(0))
                pred_index = np.argmax(predicted.cpu().detach().numpy())
                if pred_index == target.item():
                    correct_counter += 1
                loss.backward()
                optimizer.step()

            print('Epoch', epoch, 'loss', loss.item(), 'acc: ', correct_counter/targets.shape[0])


def read_data():
    textFile = open("rnn_text.txt",encoding='utf-8')
    text = textFile.read()
    # separate punctuation by spaces
    punctuation = [',', '.', ':', ';', '?', '!', '"', "'"]
    tempCharList = [' ' + c if c in punctuation else c for c in text]
    text = ''.join(tempCharList)
    text = text.lower()

    # build vocabulary
    words = text.split()
    vocabulary = list(set(words))
    vocabulary.sort()

    # labels of words from training text:
    wordLabels = [vocabulary.index(w) for w in words]

    # random indices from training text:
    indices = random.sample(range(len(words) - sequenceLength - 1), noSequences)

    inputs = [wordLabels[i: i + sequenceLength] for i in indices]
    targets = [wordLabels[i + sequenceLength] for i in indices]

    inputs = tr.tensor(inputs, dtype=tr.float)
    targets = tr.tensor(targets, dtype=tr.long)

    return vocabulary, inputs, targets


def generateText(desiredTextLength):
    while True:
        sentence = input('Introduceti %s cuvinte sau _quit: ' % sequenceLength)
        sentence.strip()
        if sentence == "_quit":
            break
        words = sentence.split()
        if len(words) != sequenceLength:
            continue

        try:
            inputLabels = [vocabulary.index(w) for w in words]
        except:
            print('Cuvintele introduse trebuie sa faca parte din vocabular.')
            continue

        sentence += ' '
        rnnInput = tr.tensor(inputLabels, dtype=tr.float)
        for i in range(desiredTextLength - sequenceLength):
            rnnOut = myRNN(rnnInput)
            outputLabel = rnnOut.argmax().item()
            outputWord = vocabulary[outputLabel]
            sentence += outputWord
            sentence += ' '
            rnnInput = tr.cat([rnnInput[1:], tr.tensor([outputLabel])])
        print(sentence)


if __name__ == '__main__':
    vocabulary, inputs, targets = read_data()

    myRNN = SimpleRNN(1, len(vocabulary), 16, 1)
    myRNN.train(inputs, targets)
    generateText(25)
    
    