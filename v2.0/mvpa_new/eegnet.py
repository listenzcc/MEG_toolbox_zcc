# %%
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

# %%


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 12, (1, 272), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(12, False)

        # Layer 2
        self.padding1 = nn.ReplicationPad2d((9, 10, 1, 2))
        self.conv2 = nn.Conv2d(1, 4, (4, 20))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d((2, 5), (2, 5))

        # Layer 3
        self.padding2 = nn.ReplicationPad2d((5, 4, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 10))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d(2, 2)

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # 4 channels, 3 features, 8 timepoints
        self.fc1 = nn.Linear(4*3*8, 1)

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.view(-1, 4*3*8)
        x = torch.sigmoid(self.fc1(x))
        return x


# %%


def evaluate(model, X, Y, params=["acc"]):
    results = []
    batch_size = 100

    predicted = []

    for i in range(int(len(X)/batch_size)):
        s = i*batch_size
        e = i*batch_size+batch_size

        inputs = Variable(torch.from_numpy(X[s:e]).cuda(0))
        pred = model(inputs)

        predicted.append(pred.data.cpu().numpy())

    inputs = Variable(torch.from_numpy(X).cuda(0))
    predicted = model(inputs)

    predicted = predicted.data.cpu().numpy()

    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2*precision*recall / (precision+recall))
    return results
