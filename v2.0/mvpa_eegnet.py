# File: mvpa_eegnet.py
# Aim: Calculate MVPA baseline using EEG net

# %%
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim

import plotly.graph_objs as go
import plotly
import mne
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd

from tools.data_manager import DataManager

import torchsummary
# from eegnet import EEGNet, numpy2torch, torch2numpy, DEVICE
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%

DEVICE = 'cuda'


def numpy2torch(array, dtype=np.float32, device=DEVICE):
    # Attach [array] to the type of torch
    return torch.from_numpy(array.astype(dtype)).to(device)


def torch2numpy(tensor):
    # Detach [tensor] to the type of numpy
    return tensor.detach().cpu().numpy()


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        # ! Input size is (-1, 1, 100, 272)

        # Layer 1
        self.padding1 = nn.ZeroPad2d((0, 0, 0, 0))
        # ! Input size is (-1, 1, 100, 272)
        self.conv1 = nn.Conv2d(1, 12, (1, 272), padding=0)
        # ! Size is (-1, 12, 100, 1)
        self.batchnorm1 = nn.BatchNorm2d(12, False)
        # Permute as (0, 3, 1, 2)
        # ! Size is (-1, 1, 12, 100)

        # Layer 2
        self.padding2 = nn.ZeroPad2d((24, 25, 0, 0))
        # ! Size is (-1, 1, 12, 149)
        self.conv2 = nn.Conv2d(1, 4, (1, 50))
        # ! Size is (-1, 4, 12, 100)
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d((2, 5))
        # ! Size is (-1, 4, 6, 20)

        # Layer 3
        self.padding3 = nn.ZeroPad2d((1, 2, 1, 2))
        # ! Size is (-1, 4, 9, 23)
        self.conv3 = nn.Conv2d(4, 4, (4, 4))
        # ! Size is (-1, 4, 6, 20)
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((3, 2))
        # ! Size is (-1, 4, 2, 10)

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        self.fc_input_num = 4 * 2 * 10
        self.fc1 = nn.Linear(self.fc_input_num, 1)

    def predict(self, x):
        return self.forward(x, dropout=0)

    def forward(self, x, dropout=0.15):
        # Layer 1
        x = self.padding1(x)
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, dropout)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding2(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, dropout)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding3(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, dropout)
        x = self.pooling3(x)

        # FC Layer
        x = x.view(-1, self.fc_input_num)
        x = torch.sigmoid(self.fc1(x))
        return x


class TrainSessions(object):
    def __init__(self, train_data, train_label):
        self.data = train_data
        self.label = train_label
        self.idxs = dict(
            targets=np.where(train_label == 1)[0],
            far_non_targets=np.where(train_label == 2)[0],
            near_non_targets=np.where(train_label == 4)[0]
        )

    def _shuffle(self):
        for key in self.idxs:
            np.random.shuffle(self.idxs[key])

    def random(self, num=10):
        selects = np.concatenate([self.idxs[e][:num]
                                  for e in self.idxs])
        return self.data[selects], self.label[selects]


net = EEGNet().cuda()
input_size = (1, 100, 272)
print('-' * 80)
print('Input size is {}, {}'.format(-1, input_size))
torchsummary.summary(net, input_size=input_size)

# %%
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(),
                       lr=0.01)
# scheduler = optim.lr_scheduler.StepLR(optimizer,
#                                       step_size=20,
#                                       gamma=0.5)

# tsession = TrainSessions(_train_data, train_label)
# data, label = tsession.random()

# X = numpy2torch(data[:, np.newaxis, :].transpose([0, 1, 3, 2]))
# # X *= 1e12
# y_true = numpy2torch(label[:, np.newaxis])
# y_true[y_true != 1] = 0
# print(X.shape, y_true.shape)

# for j in range(100):
#     optimizer.zero_grad()
#     y = net.forward(X)
#     loss = criterion(y, y_true)
#     loss.backward()
#     optimizer.step()
#     if j % 10 == 0:
#         print(loss.item())


# fig, axes = plt.subplots(2, 1)
# axes[0].plot(torch2numpy(y))
# axes[1].plot(torch2numpy(y_true))
# print()

# %%
pass

# %%


class EEGNet_classifier():
    def __init__(self, net):
        self.net = net
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=20,
                                                   gamma=0.5)

    def fit(self, X, y, quiet=True):
        for epoch in range(100):
            running_loss = 0
            _skf = StratifiedKFold(n_splits=10, shuffle=True)
            for more, less in _skf.split(X, y):
                # Get data and labels
                inputs = torch.from_numpy(X[more])
                labels = torch.FloatTensor(np.array([y[more]]).T*1.0)

                # Make sure positive and negative samples have the same number
                inputs = inputs[:torch.sum(labels).type(torch.int16)*2]
                labels = labels[:torch.sum(labels).type(torch.int16)*2]

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.data

            self.scheduler.step()

            if not quiet:
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}: {running_loss}')

    def predict(self, X):
        inputs = torch.from_numpy(X)
        inputs = Variable(inputs.cuda())
        return self.net(inputs)


# %%

plotly.offline.init_notebook_mode(connected=True)


def plot(scatters, title='Title'):
    if isinstance(scatters, dict):
        scatters = [scatters]
    layout = go.Layout(title=title)
    data = [go.Scatter(**scatter) for scatter in scatters]
    plotly.offline.iplot(dict(data=data,
                              layout=layout))

# %%


def relabel_events(events):
    # Relabel events
    # Relabeled event:
    #  1: Target
    #  2: Far non-target
    #  3: Button motion
    #  4: Near non-target

    print(f'Relabel {len(events)} events')

    events[events[:, -1] == 4, -1] = 2

    for j, event in enumerate(events):
        if event[2] == 1:
            count, k = 0, 0
            while True:
                k += 1
                try:
                    if events[j+k, 2] == 2:
                        events[j+k, 2] = 4
                        count += 1
                except IndexError:
                    break
                if count == 5:
                    break

            count, k = 0, 0
            while True:
                k += 1
                try:
                    if events[j-k, 2] == 2:
                        events[j-k, 2] = 4
                        count += 1
                except IndexError:
                    break
                if count == 5:
                    break

    return events


class CV_split(object):
    # CV split object for MVPA analysis
    # dm: DataManager of MEG dataset
    # total: Total number of epochs sessions in [dm]

    def __init__(self, dm):
        self.dm = dm
        self.total = len(dm.epochs_list)
        self.reset()

    def reset(self):
        # Reset the index of testing session
        self.exclude = 0
        print(f'Reset CV_split, {self.total} splits to go.')

    def is_valid(self):
        # Return if the next split is valid
        return all([self.exclude > -1,
                    self.exclude < self.total])

    def next_split(self, n_components=6, svm_params=None):
        # Roll to next split
        # Generate includes and excludes epochs
        # Init necessary objects for MVPA analysis
        includes = [e for e in range(self.total) if e != self.exclude]
        excludes = [self.exclude]
        epochs_train, epochs_test = dm.leave_one_session_out(includes=includes,
                                                             excludes=excludes)
        self.exclude += 1
        print(f'New split, {self.exclude} | {self.total}')

        # Init xdawn object
        xdawn = mne.preprocessing.Xdawn(n_components=n_components)

        # Init classifier
        if svm_params is None:
            # Setup default parameters of SVM
            svm_params = dict(gamma='scale',
                              kernel='rbf',
                              class_weight='balanced')

        clf = make_pipeline(mne.decoding.Vectorizer(),
                            StandardScaler(),
                            PCA(n_components=.95),
                            svm.SVC(**svm_params))

        return dict(includes=epochs_train,
                    excludes=epochs_test,
                    xdawn=xdawn,
                    clf=clf)


# %%


bands = dict(raw=(0.1, 13),
             delta=(0.1, 3),
             theta=(3.5, 7.5),
             alpha=(7.5, 13))

n_jobs = 48

# %%
for name in ['MEG_S02', 'MEG_S03', 'MEG_S04', 'MEG_S05']:
    # Load MEG data
    dm = DataManager(name)
    dm.load_epochs(recompute=False)

    # Init cross validation
    cv = CV_split(dm)
    cv.reset()

    # MVPA parameters
    n_components = 6

    # Cross validation
    # y_pred and y_true will be stored in [labels]
    labels = []

    while cv.is_valid():
        # Recursive
        # Get current split
        split = cv.next_split(n_components)
        include_epochs = split['includes']
        exclude_epochs = split['excludes']

        # Get scaler, xdawn and clf
        xdawn = split['xdawn']
        clf = split['clf']

        # Re-label the events
        include_epochs.events = relabel_events(include_epochs.events)
        exclude_epochs.events = relabel_events(exclude_epochs.events)

        labels.append(dict())

        # for band in bands:
        # Select events of ['1', '2', '4']
        train_epochs = include_epochs['1', '2', '4']
        test_epochs = exclude_epochs['1', '2', '4']

        # train_epochs.filter(bands[band][0], bands[band][1], n_jobs=n_jobs)
        # test_epochs.filter(bands[band][0], bands[band][1], n_jobs=n_jobs)

        # Xdawn preprocessing -----------------------------
        # Fit xdawn
        xdawn.fit(train_epochs)

        # Apply baseline
        # !!! Make sure applying baseline **AFTER** xdawn fitting
        train_epochs.apply_baseline((None, 0))
        test_epochs.apply_baseline((None, 0))

        # Apply xdawn
        train_data = xdawn.apply(train_epochs)['1'].get_data()
        test_data = xdawn.apply(test_epochs)['1'].get_data()

        # Get labels and select events
        train_label = train_epochs.events[:, -1]
        test_label = test_epochs.events[:, -1]

        # Just print something to show data have been prepared
        print(train_data.shape, train_label.shape,
              test_data.shape, test_label.shape)

        times = train_epochs.times
        tmin, tmax = 0, 1

        selects = [j for j, e in enumerate(times)
                   if all([e < tmax,
                           e >= tmin])]
        _train_data = train_data[:, :, selects]
        _test_data = test_data[:, :, selects]
        # Size is [-1, 272, 100]

# %%
# EEG net MVPA ------------------------------------------
X_test = numpy2torch(
    _test_data[:, np.newaxis, :].transpose([0, 1, 3, 2])) * 1e12

y_test = numpy2torch(test_label[:, np.newaxis])
y_test[y_test != 1] = 0

# Fit net
net = EEGNet().cuda()

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(),
                        lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=100,
                                      gamma=0.8)

tsession = TrainSessions(_train_data, train_label)
for session_id in range(300):
    data, label = tsession.random(num=100)
    X = numpy2torch(
        data[:, np.newaxis, :].transpose([0, 1, 3, 2])) * 1e12
    y_true = numpy2torch(label[:, np.newaxis])
    y_true[y_true != 1] = 0

    for _ in range(1):
        optimizer.zero_grad()
        y = net.forward(X)
        _y = net.forward(X_test)
        _loss = criterion(_y, y_test)
        loss = criterion(y, y_true)
        loss.backward()
        optimizer.step()
        scheduler.step()

    if session_id % 10 == 0:
        print('{idx}, Loss: {loss:.6f}, test loss: {test_loss:.6f}'.format(
            **dict(idx=session_id,
        loss=loss.item(),
        test_loss=_loss.item())))

print('EEG net training is done.')

# Predict using EEG net
X = numpy2torch(
    _test_data[:, np.newaxis, :].transpose([0, 1, 3, 2])) * 1e12
y_true = test_label.copy()
y_true[y_true != 1] = 0
y = np.ravel(torch2numpy(net.predict(X)))
y_pred = np.fix(y + 0.5)
print(metrics.classification_report(y_true=y_true,
                                    y_pred=y_pred))
stophere

# %%

        # Restore labels
        labels[-1]['y_true'] = test_label
        labels[-1]['y_pred'] = label

        # Print something to show MVPA in this folder is done
        print(f'---- {name} ---------------------------')
        print(metrics.classification_report(y_true=labels[-1]['y_true'],
                                            y_pred=labels[-1]['y_pred'],))

    # Save labels of current [name]
    frame=pd.DataFrame(labels)
    frame.to_json(f'{name}.json')
    print(f'{name} MVPA is done')
    # break

print('All done.')

# %%

for name in ['MEG_S02', 'MEG_S03', 'MEG_S04', 'MEG_S05']:
    print('-' * 80)
    print(name)

    try:
        frame=pd.read_json(f'{name}.json')
    except:
        continue

    y_true=np.concatenate(frame.y_true.to_list())
    y_pred=np.concatenate(frame.y_pred.to_list())
    print('Classification report\n',
          metrics.classification_report(y_pred=y_pred, y_true=y_true))
    print('Confusion matrix\n',
          metrics.confusion_matrix(y_pred=y_pred, y_true=y_true))


# %%
# plot([dict(y=y_true, name='True'),
#       dict(y=2-y_pred, name='Pred')])

# %%
# epochs_1, epochs_2 = dm.leave_one_session_out(includes=[1, 3, 5],
#                                               excludes=[2, 4, 6])
# epochs_1, epochs_2

# # %%
# event = '2'
# epochs = epochs_1[event]
# epochs.filter(l_freq=0.1, h_freq=7, n_jobs=48)

# epochs_1[event].average().plot_joint(title=event)
# print()

# epochs.average().plot_joint(title=event)
# print()

# # %%
# xdawn = mne.preprocessing.Xdawn(n_components=6)
# xdawn.fit(epochs_1)
# xdawn_epochs_1 = xdawn.apply(epochs_1)
# xdawn_epochs_2 = xdawn.apply(epochs_2)
# xdawn_epochs_1, xdawn_epochs_2

# # %%
# for event in ['1', '2', '3']:
#     epochs_1[event].average().plot_joint(title=event)
#     xdawn_epochs_1['3'][event].average().plot_joint(title=event)
# print()
# # %%
# help(xdawn.apply)
# # %%
# xdawn.apply(epochs_1, ['1', '2'])
# # %%
