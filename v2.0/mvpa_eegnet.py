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


class TrainSessions(object):
    def __init__(self, train_data, train_label):
        self.data = train_data
        self.label = train_label
        self.idxs = dict(
            targets=np.where(train_label == 1)[0],
            far_non_targets=np.where(train_label == 2)[0],
            near_non_targets=np.where(train_label == 4)[0],
            # button=np.where(train_label == 3)[0]
        )

    def _shuffle(self):
        for key in self.idxs:
            np.random.shuffle(self.idxs[key])

    def random(self, num=10):
        self._shuffle()
        selects = np.concatenate([self.idxs[e][:num]
                                  for e in self.idxs])
        return self.data[selects], self.label[selects]


class OneHotVec(object):
    # Onehot vector encoder and decoder
    def __init__(self, coding={1: 0, 2: 1, 4: 2}):
        # Init
        # coding: coding table
        self.coding = coding
        # Generate inverse coding table
        self.coding_inv = dict()
        for c in coding:
            self.coding_inv[coding[c]] = c
        # length: The length of the onehot code
        self.length = len(coding)

    def encode(self, vec):
        vec = np.ravel(vec)
        # num: How many vectors to be encoded
        num = len(vec)
        # code: matrix size of num x self.length
        code = np.zeros((num, self.length))
        # Set up code
        for j, v in enumerate(vec):
            code[j, self.coding[v]] = 1
        return code

    def decode(self, code):
        # num: How many codes to be decoded
        num = len(code)
        # vec: vector size of num
        vec = np.zeros((num, 1))
        # Set up vec
        for j, v in enumerate(code):
            vec[j] = self.coding_inv[np.where(v == max(v))[0][0]]
        return vec


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.C = 272
        self.T = 120
        self.set_parameters()

    def set_parameters(self):
        # Layer 1
        self.conv1 = nn.Conv2d(1, 25, (1, 5), padding=(0, 2))
        self.conv11 = nn.Conv2d(25, 25, (272, 1), padding=(0, 0))
        self.batchnorm1 = nn.BatchNorm2d(25, False)
        self.pooling1 = nn.MaxPool2d(1, 2)

        # Layer 2
        self.conv2 = nn.Conv2d(25, 50, (1, 5), padding=(0, 2))
        self.batchnorm2 = nn.BatchNorm2d(50, False)
        self.pooling2 = nn.MaxPool2d(1, 2)

        # Layer 3
        self.conv3 = nn.Conv2d(50, 100, (1, 5), padding=(0, 2))
        self.batchnorm3 = nn.BatchNorm2d(100, False)
        self.pooling3 = nn.MaxPool2d(1, 2)

        # Layer 4
        self.conv4 = nn.Conv2d(100, 200, (1, 5), padding=(0, 2))
        self.batchnorm4 = nn.BatchNorm2d(200, False)
        self.pooling4 = nn.MaxPool2d(1, 2)

        # FC layer
        self.fc = nn.Linear(200 * 8, 3)

    def predict(self, x):
        return self.forward(x, dropout=0)

    def max_norm(self):
        # def max_norm(model, max_val=2, eps=1e-8):
        #     for name, param in model.named_parameters():
        #         if 'bias' not in name:
        #             norm = param.norm(2, dim=0, keepdim=True)
        #             # desired = torch.clamp(norm, 0, max_val)
        #             param = torch.clamp(norm, 0, max_val)
        #             # param = param * (desired / (eps + norm))
        eps = 1e-8
        for name, param in self.named_parameters():
            if 'bias' in name:
                continue

            max_val = 2
            if any([name.startswith(e) for e in ['conv1', 'conv11', 'conv2', 'conv3', 'conv4']]):
                norm = param.norm(2, dim=None, keepdim=True)
                desired = torch.clamp(norm, 0, max_val)
                param = param * desired / (eps + norm)
                continue

            max_val = 0.5
            if name.startswith('fc'):
                norm = param.norm(2, dim=None, keepdim=True)
                desired = torch.clamp(norm, 0, max_val)
                param = param * desired / (eps + norm)
                continue

    def feature_1(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.batchnorm1(x)
        return x

    def forward(self, x, dropout=0.25):
        # Layer 1
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.pooling1(x)

        x = F.dropout(x, dropout)

        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling2(x)

        x = F.dropout(x, dropout)

        # Layer 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling3(x)

        x = F.dropout(x, dropout)

        # Layer 4
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.elu(x)
        x = self.pooling4(x)

        x = F.dropout(x, dropout)

        # FC Layer
        x = x.view(-1, 200 * 8)
        x = self.fc(x)
        x = F.softmax(x)

        return x


net = EEGNet().cuda()
input_size = (1, 272, 120)
print('-' * 80)
print('Input size is {}, {}'.format(-1, input_size))
torchsummary.summary(net, input_size=input_size)

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
for name in ['MEG_S01', 'MEG_S02', 'MEG_S03', 'MEG_S04', 'MEG_S05', 'MEG_S06', 'MEG_S07', 'MEG_S08', 'MEG_S09', 'MEG_S10']:
    # Load MEG data
    name = 'MEG_S02'
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

        # for band in bands:
        # Select events of ['1', '2', '4']
        train_epochs = include_epochs['1', '2', '4']
        test_epochs = exclude_epochs['1', '2', '4']

        # train_epochs.filter(bands[band][0], bands[band][1], n_jobs=n_jobs)
        # test_epochs.filter(bands[band][0], bands[band][1], n_jobs=n_jobs)

        # Xdawn preprocessing -----------------------------
        # Fit xdawn
        # xdawn.fit(train_epochs)

        # Apply baseline
        # !!! Make sure applying baseline **AFTER** xdawn fitting
        train_epochs.apply_baseline((None, 0))
        test_epochs.apply_baseline((None, 0))

        # Apply xdawn
        # no_xdawn
        train_data = train_epochs.get_data()
        test_data = test_epochs.get_data()
        # train_data = xdawn.apply(train_epochs)['1'].get_data()
        # test_data = xdawn.apply(test_epochs)['1'].get_data()

        # Get labels and select events
        train_label = train_epochs.events[:, -1]
        test_label = test_epochs.events[:, -1]
        # train_label[train_label == 4] = 2
        # test_label[test_label == 4] = 2

        # Just print something to show data have been prepared
        print(train_data.shape, train_label.shape,
              test_data.shape, test_label.shape)

        times = train_epochs.times
        tmin, tmax = -0.2, 1

        selects = [j for j, e in enumerate(times)
                   if all([e < tmax,
                           e >= tmin])]
        _train_data = train_data[:, :, selects]
        _test_data = test_data[:, :, selects]
        # Size is [-1, 272, 120]

        # EEG net MVPA ------------------------------------------
        # _train_data = np.load('_train_data.npy')
        # _test_data = np.load('_test_data.npy')
        # train_label = np.load('train_label.npy')
        # test_label = np.load('test_label.npy')

        ohv = OneHotVec()
        _max, _min = np.max(_train_data), np.min(_train_data)
        X_test = numpy2torch(
            (_test_data[:, np.newaxis, :] + _min) / (_max - _min))
        y_test = numpy2torch(ohv.encode(test_label[:, np.newaxis]))

        # Fit net
        net = EEGNet().cuda()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(),
                               lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.8)

        tsession = TrainSessions(_train_data, train_label)
        for session_id in range(500):
            data, label = tsession.random(num=100)
            X = numpy2torch((data[:, np.newaxis, :] + _min) / (_max - _min))
            y_true = numpy2torch(ohv.encode(label[:, np.newaxis]))

            for _ in range(1):
                optimizer.zero_grad()
                y = net.forward(X)
                _y = net.forward(X_test)
                _loss = criterion(_y, y_test)
                loss = criterion(y, y_true)
                loss.backward()
                optimizer.step()
                scheduler.step()
                net.max_norm()

            if session_id % 10 == 0:
                print('{idx}, Loss: {loss:.6f}, test loss: {test_loss:.6f}'.format(
                    **dict(idx=session_id,
                           loss=loss.item(),
                           test_loss=_loss.item())))

        print('EEG net training is done.')
        stophere

        # Predict using EEG net
        y = torch2numpy(net.predict(X_test))
        y_dict = dict(y_true=ohv.decode(torch2numpy(y_test)),
                      y_pred=ohv.decode(y),
                      y_prob=y)
        # print('Classification report\n', metrics.classification_report(**y_dict))
        # print('Confusion matrix\n', metrics.confusion_matrix(**y_dict))

        # Restore labels
        labels.append(y_dict)

        # Print something to show MVPA in this folder is done
        print(f'---- {name} ---------------------------')
        print(metrics.classification_report(y_true=labels[-1]['y_true'],
                                            y_pred=labels[-1]['y_pred'],))

    # Save labels of current [name]
    frame = pd.DataFrame(labels)
    frame.to_json(f'no_xdawn_eegnet_3classes/{name}.json')
    print(f'{name} MVPA is done')
    # break

print('All done.')

# %%
net

# %%
for c in net.conv11.parameters():
    break
# %%
c = torch2numpy(c)
c = c[:, :, :, 0]
c.shape


# %%
evoked = test_epochs['1'].average()
times = evoked.times
evoked.data[:, 0] = c[0, 0]
evoked.plot_topomap(times=[-0.2])
# %%
X.shape
# %%
X[:100].shape

# %%
f = net.feature_1(X[:100])
f.shape

# %%
