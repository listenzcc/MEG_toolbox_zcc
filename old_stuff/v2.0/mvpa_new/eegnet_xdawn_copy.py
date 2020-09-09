# %%
import matplotlib.pyplot as plt
import os
import sys
import pickle
import numpy as np
import multiprocessing

import mne
from mne.decoding import Vectorizer
import sklearn.metrics
from sklearn import svm
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # noqa
import deploy
from local_tools import FileLoader, Enhancer

import torchsummary
from eegnet import torch, nn, EEGNet, optim, Variable, evaluate
net = EEGNet().cuda()
print(net.forward(Variable(torch.Tensor(np.random.rand(3, 1, 81, 272)).cuda())))
torchsummary.summary(net, input_size=(1, 81, 272))

# %%


class EEGNet_classifier():
    def __init__(self):
        self.net = EEGNet().cuda()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=20,
                                                   gamma=0.8)

    def fit(self, X, y, quiet=True):
        for epoch in range(200):
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
                print(f'Epoch {epoch}: {running_loss}')
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}: {running_loss}')

    def predict(self, X):
        inputs = torch.from_numpy(X)
        inputs = Variable(inputs.cuda())
        return self.net(inputs)

# %%


results_dir = os.path.join('.', 'EEGnet_xdawn')
try:
    os.mkdir(results_dir)
except:
    pass
finally:
    assert(os.path.isdir(results_dir))


def prepare_epochs(epochs,
                   events=['1', '2', '4'],
                   baseline=(None, 0),
                   crop=(0.0, 0.8)):
    # A tool for prepare epochs
    epochs = epochs[events]
    epochs.apply_baseline(baseline)
    return epochs.crop(crop[0], crop[1])


def relabel(events, sfreq=1200, T=0.5):
    """Re-label 2-> 4 when 2 is near to 1

    Arguments:
        events {array} -- The events array, [[idx], 0, [label]],
                          assume the [idx] column has been sorted.
        sfreq {float} -- The sample frequency

    Returns:
        {array} -- The re-labeled events array
    """
    events[events[:, -1] == 4, -1] = 2

    # Init the pointer [j]
    j = 0
    # Repeat for every '1' event, remember as [a]
    for a in events[events[:, -1] == 1]:
        # Do until...
        while True:
            # Break,
            # if [j] is far enough from latest '2' event,
            # it should jump to next [a]
            if events[j, 0] > a[0] + sfreq * T:
                break
            # Switch '2' into '4' event if it is near enough to the [a] event
            if all([events[j, -1] == 2,
                    abs(events[j, 0] - a[0]) < sfreq * T]):
                events[j, -1] = 4
            # Add [j]
            j += 1
            # If [j] is out of range of events,
            # break out the 'while True' loop.
            if j == events.shape[0]:
                break
    # Return re-labeled [events]
    return events


def get_X_y(epochs):
    # Get data [X] and [y] from [epochs]
    X = epochs.get_data()
    y = relabel(epochs.events)[:, -1]
    return (X.astype('float32'),
            y.astype('float32'))


def mvpa(name):
    # Perform MVPA
    # Load epochs
    loader = FileLoader(name)
    loader.load_epochs(recompute=False)
    print(loader.epochs_list)

    # Prepare [predicts] for results
    predicts = []

    # Cross validation
    num_epochs = len(loader.epochs_list)
    for exclude in range(num_epochs):
        # Start on separate training and testing dataset
        print(f'---- {name}: {exclude} | {num_epochs} ----------------------')
        includes = [e for e in range(
            len(loader.epochs_list)) if not e == exclude]
        excludes = [exclude]
        train_epochs, test_epochs = loader.leave_one_session_out(includes,
                                                                 excludes)
        print(train_epochs, test_epochs)

        print('Xdawn --------------------------------')
        enhancer = Enhancer(train_epochs=train_epochs,
                            test_epochs=test_epochs)
        train_epochs, test_epochs = enhancer.fit_apply()

        # Prepare epochs
        train_epochs = prepare_epochs(train_epochs)
        test_epochs = prepare_epochs(test_epochs)

        X_train, y_train = get_X_y(train_epochs)
        X_test, y_test = get_X_y(test_epochs)

        print('Preprocess ---------------------------')
        pipeline = make_pipeline(Vectorizer(), StandardScaler())
        X_train = pipeline.fit(X_train)
        X_test = pipeline.fit(X_test)
        y_train[not y_train == 1] = 0
        y_test[not y_test == 1] = 0

        print('Training -----------------------------')
        eegnet = EEGNet_classifier()
        eegnet.fit(X_train, y_train, quiet=False)
        y_pred = eegnet.predict(X_test)
        # y_pred = y_test

        print('Testing ------------------------------')
        predicts.append(dict(y_test=y_test,
                             y_pred=y_pred))

    with open(os.path.join(results_dir,
                           f'{name}.json'), 'wb') as f:
        pickle.dump(predicts, f)

        pass


# %%

for idx in range(1, 11):
    name = f'MEG_S{idx:02d}'
    # mvpa(name)
    # p = multiprocessing.Process(target=mvpa, args=(name,))
    # p.start()


# %%
idx = 3
name = f'MEG_S{idx:02d}'
# Perform MVPA
# Load epochs
loader = FileLoader(name)
loader.load_epochs(recompute=False)
print(loader.epochs_list)

exclude = 2
# Start on separate training and testing dataset
includes = [e for e in range(
    len(loader.epochs_list)) if not e == exclude]
excludes = [exclude]
train_epochs, test_epochs = loader.leave_one_session_out(includes,
                                                         excludes)
print(train_epochs, test_epochs)

print('Xdawn --------------------------------')
enhancer = Enhancer(train_epochs=train_epochs,
                    test_epochs=test_epochs)
train_epochs, test_epochs = enhancer.fit_apply()

# Prepare epochs
train_epochs = prepare_epochs(train_epochs)
test_epochs = prepare_epochs(test_epochs)


# %%
X_train, y_train = get_X_y(train_epochs)
X_test, y_test = get_X_y(test_epochs)
m = np.max(abs(X_train))
display(m)
X_train = X_train / m
X_test = X_test / m

# %%
print('Training -----------------------------')
_X_train = X_train.transpose([0, 2, 1])[:, np.newaxis]
_X_train = _X_train[y_train != 4]
_y_train = y_train[y_train != 4]
_y_train[_y_train != 1] = 0

eegnet12 = EEGNet_classifier()
eegnet12.fit(_X_train, _y_train, quiet=False)

# %%
print('Training -----------------------------')
_X_train = X_train.transpose([0, 2, 1])[:, np.newaxis]
_X_train = _X_train[y_train != 2]
_y_train = y_train[y_train != 2]
_y_train[_y_train != 1] = 0

eegnet14 = EEGNet_classifier()
eegnet14.fit(_X_train, _y_train, quiet=False)

# %%
print('Training -----------------------------')
_X_train = X_train.transpose([0, 2, 1])[:, np.newaxis]
_y_train = y_train.copy()
_y_train[_y_train != 1] = 0

eegnet124 = EEGNet_classifier()
eegnet124.fit(_X_train, _y_train, quiet=False)

# %%
_X_test = X_test.transpose([0, 2, 1])[:, np.newaxis]
y_pred12 = eegnet12.predict(_X_test).detach().cpu().numpy()
y_pred14 = eegnet14.predict(_X_test).detach().cpu().numpy()
y_pred124 = eegnet124.predict(_X_test).detach().cpu().numpy()
y_pred_mix = (y_pred12 + y_pred14 + y_pred124) / 3

y_pred = y_pred_mix.copy()
events = test_epochs.events.copy()
for j, es in enumerate(events):
    print(j)
    finds = []
    for k, e in enumerate(events[max(j-20, 0):min(j+20, len(events))]):
        d = np.abs(es[0] - e[0])
        if d < 1200:
            finds.append(y_pred_mix[j-20+k])
    if not y_pred_mix[j] == np.max(finds):
        y_pred[j] = 0


# %%
order = np.argsort(y_test)
order = range(len(order))
# order = np.where(y_test == 1)[0]
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(y_pred12[order], label='12')
ax.plot(1+y_pred14[order], label='14')
ax.plot(2+y_pred124[order], label='124')
ax.plot(3+y_pred_mix[order], label='mix')
ax.plot(-y_test[order], label='y')
ax.plot(-y_pred[order])
ax.legend()

y_true = y_test.copy()
y_true[y_true != 1] = 0

print('12')
print(sklearn.metrics.classification_report(y_pred=y_pred12 > 0.5,
                                            y_true=y_true))

print('14')
print(sklearn.metrics.classification_report(y_pred=y_pred14 > 0.5,
                                            y_true=y_true))

print('124')
print(sklearn.metrics.classification_report(y_pred=y_pred124 > 0.5,
                                            y_true=y_true))

print('mix')
print(sklearn.metrics.classification_report(y_pred=y_pred_mix > 0.5,
                                            y_true=y_true))

print('pred')
print(sklearn.metrics.classification_report(y_pred=y_pred > 0.5,
                                            y_true=y_true))


# %%

TP, TN, FP, FN = 0, 0, 0, 0
for j, _y in enumerate(y_test):
    p12 = y_pred12[j] > 0.5
    p14 = y_pred14[j] > 0.5
    p124 = y_pred124[j] > 0.5

    if _y == 1:
        if any([p12, p14, p124]):
            TP += 1
        else:
            FN += 1
    else:
        if not all([p12, p14, p124]):
            TN += 1
        else:
            FP += 1

print('--------------------------------------------')
print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')

print('Recall:', TP / (TP + FN))
print('Precision:', TP / (TP + FP))

# %%
finds_1 = np.where(y_test == 1)
# finds_2 = np.where(y_test == 2)
finds_4 = np.where(y_test == 4)
finds = np.concatenate([
    finds_1[0],
    # finds_2[0],
    finds_4[0],
])

fig, ax = plt.subplots(1, 1)
ax.plot(-y_pred[finds])
ax.plot(_y_test[finds])

report = sklearn.metrics.classification_report(y_pred=y_pred[finds] > 0.5,
                                               y_true=_y_test[finds],
                                               output_dict=True)
display(report)

# %%
sklearn.metrics.roc_auc_score(_y_test, y_pred)

# %%
