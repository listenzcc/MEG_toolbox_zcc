# %% Importing
# System -----------------------------------------
import os
import sys

# Computing --------------------------------------
# numpy
import numpy as np

# MNE
import mne
from mne.decoding import SlidingEstimator

# sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# Plotting --------------------------------------
import matplotlib.pyplot as plt

# Local tools -----------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))  # noqa
from MEG_worker import MEG_Worker

# Settings --------------------------------------
n_jobs = 48

# %%
# for idx in range(1, 11):

# Loading data ------------------------------------------
idx = 3
running_name = f'MEG_S{idx:02d}'
band_name = 'U07'

worker = MEG_Worker(running_name=running_name)
worker.pipeline(band_name=band_name)

# %%
# input('Press enter to escape.')

# %%
# MVPA ----------------------------------------------------------------
svm = svm.SVC(gamma='scale', kernel='rbf', class_weight='balanced')
clf = make_pipeline(StandardScaler(), svm)
estimator = SlidingEstimator(clf, n_jobs=n_jobs, scoring='f1', verbose=1)

# %%


def pair_X_y(epochs, label):
    X = epochs.get_data()
    num = X.shape[0]
    y = np.zeros(num,) + label
    print(f'Got paired X: {X.shape} and y: {y.shape}')
    return X, y


X1, y1 = pair_X_y(worker.clean_epochs, 1)
X2, y2 = pair_X_y(worker.denoise_epochs['2'], 2)

X_all = np.concatenate([X1, X2], axis=0)
y_all = np.concatenate([y1, y2], axis=0)
X_all.shape, y_all.shape

times = worker.clean_epochs.times

n_splits = int(y1.shape[0] / 64)

y_pred_time = np.zeros((3313, 141))

skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
for train_index, test_index in skf.split(X_all, y_all):
    X_train, y_train = X_all[train_index], y_all[train_index]
    X_test, y_test = X_all[test_index], y_all[test_index]

    estimator.fit(X_train, y_train)
    y_pred_time[test_index] = estimator.predict(X_test)


# %%
y_pred_time.shape


# %%
report = metrics.classification_report(y_true=y_all,
                                       y_pred=y_pred_time[:, 60],
                                       output_dict=True)
print(report)

# %%


def fuck_report(report):
    # I don't know why report has to be 2-layers dictionary,
    # this method is to make it human useable.
    fucked_report = dict()
    for key1 in report:
        if isinstance(report[key1], dict):
            for key2 in report[key1]:
                fucked_report[f'{key1}-{key2}'] = report[key1][key2]
        else:
            fucked_report[key1] = report[key1]

    keys = [e for e in fucked_report.keys()]
    for key in keys:
        if key.endswith('support'):
            fucked_report.pop(key)

    return fucked_report


def get_time_report(y_true, y_pred_time):
    report = metrics.classification_report(y_true=y_true,
                                           y_pred=y_pred_time[:, 0],
                                           output_dict=True)
    time_report = fuck_report(report)
    for key in time_report:
        print(key)
        time_report[key] = []

    for j, y_pred in enumerate(y_pred_time.transpose()):
        report = metrics.classification_report(y_true=y_true,
                                               y_pred=y_pred,
                                               output_dict=True)
        report = fuck_report(report)

        for key in time_report:
            time_report[key].append(report[key])

    return time_report


time_report = get_time_report(y_true=y_all,
                              y_pred_time=y_pred_time)


# %%
plt.style.use('ggplot')
fig, axes = plt.subplots(5, 1, figsize=(3, 15))

groups = ['1.0', '2.0', 'macro', 'weighted', 'accuracy']

for j, prefix in enumerate(groups):
    for key in [e for e in time_report if e.startswith(prefix)]:
        axes[j].plot(times, time_report[key], label=key)

for j in range(5):
    axes[j].legend(loc='upper left', bbox_to_anchor=(1, 1))


# %%
