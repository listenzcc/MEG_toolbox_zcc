# %%
import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import mne
from mne.decoding import Vectorizer

# import sklearn.svm as svm
# import sklearn.metrics as metrics
# import sklearn.decomposition.pca as pca
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import pca
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %%
try:
    display('Display is useable')
except:
    def display(*args, **kwargs):
        # Display method in python environ
        print(args)
        print(kwargs)


def select_events(epochs):
    # return [e for e in range(len(epochs))]
    # Select event 1
    # Select event 4 that near to event 1
    # Randomly select event 2
    # Init ratio of event 2
    n1 = len(epochs['1'])
    n2 = len(epochs['2'])
    ratio = n1 / n2 * 2

    # Selection
    selects = []
    for j, event in enumerate(epochs.events):
        # Event 1 and 4
        if event[-1] == 1:
            [selects.append(j + e) for e in [-2, -1, 0, 1, -2]]
            continue

        # Event 2
        if event[-1] == 2:
            if random.random() < ratio:
                selects.append(j)
            continue

    # Return
    return selects


def convert(X):
    shape = X.shape
    new_X = np.zeros((shape[0], shape[2], shape[2]))
    for j in range(shape[0]):
        new_X[j] = np.dot(X[j].transpose(), X[j])
    return new_X


def get_X_y(epochs):
    # Get data [X] and label [y] from [epochs]
    X = epochs.get_data()
    # X = convert(X)
    y = epochs.events[:, 2]
    print(f'Got X: {X.shape}, y: {y.shape}')
    return X, y


# %% Walk through dir -------------------------------------
# Files should be in [data_folder]
# List the unique ids [uids]
# An uid is like 'MEG_S02-3' refers MEG data, subject 02 and 3 session exclusion
data_folder = os.path.join('..', 'MVPA_data_xdawn_raw_v3')
uids = sorted(set(e[:9] for e in os.listdir(data_folder)))
print(uids)

# split_folder = os.path.join('..', 'MVPA_data_xdawn_split_v3')
# uids = sorted(set(e[:9] for e in os.listdir(split_folder)))
# print(uids)

# Prepare results folder
results_folder = os.path.join('Results_raw_crop')
if not os.path.exists(results_folder):
    os.mkdir(results_folder)

# %% Main loop ----------------------------------------------
# For each uid
for uid in uids:
    uid = uids[8]
    result_name = f'{uid}-crop.pkl'
    if os.path.exists(os.path.join(results_folder, result_name)):
        print(f'{result_name} exists, doing nothing.')
        pass
        # continue

    # Init train and test epochs -----------------------
    train_epochs_name = f'{uid}-train-epo.fif'
    test_epochs_name = f'{uid}-test-epo.fif'

    # Read train and test epochs -----------------------
    train_epochs = mne.read_epochs(os.path.join(
        data_folder, train_epochs_name))['1', '2', '4']
    test_epochs = mne.read_epochs(os.path.join(
        data_folder, test_epochs_name))['1', '2', '4']
    times = test_epochs.times

    # Baseline correction
    train_epochs.apply_baseline((None, 0))
    test_epochs.apply_baseline((None, 0))

    # Crop
    train_epochs = train_epochs.crop(0.2, 0.8)
    test_epochs = test_epochs.crop(0.2, 0.8)

    # Select epochs in [train_epochs]
    selects = select_events(train_epochs)
    selected_train_epochs = train_epochs[selects]

    display(train_epochs, selected_train_epochs, test_epochs)

    # Get X and y
    train_X, train_y = get_X_y(selected_train_epochs)
    test_X, test_y = get_X_y(test_epochs)

    # Fit and pred ---------------------------------------------
    # Init
    clf = make_pipeline(Vectorizer(),
                        StandardScaler(),
                        pca.PCA(n_components=.95),
                        svm.SVC(gamma='scale',
                                kernel='rbf',
                                class_weight='balanced',))

    # Fit
    print('Fitting')
    clf.fit(X=train_X, y=train_y)

    # Predict
    print('Predicting')
    crop_pred_y = clf.predict(X=test_X)

    break

    # Save results
    print('Saving')
    results = dict(crop_pred_y=crop_pred_y,
                   test_y=test_y)

    pickle.dump(results,
                open(os.path.join(results_folder, result_name), 'wb'))


# %% Plotting -------------------------------------------------
# uid = 'MEG_S01-0'
print('-')
print(metrics.classification_report(y_true=test_y, y_pred=crop_pred_y))

# plt.style.use('ggplot')
# fig, ax = plt.subplots(1, 1)
# for key in results:
#     print(key)
#     ax.plot(times, results[key], label=key)
# ax.legend()


# %%
