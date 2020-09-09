# %%
import os
import sys
import pickle
import multiprocessing

import mne
from mne.decoding import Vectorizer
from sklearn import svm
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # noqa
import deploy
from local_tools import FileLoader, Enhancer

import numpy as np
import matplotlib.pyplot as plt

# %%
times = np.linspace(-0.10, 0.10, np.int(100 * 0.2) + 1)
times
s2 = 0.2 ** 2
cos = np.cos(2 * np.pi / 0.2 * times)
exp = np.exp(- times ** 2 / s2)
GABOR_KERNEL = cos * exp


results_dir = os.path.join('.', 'SVM_xdawn')
try:
    os.mkdir(results_dir)
except:
    pass
finally:
    assert(os.path.isdir(results_dir))


def get_X_y(epochs):
    # Get data [X] and label [y] from [epochs]
    X = epochs.get_data()
    y = epochs.events[:, -1]

    return X, y


def mvpa(name):
    # Perform MVPA
    # Setting
    CROP = (0, 0.8, 0.2)
    EVENTS = ['1', '2', '4']

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

        print('Baseline and Crop --------------------')
        train_epochs = train_epochs.crop(CROP[0], CROP[1])
        train_epochs.apply_baseline((CROP[0], CROP[2]))
        test_epochs = test_epochs.crop(CROP[0], CROP[1])
        test_epochs.apply_baseline((CROP[0], CROP[2]))

        print('Get data -----------------------------')
        X_train, y_train = get_X_y(train_epochs[EVENTS])
        X_test, y_test = get_X_y(test_epochs[EVENTS])

        print('Training -----------------------------')
        clf = svm.SVC(gamma='scale', kernel='rbf', class_weight='balanced')
        pipeline = make_pipeline(Vectorizer(), StandardScaler(), clf)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        print('Saving ------------------------------')
        try:
            times = train_epochs.times
        except:
            times = 0
        predicts.append(dict(y_pred=y_pred,
                             y_test=y_test,
                             X_test=X_test,
                             y_train=y_train,
                             X_train=X_train,
                             times=times))

    with open(os.path.join(results_dir,
                           f'{name}.pkl'), 'wb') as f:
        pickle.dump(predicts, f)
        pass


# %%

for idx in range(1, 11):
    name = f'MEG_S{idx:02d}'
    # mvpa(name)
    p = multiprocessing.Process(target=mvpa, args=(name,))
    p.start()


# %%
# name = 'MEG_S05'
# # Perform MVPA
# # Setting
# BASELINE = (None, 0)
# CROP = (0, 0.8)
# EVENTS = ['1', '2', '4']

# # Load epochs
# loader = FileLoader(name)
# loader.load_epochs(recompute=False)
# print(loader.epochs_list)

# # Prepare [predicts] for results
# predicts = []

# # Cross validation
# num_epochs = len(loader.epochs_list)
# for exclude in range(num_epochs):
#     # Start on separate training and testing dataset
#     print(f'---- {name}: {exclude} | {num_epochs} ----------------------')
#     includes = [e for e in range(
#         len(loader.epochs_list)) if not e == exclude]
#     excludes = [exclude]
#     train_epochs, test_epochs = loader.leave_one_session_out(includes,
#                                                              excludes)
#     print(train_epochs, test_epochs)

#     print('Xdawn --------------------------------')
#     enhancer = Enhancer(train_epochs=train_epochs,
#                         test_epochs=test_epochs)
#     train_epochs, test_epochs = enhancer.fit_apply()
#     train_epochs.apply_baseline((None, 0))
#     test_epochs.apply_baseline((None, 0))

#     X_train, y_train = get_X_y(train_epochs[EVENTS])
#     X_test, y_test = get_X_y(test_epochs[EVENTS])
#     print('Got data ----------------------------')
#     break


# # %%
# X_test.shape

# # %%
# fig, axes = plt.subplots(3, 1)
# for j, eid in enumerate([1, 2, 4]):
#     X = X_test[y_test == eid]
#     axes[j].plot(np.mean(X, axis=0).transpose())

# # %%
# epochs = mne.BaseEpochs(info=train_epochs[EVENTS].info,
#                         events=train_epochs[EVENTS].events,
#                         data=X_train,
#                         tmin=train_epochs.times[0],
#                         tmax=train_epochs.times[-1])

# # %%
# for eid in ['1', '2', '4']:
#     epochs[eid].average().plot_joint(title=f'New {eid}')
#     train_epochs[eid].average().plot_joint(title=f'Old {eid}')
# # %%

# # %%
