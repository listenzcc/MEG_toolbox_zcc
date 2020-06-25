# %% Importing
# System -----------------------------------------
import os
import sys
import pickle

# Computing --------------------------------------
# numpy
import numpy as np

# MNE
import mne
from mne.decoding import GeneralizingEstimator

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

# Local Settings --------------------------------------
# Number of parall jobs
n_jobs = 48

# Init results_folder
RESULTS_FOLDER = os.path.join('.', 'results')

# Make sure RESULTS_FOLDER is not a file
assert(not os.path.isfile(RESULTS_FOLDER))

# Mkdir if RESULTS_FOLDER does not exist
if not os.path.exists(RESULTS_FOLDER):
    os.mkdir(RESULTS_FOLDER)


def pair_X_y(epochs, label):
    """pair_X_y Get paired X and y,

    Make sure the epochs has only one class of data,
    the label is the int representation of the class.

    Args:
        epochs ({Epochs}): The epochs object to get data from
        label ({int}): The label of the data

    Returns:
        [type]: [description]
    """
    X = epochs.get_data()
    num = X.shape[0]
    y = np.zeros(num,) + label
    print(f'Got paired X: {X.shape} and y: {y.shape}')
    return X, y


# %%
for idx in range(1, 11):
    # Loading data ------------------------------------------
    running_name = f'MEG_S{idx:02d}'
    band_name = 'U07'

    worker = MEG_Worker(running_name=running_name)
    worker.pipeline(band_name=band_name)

    # MVPA ----------------------------------------------------------------
    # Prepare classifiers
    _svm = svm.SVC(gamma='scale', kernel='rbf', class_weight='balanced')
    clf = make_pipeline(StandardScaler(), _svm)
    estimator = GeneralizingEstimator(clf, n_jobs=n_jobs,
                                      scoring='f1', verbose=1)

    # Prepare paired X and y
    # Get X and y for class 1
    X1, y1 = pair_X_y(worker.clean_epochs, 1)

    # Get X and y for class 2
    X2, y2 = pair_X_y(worker.denoise_epochs['2'], 2)

    # Concatenate X and y
    X_all = np.concatenate([X1, X2], axis=0)
    y_all = np.concatenate([y1, y2], axis=0)

    # Get time line
    times = worker.clean_epochs.times

    # Estimate n_splits
    n_splits = int(y1.shape[0] / 56)
    print(f'Splitting in {n_splits} splits')

    # Cross validation using sliding window -------------------------------
    # Prepare predicted label matrix
    num_samples, num_times = X_all.shape[0], X_all.shape[2]
    y_pred_generalizing = np.zeros((num_samples, num_times, num_times))

    # Cross validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    for train_index, test_index in skf.split(X_all, y_all):
        # Separate training and testing data
        X_train, y_train = X_all[train_index], y_all[train_index]
        X_test, y_test = X_all[test_index], y_all[test_index]

        # Fit estimator
        estimator.fit(X_train, y_train)

        # Predict
        y = estimator.predict(X_test)
        y_pred_generalizing[test_index] = y

    # Summary results
    output_dict = dict(
        times=times,
        y_all=y_all,
        y_pred_generalizing=y_pred_generalizing,
    )

    # Save results
    with open(os.path.join(RESULTS_FOLDER,
                           f'{running_name}_generalizing.pkl'), 'wb') as f:
        pickle.dump(output_dict, f)


# %%
