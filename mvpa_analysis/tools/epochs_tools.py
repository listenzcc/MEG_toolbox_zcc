# File: epochs_tools.py
# Aim: Easy-to-use epochs tools

import numpy as np


def epochs_get_MVPA_data(epochs):
    # Get X, y[,z] from epochs
    # X: Data matrix, size is sample x channel x times
    # y: Label vector, size is sample
    # z: Session vector, size is sample (only available if epochs is list)
    # epochs is <list>: Multi session mode, return X, y and z
    # epochs is mne.Epochs: Single session mode, return X and y

    if isinstance(epochs, list):
        # Multi session mode
        # Get X and y from a list of epochs objects
        Xs, ys, zs = [], [], []
        for j, epo in enumerate(epochs):
            X = epo.get_data()
            y = epo.events[:, -1]
            z = y * 0 + j
            Xs.append(X)
            ys.append(y)
            zs.append(z)
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        z = np.concatenate(zs)
        print(f'Got X {X.shape}, y {y.shape} and z {z.shape} from {epochs}')
        return X, y, z

    # Single session mode
    # Get X and y from single epochs object
    X = epochs.get_data()
    y = epochs.events[:, -1]
    print(f'Got X {X.shape} and y {y.shape} from {epochs}')
    return X, y
