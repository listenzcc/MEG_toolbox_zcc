# File: filter_toeplitz.py
# Aim: Filter the epochs using toeplitz matrix.
# X = D \cdot R \cdot S
# X: Epoch data, times x sensors(272)
# D: Toeplitz matrix, times x 600ms
# R: ERP temporal pattern, 600ms x channels
# S: ERP spatial pattern, channels x sensors

# %%
import mne
import time
import numpy as np
import matplotlib.pyplot as plt
from tools.data_manager import DataManager
from tools.epochs_tools import get_MVPA_data

# %%
name = 'MEG_S02'

dm = DataManager(name)
dm.load_epochs(recompute=False)

# %%
epochs_1, epochs_2 = dm.leave_one_session_out(includes=[1, 3, 5],
                                              excludes=[2, 4, 6])


def relabel(_epochs):
    _epochs.events[_epochs.events[:, -1] == 4, -1] = 2
    _epochs.event_id.pop('4')


relabel(epochs_1)
relabel(epochs_2)

epochs_1, epochs_2

# %%


def get_toeplitz(event, events, specials=None):
    # Make toeplitz matrix
    # event: Input event
    # events: All events
    # specials[=None]: Targets and buttons events table

    # Create specials if [specials] is None
    if specials is None:
        specials = dict(
            targets=np.where(events[:, -1] == 1),
            others=np.where(events[:, -1] == 2),
            buttons=np.where(events[:, -1] == 3),
        )

    # Init matrix
    matrix = np.zeros((1000, 600))
    # for j, row in enumerate(matrix):
    #     row[range(j % 100, 600, 100)] = 2

    # if event[-1] == 1:
    #     diag = range(600)
    #     matrix[diag, diag] = 1

    names = dict(targets=1,
                 others=2,
                 buttons=3)

    # Mark 'diags'
    for name in names:
        for special_event in events[specials[name]]:
            # Calculate delay
            d = special_event[0] - event[0]

            # Continue if delay is too large
            if abs(d) > 1000:
                continue

            # Get pair_idxs, [[x1, y1], [x2, y2], ... ]
            pair_idxs = [[e, e-d] for e in range(d, d+600)]
            # Make sure every [x, y] is in toeplitz matrix
            pair_idxs = [e for e in pair_idxs if all([e[0] > -1,
                                                      e[0] < 1000,
                                                      e[1] > -1,
                                                      e[1] < 600])]

            # Continue if no diag to mark
            if len(pair_idxs) == 0:
                continue

            # Regulation the pair_idxs, make it into Int type
            pair_idxs = np.array(pair_idxs, dtype=np.int)

            # Mark diag
            matrix[pair_idxs[:, 0], pair_idxs[:, 1]] = names[name]

    return matrix, specials


idx_1 = np.where(epochs_1.events[:, -1] == 1)
print(idx_1)

specials = None
for idx in [446, 447, 448]:
    matrix, specials = get_toeplitz(epochs_1.events[idx],
                                    epochs_1.events,
                                    specials)
    fig, ax = plt.subplots(1, 1, figsize=(3, 5))
    ax.imshow(matrix)
    ax.set_title(idx)


# %%
