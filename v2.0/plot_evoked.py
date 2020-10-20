# File: plot_evoked.py
# Aim: Plot epochs in evoked manner

# %%
import mne
import numpy as np
import matplotlib.pyplot as plt
from tools.data_manager import DataManager

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

        clf = None

        return dict(includes=epochs_train,
                    excludes=epochs_test,
                    xdawn=xdawn,
                    clf=clf)


# %%
# Parameters
n_components = 6
name = 'MEG_S02'

# Load data
dm = DataManager(name)
dm.load_epochs(recompute=False)

# Init cross validation
cv = CV_split(dm)
cv.reset()


# %%

# Get current split
split = cv.next_split(n_components)
include_epochs = split['includes']
exclude_epochs = split['excludes']

# Get scaler, xdawn and clf
xdawn = split['xdawn']

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
xdawn.fit(train_epochs)

# Apply baseline
# !!! Make sure applying baseline **AFTER** xdawn fitting
train_epochs.apply_baseline((None, 0))
test_epochs.apply_baseline((None, 0))

# Apply xdawn
applied = xdawn.apply(train_epochs)
# train_data = xdawn.transform(train_epochs)[:, :n_components]
# test_data = xdawn.transform(test_epochs)[:, :n_components]


# %%
bands = dict(delta=(0.1, 3),
             theta=(3.5, 7.5),
             alpha=(7.5, 13))
n_jobs = 48

# %%


def plot_waveform(epochs, prefix, bands=bands):
    # Filter and plot
    epochs_filtered = dict(full=epochs.copy())

    for band in bands:
        print(prefix, band)
        l_freq, h_freq = bands[band]
        epochs_filtered[band] = epochs.copy().filter(
            l_freq, h_freq, n_jobs=n_jobs)
        fig = epochs_filtered[band].average().plot_joint(
            title=band.title(), ts_args=dict(ylim=[-150, 150]))
        # We need 6 inches-width figure with 300 dpi
        fig.set_dpi(300)
        fig.set_figwidth(8)
        # Set ylim
        axes = fig.get_children()
        # axes[1].get_ylim()
        # axes[1].set_ylim([-200, 180])
        # Save fig
        # fig.tight_layout()
        fig.savefig(f'{prefix}_{band}.png')


# Epochs is the epochs to be plotted
for epochs, prefix in zip([train_epochs['1'].copy(),
                           train_epochs['2'].copy(),
                           train_epochs['4'].copy()],
                          ['raw_target',
                           'raw_nontarget',
                           'raw_nearnontarget']):

    plot_waveform(epochs, prefix)

print('Done.')
# %%
