# File: mvpa_svm.py
# Aim: Calculate MVPA baseline using SVM

# %%
import mne
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

        clf = make_pipeline(mne.decoding.Vectorizer(),
                            StandardScaler(),
                            PCA(n_components=.95),
                            svm.SVC(**svm_params))

        return dict(includes=epochs_train,
                    excludes=epochs_test,
                    xdawn=xdawn,
                    clf=clf)


class MEGSensorSelection(object):
    # Down sample the MEG sensors,
    # from [raw_num] to [new_num]
    def __init__(self):
        pass

    def fit(self, raw_num=272, new_num=64):
        self.raw_num = raw_num
        self.new_num = new_num
        permutation = np.random.permutation(range(self.raw_num)).astype(int)
        self.drops = permutation[self.new_num:]

    def transform(self, epochs):
        ch_names = epochs.ch_names
        epochs.drop_channels([ch_names[j] for j in self.drops])
        return epochs

# %%


bands = dict(raw=(0.1, 13),
             delta=(0.1, 3),
             theta=(3.5, 7.5),
             alpha=(7.5, 13))

n_jobs = 48

# %%
modal = 'MEG'
for subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
    name = f'{modal}_{subject}'
    # Load EEG data
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
        # Random select 64 sensors in MEG data
        mss = MEGSensorSelection()
        if modal == 'MEG':
            mss.fit()

        # Recursive
        # Get current split
        split = cv.next_split(n_components)
        include_epochs = split['includes']
        exclude_epochs = split['excludes']

        # Random select 64 sensors in MEG data
        include_epochs = mss.transform(include_epochs)
        exclude_epochs = mss.transform(exclude_epochs)
        # stophere

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

        # Transfrom using xdawn
        train_data = xdawn.transform(train_epochs)[:, :n_components]
        test_data = xdawn.transform(test_epochs)[:, :n_components]

        # Get labels and select events
        train_label = train_epochs.events[:, -1]
        test_label = test_epochs.events[:, -1]

        # stophere

        # Relabel 4 to 2, to generate 2-classes situation
        train_label[train_label == 4] = 2
        test_label[test_label == 4] = 2

        # Just print something to show data have been prepared
        print(train_data.shape, train_label.shape,
              test_data.shape, test_label.shape)

        times = train_epochs.times
        tmin, tmax = 0, 1

        selects = [j for j, e in enumerate(times)
                   if all([e < tmax,
                           e > tmin])]
        _train_data = train_data[:, :, selects]
        _test_data = test_data[:, :, selects]

        # SVM MVPA ------------------------------------------
        # Fit svm
        clf.fit(_train_data, train_label)
        print('Clf training is done.')

        # Predict using svm
        label = clf.predict(_test_data)

        # Restore labels
        labels[-1]['y_true'] = test_label
        labels[-1]['y_pred'] = label

        # Print something to show MVPA in this folder is done
        print(f'---- {name} ---------------------------')
        print(metrics.classification_report(y_true=labels[-1]['y_true'],
                                            y_pred=labels[-1]['y_pred'],))

    # Save labels of current [name]
    frame = pd.DataFrame(labels)
    folder_name = 'svm_2classes_meg64'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    frame.to_json(os.path.join(folder_name, f'{name}.json'))
    print(f'{name} MVPA is done')
    # break

print('All done.')

# %%
