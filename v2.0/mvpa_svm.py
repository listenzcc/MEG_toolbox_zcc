# File: mvpa_svm.py
# Aim: Calculate MVPA baseline using SVM

# %%
import seaborn as sns
from sklearn.manifold import TSNE
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

# %%
import plotly
import plotly.graph_objs as go

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

        stophere

        # Relabel 4 to 2, to generate 2-classes situation
        # train_label[train_label == 4] = 2
        # test_label[test_label == 4] = 2

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
    frame.to_json(f'svm_3classes/{name}.json')
    print(f'{name} MVPA is done')
    # break

print('All done.')

# %%

for name in ['MEG_S02', 'MEG_S03', 'MEG_S04', 'MEG_S05']:
    print('-' * 80)
    print(name)

    try:
        frame = pd.read_json(f'{name}.json')
    except:
        continue

    y_true = np.concatenate(frame.y_true.to_list())
    y_pred = np.concatenate(frame.y_pred.to_list())
    print('Classification report\n',
          metrics.classification_report(y_pred=y_pred, y_true=y_true))
    print('Confusion matrix\n',
          metrics.confusion_matrix(y_pred=y_pred, y_true=y_true))


# %%
train_data.shape, train_label.shape, np.unique(train_label)

# %%
d1 = train_data[train_label == 1]
d2 = train_data[train_label == 2]
d4 = train_data[train_label == 4]
np.random.shuffle(d1)
np.random.shuffle(d2)
np.random.shuffle(d4)

data = np.concatenate([d1[:100], d2[:100], d4[:100]], axis=0)
label = np.zeros((300))
label[:100] = 1
label[100:200] = 2
label[200:] = 4

data.shape, label.shape

# %%
# Plot tsne manifold
plt.style.use('ggplot')

fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=300)
axes = np.ravel(axes)

for j in range(6):
    print(j)
    tsne = TSNE(n_components=2)
    x = np.squeeze(data[:, j])
    x_tsne = tsne.fit_transform(x)

    df = pd.DataFrame(x_tsne)
    t = ['', 'a', 'b', '', 'c']
    df['label'] = [t[int(e)] for e in label]

    ax = axes[j]
    sns.scatterplot(data=df, x=df[0], y=df[1],
                    hue=df.label.to_list(), legend=False, ax=ax, alpha=0.6)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(j)

fig.tight_layout()
fig.savefig('components_distribution.png')

# %%
# Plot waveforms

# Generate data
# Setup label
label_names = dict(
    Target=1,
    Far=2,
    Near=4
)

# Load data into dataframe [df]
times = train_epochs.times
dfs = []
for j, name in enumerate(label_names):
    # Compute averaged waveform of each classes,
    # restore it into [df],
    # besides with label and times
    wave = np.mean(data[label == label_names[name]], axis=0).transpose()
    df = pd.DataFrame(wave)
    df['times'] = times
    df['label'] = name
    # Append the [df] into [dfs] list
    dfs.append(df)

# Concatenate [dfs] into the big dataframe [df]
df = pd.concat(dfs, axis=0)

# Plot
# Setup style
plt.style.use('ggplot')

# Init fig and axes
fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=300)
axes = np.ravel(axes)

# Draw
for j in range(6):
    print(j)
    ax = axes[j]
    sns.lineplot(data=df, x='times', y=j, hue='label',
                 ax=ax, legend=False)
    ax.set_title(j)
    ax.set_xlabel('')
    ax.set_ylabel('')

fig.tight_layout()
fig.savefig('components_averaged.png')

# %%
