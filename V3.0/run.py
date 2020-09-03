# %%
import mne
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import vectorize
from tools import Drawer
from data_manager import get_objects
from sklearn.manifold import TSNE

# %%
raws = get_objects.get_raws('MEG_S02')
raw = get_objects.concatenate_raws(raws)

# %%
# epochs = get_objects.get_epochs(raws[3])
epochs = get_objects.get_epochs(raw)
print(epochs.info)
epochs

# %%
freqs = np.logspace(0.5, 1.5, 20)
freqs = np.array([3, 5, 7, 10, 15, 20, 30])
n_cycles = np.ceil(freqs / 2).astype(np.int)
evoked = epochs['1'].average()

for event in ['1', '2', '3']:
    print(event)
    power, itc = mne.time_frequency.tfr_morlet(epochs[event],
                                               freqs,
                                               n_cycles,
                                               use_fft=False,
                                               return_itc=True,
                                               decim=1,
                                               n_jobs=48,
                                               picks='mag',
                                               zero_mean=True,
                                               average=True,
                                               output='power',
                                               verbose=None)

    drawer = Drawer()
    for j, freq in enumerate(freqs):
        _power = power.data[:, j, :]
        evoked.data = _power
        drawer.fig = evoked.plot_joint(title=freq, show=False)

    drawer.save(f'event_{event}.pdf')

# %%
epochs['1'].average().plot_joint()
epochs['2'].average().plot_joint()
print()

# %%


class MyFilter(object):
    def __init__(self):
        pass

    def fit(self, data):
        # Solve matrix function
        # B: D = A \cdot B
        # D is data matrix, 272 x 101
        #                   272: sensors
        #                   101: time points
        # B is Toeplitz matrix
        # A is spatial filter
        # Dimension of A and B is 10
        # 10 is because sample rate is 100 Hz
        print('Fitting')
        D = data.copy()
        B = np.zeros((10, D.shape[1]))
        for j in range(10):
            idxs = [e for e in range(j, 101, 10)]
            B[j, idxs] = 1
        b = np.linalg.pinv(B)
        A = np.matmul(D, b)

        self.spatial_filter = A
        print(f'Yielded spatial filter of size {A.shape}')

    def transform(self, data, zero_out=False):
        # Solve matrix function
        # B: D = A \cdot B
        # D is data matrix, 272 x ??
        #                   same as self.fit
        # A is spatial_filter in self.fit
        # B is time series matrix
        # Make sure operate this after self.fit
        # Return A \cdot \cap{B} as transformed data
        print(f'Transforming {data.shape}')
        if zero_out:
            print('Zero out mode is on, transformed data will be substituted from data')

        A = self.spatial_filter
        a = np.linalg.pinv(A)

        if len(data.shape) == 2:
            D = data
            B = np.matmul(a, D)
            if zero_out:
                return data - np.matmul(A, B)
            else:
                return np.matmul(A, B)

        if len(data.shape) == 3:
            new_D = []
            for D in data:
                B = np.matmul(a, D)
                new_D.append(np.matmul(A, B))
            if zero_out:
                return data - np.array(new_D)
            else:
                return np.array(new_D)

        raise ValueError(f'Can not handle data with size of {data.shape}')


evoked2 = epochs['2'].average()
evoked2.crop(tmin=0, tmax=1)

evoked1 = epochs['1'].average()
evoked1_no_crop = evoked1.copy()
evoked1.crop(tmin=0, tmax=1)

my_filter = MyFilter()

my_filter.fit(evoked2.data)
new_data1 = my_filter.transform(evoked1.data, zero_out=True)
new_data2 = my_filter.transform(evoked2.data, zero_out=True)
new_epochs_data1 = my_filter.transform(epochs['1'].get_data(), zero_out=True)

# Plot filtered evoked
# Plot joint of 2
evoked2.plot_joint(title='Raw 2')
_evoked2 = evoked2.copy()
_evoked2.data = new_data2
_evoked2.plot_joint(title='Filtered 2')

# Plot joint of 1
evoked1.plot_joint(title='Raw 1')
_evoked1 = evoked1.copy()
_evoked1.data = new_data1
_evoked1.plot_joint(title='Filtered 1')

# Plot joint of 2,
# test  filter on epochs data
evoked1_no_crop.plot_joint(title='Raw 1 nc')
_evoked1_nc = evoked1_no_crop.copy()
_evoked1_nc.data = np.mean(new_epochs_data1, axis=0)
_evoked1_nc.plot_joint(title='Filtered 1 nc')

print('Done')

# %%
_epochs = epochs[['1', '2', '5']]
data = _epochs.get_data()
new_data = my_filter.transform(data, zero_out=True)
label = _epochs.events[:, -1]
new_epochs = mne.BaseEpochs(_epochs.info,
                            new_data,
                            _epochs.events,
                            tmin=_epochs.times[0],
                            tmax=_epochs.times[-1],
                            baseline=None)

'Shapes:', new_data.shape, label.shape

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes = np.ravel(axes)

axes[0].plot(np.mean(new_data[label == 1], axis=0).transpose())
axes[0].set_title('New 1')

axes[1].plot(np.mean(new_data[label == 2], axis=0).transpose())
axes[1].set_title('New 2')

axes[2].plot(np.mean(data[label == 1], axis=0).transpose())
axes[2].set_title('Old 1')

axes[3].plot(np.mean(data[label == 2], axis=0).transpose())
axes[3].set_title('Old 2')

for ax in axes:
    ax.set_ylim([-3e-13, 3e-13])

print()

# %%
# tsne = TSNE(n_components=2, n_jobs=48)
# vectorizer = mne.decoding.Vectorizer()
# d2 = tsne.fit_transform(vectorizer.fit_transform(new_data))

# 'Shapes:', d2.shape

# # Plot samples in 2-dimension space
# plt.style.use('ggplot')
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# for j in np.unique(label):
#     selects = label == j
#     ax.scatter(d2[selects, 0], d2[selects, 1], alpha=0.4, label=j)
# ax.legend()
# print('Done')


# %%
_epochs = epochs[['1', '2', '5']]
_epochs.apply_baseline(None).load_data()
cov = mne.compute_covariance(_epochs)

# %%
xdawn = mne.preprocessing.Xdawn(n_components=6, signal_cov=cov)
xdawn.fit(_epochs)
xdawn_epochs = xdawn.apply(_epochs, event_id=['1'], include=[0])
xdawn_epochs

# %%
_epochs['1'].average().plot_joint(title='Raw 1')
_epochs['2'].average().plot_joint(title='Raw 2')
xdawn_epochs['1']['1'].average().plot_joint(title='New 1')
xdawn_epochs['1']['2'].average().plot_joint(title='New 2')
print()

# %%
help(xdawn.apply)
# %%
