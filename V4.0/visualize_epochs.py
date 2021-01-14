# Visualize epochs

# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import mne
import pandas as pd
import configparser

from tqdm.auto import tqdm

# %%


def mean(x):
    return np.mean(x, axis=0)


def std(x):
    return np.std(x, axis=0)


def norm_std(x):
    return np.std(x, axis=0) / np.mean(x, axis=0)


def zscore(x):
    return (x - np.mean(x)) / np.std(x)


def autocorr(a, b, use_zscore=False):
    #
    assert(len(a.shape) == 1)
    assert(len(b.shape) == 1)
    assert(len(a) == len(b))
    if use_zscore:
        coef = np.correlate(zscore(a), zscore(b), mode='same') / len(a)
    else:
        coef = np.correlate(a, b, mode='same') / len(a)
    return coef


# %%
epochs_inventory = pd.read_json('inventory-epo.json')
epochs_inventory

# %%


def fetch_subject(subject, frame=epochs_inventory):
    return frame.query(f'subject == "{subject}"')


epochs_list = [mne.read_epochs(e) for e in
               fetch_subject('MEG_S02')['epochsPath'].tolist()]

epochs = mne.concatenate_epochs(epochs_list)

epochs_1 = epochs['1'].copy()
epochs_1.apply_baseline((None, 0))
# epochs_1.filter(l_freq=0.1, h_freq=7.0, n_jobs=48)
epochs_1.average().plot_joint()
# '1' event has 465 epochs
epochs_1

# %%
data = epochs_1.get_data()
mean_data = epochs_1.average().data
times = epochs_1.times
lags = np.array([e-int(len(times)/2) for e in range(len(times))]) / 100
# The shape of data is (465, 272, 141)
# The shape of mean_data is (272, 141)
data.shape, mean_data.shape

# %%
# Use the analysis as in "P300 Speller Performance Predictor Based on RSVP Multi-feature"

# Compute matrix of peak of amplitude / latency
shape = (465, 272)
peak_amplitude = np.zeros(shape)
peak_latency = np.zeros(shape)
autocorr_lag = np.zeros(shape)
for j in tqdm(range(shape[1])):
    m = zscore(mean_data[j])
    for i in range(shape[0]):
        d = data[i, j, :]
        coef = autocorr(zscore(d), m)
        peak_amplitude[i, j] = np.max(d)
        peak_latency[i, j] = times[np.argmax(d)]
        autocorr_lag[i, j] = lags[np.argmax(coef)]

# Draw as matrix
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

im = axes[0].matshow(peak_amplitude)
fig.colorbar(im, ax=axes[0])
axes[0].set_title('Peak-Amplitude')

im = axes[1].matshow(peak_latency)
fig.colorbar(im, ax=axes[1])
axes[1].set_title('Peak-Latency')

im = axes[2].matshow(autocorr_lag)
fig.colorbar(im, ax=axes[2])
axes[2].set_title('Autocorr-Lag')

fig.tight_layout()

# Draw as topo map
fig, axes = plt.subplots(4, 2, figsize=(12, 12))
info = epochs_1.info


def draw(topo, ax, title='Title', info=info, show=False, fig=fig, cmap='inferno'):
    im = mne.viz.plot_topomap(topo, info, axes=ax, show=show, cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im[0], ax=ax)
    return im


row = 0
draw(mean(peak_amplitude), ax=axes[row][0], title='Mean-Amplitude')
draw(std(peak_amplitude), ax=axes[row][1], title='Std-Amplitude')

row = 2
draw(mean(peak_latency), ax=axes[row][0], title='Mean-Latency')
draw(std(peak_latency), ax=axes[row][1], title='Std-Latency')

row = 1
draw(norm_std(peak_amplitude), ax=axes[row][1], title='NormStd-Amplitude')

row = 3
draw(mean(autocorr_lag), ax=axes[row][0], title='Mean-Lag', cmap='Spectral_r')
draw(std(autocorr_lag), ax=axes[row][1], title='Std-Lag')

fig.tight_layout()

# %%
