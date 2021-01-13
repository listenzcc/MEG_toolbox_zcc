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
epochs_inventory = pd.read_json('inventory-epo.json')
epochs_inventory

# %%


def fetch_subject(subject, frame=epochs_inventory):
    return frame.query(f'subject == "{subject}"')


# %%
epochs_list = [mne.read_epochs(e) for e in
               fetch_subject('MEG_S02')['epochsPath'].tolist()]


# %%
epochs = mne.concatenate_epochs(epochs_list)
epochs

# %%
epochs_1 = epochs['1'].copy()
epochs_1.apply_baseline((None, 0))
# epochs_1.filter(l_freq=0.1, h_freq=7.0, n_jobs=48)
epochs_1.average().plot_joint()
# '1' event has 465 epochs
epochs_1

# %%
data = epochs_1.get_data()
times = epochs_1.times
# The shape of data is (465, 272, 141)
data.shape

# %%
# Use the analysis as in "P300 Speller Performance Predictor Based on RSVP Multi-feature"
shape = (465, 272)
# data = np.abs(data)
peak_amplitude = np.zeros(shape)
peak_latency = np.zeros(shape)
for i in tqdm(range(shape[0])):
    for j in range(shape[1]):
        d = data[i, j, :]
        # d[:20] = -np.inf
        # d[-40:] = -np.inf
        peak_amplitude[i, j] = np.max(d)
        peak_latency[i, j] = times[np.argmax(d)]

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].matshow(peak_amplitude)
im = axes[1].matshow(peak_latency)
fig.colorbar(im, ax=axes[1])

# %%
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
info = epochs_1.info


def draw(topo, ax, title='Title', info=info, show=False, fig=fig):
    im = mne.viz.plot_topomap(topo, info, axes=ax, show=show, cmap='inferno')
    ax.set_title(title)
    fig.colorbar(im[0], ax=ax)
    return im


def mean(x):
    return np.mean(x, axis=0)


def std(x):
    return np.std(x, axis=0)


def norm_std(x):
    return np.std(x, axis=0) / np.mean(x, axis=0)


draw(mean(peak_amplitude), ax=axes[0][0], title='Mean-Amplitude')
draw(mean(peak_latency), ax=axes[0][1], title='Mean-Latency')

draw(std(peak_amplitude), ax=axes[1][0], title='Std-Amplitude')
draw(std(peak_latency), ax=axes[1][1], title='Std-Latency')

draw(norm_std(peak_amplitude), ax=axes[2][0], title='NormStd-Amplitude')

fig.tight_layout()
# %%
