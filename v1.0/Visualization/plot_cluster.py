# %% Importing
# System
import os
import sys

# Computing
import mne
import numpy as np
from sklearn.manifold import TSNE

# Plotting
import matplotlib.pyplot as plt

# Local tools
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))  # noqa
from MEG_worker import MEG_Worker
from figure_tools import Drawer

# Settings
BAND_NAME = 'U07'

INFOS = dict(
    a=dict(crop=(0.2, 0.4),
           center=0.3),
    b=dict(crop=(0.4, 0.6),
           center=0.5),
    c=dict(crop=(0.6, 0.8),
           center=0.7),
)

COLORS_272 = np.load(os.path.join(os.path.dirname(__file__),
                                  '..', 'tools', 'colors_272.npy'))

DRAWER = Drawer()

SHOW = False

# %%

for idx in range(1, 11):
    # Setting -------------------------------------------
    running_name = f'MEG_S{idx:02d}'

    plt.style.use('ggplot')

    # Worker pipeline -----------------------------------
    worker = MEG_Worker(running_name=running_name)
    worker.pipeline(band_name=BAND_NAME)

    # Compute segments ----------------------------------
    # Init empty segments and segment_data
    segments = dict()
    segment_data = dict()

    # Plot evoked
    DRAWER.fig = worker.clean_epochs.average().plot_joint(
        times=[e['center'] for e in INFOS.values()],
        title=running_name,
        show=SHOW)

    # Set values
    for key in INFOS:
        crop = INFOS[key]['crop']
        segments[key] = worker.clean_epochs.copy().crop(crop[0], crop[1])
        segment_data[key] = np.mean(segments[key].get_data(), axis=-1)

    # t-SNE analysis ------------------------------------
    # Concatenate data and colors
    all_data = np.concatenate([segment_data[e] for e in INFOS], axis=1)
    all_colors = np.concatenate([COLORS_272 for e in INFOS], axis=0)

    print(all_data.shape, all_colors.shape)

    # t-SNE analysis on concatenated data space
    tsne = TSNE(n_components=2)
    proj_all_data = tsne.fit_transform(all_data.transpose())

    # Plot -----------------------------------------------
    # Init figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = np.ravel(axes)

    # Joint plotting
    ax = axes[0]
    ax.scatter(proj_all_data[:, 0], proj_all_data[:, 1], c=all_colors)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Separate plotting
    for j, key in enumerate(INFOS):
        start, stop = 272 * j, 272 * (j+1)
        ax = axes[j+1]
        ax.scatter(proj_all_data[start: stop, 0],
                   proj_all_data[start: stop, 1], c=COLORS_272)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        title = '{} - {}'.format(key, INFOS[key]['center'])
        ax.set_title(title)

    # Set super title
    fig.suptitle(running_name)

    # Save fig
    DRAWER.fig = fig

# %% Print into pdf
DRAWER.save('cluster.pdf')

# %%
