# %% Importing
# System
import os
import sys

# Computing
import mne
import numpy as np

# Plotting
import matplotlib.pyplot as plt

# Tools
sys.path.append(os.path.dirname(__file__))  # noqa
from figure_tools import Drawer

# %% Class


class Visualizer():
    def __init__(self, title_prefix='', show=False):
        self.epochs = None
        self.drawer = Drawer()
        self.title_prefix = title_prefix
        self.show = show

    def _title(self, title):
        # Built-in title method
        prefix = self.title_prefix
        return f'{prefix}-{title}'

    def load_epochs(self, epochs):
        # Load epochs for ploting
        self.epochs = epochs
        print(f'New epochs are loaded: {epochs}')

    def plot_lags(self, paired_lags_timelines, title=None):
        # Methods for visualizing the lags and timelines of button effect
        # Make sure we have a title
        if title is None:
            title = 'lags(reds) and timelines(background)'

        # Prepare times, _lags, _timelines,
        # _lags: the lags of behavior button response to target picture,
        # _timelines: the estimated time line of the button effect.
        times = self.epochs.times
        _lags = paired_lags_timelines['sorted_lags']
        _timelines = paired_lags_timelines['sorted_timelines']
        num_samples = _lags.shape[0]

        # Plot in two layers,
        # Bottom is the _timelines matrix,
        # Top is the _lags curve.
        fig, ax = plt.subplots(1, 1)
        ax.plot(_lags, c='red', alpha=0.7, linewidth=3)
        ax.set_ylim([min(times), max(times)])
        im = ax.imshow(_timelines.transpose(),
                       extent=(0, num_samples-1, min(times), max(times)),
                       aspect=200,
                       origin='lower')
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

        # Save fig into drawer
        self.drawer.fig = fig

    def plot_joint(self, event_id, title=None):
        # Plot joint
        # Make sure we have a title
        if title is None:
            title = event_id

        # Compute evoked
        evoked = self.epochs[event_id].average()

        # Plot evoked in joint plot
        self.drawer.fig = evoked.plot_joint(show=self.show,
                                            title=self._title(title))

    def plot_psd(self, event_id, title=None):
        # Plot pad [not done yet]
        # Make sure we have a title
        if title is None:
            title = event_id

        # Get epochs
        epochs = self.epochs[event_id]

        pass

    def save_figs(self, path):
        # Save plotted figures into [path]
        self.drawer.save(path)

# %% Methods for visualizing the lags and timelines of button effect

# times = meg_worker.clean_epochs.times
# _lags = meg_worker.paried_lags_timelines['sorted_lags']
# _timelines = meg_worker.paried_lags_timelines['sorted_timelines']

# num_samples = _lags.shape[0]

# fig, ax = plt.subplots(1, 1)
# ax.plot(_lags, c='red', alpha=0.7, linewidth=3)
# ax.set_ylim([min(times), max(times)])
# im = ax.imshow(_timelines.transpose(),
#                extent=(0, num_samples-1, min(times), max(times)),
#                aspect=200,
#                origin='lower')
# ax.set_title('lags(reds) and timelines(background)')
# fig.colorbar(im, ax=ax)
