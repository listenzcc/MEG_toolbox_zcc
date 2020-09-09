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
            title = 'Lags vs. Timeline'

        # Prepare times, _lags, _timelines,
        # _lags: the lags of behavior button response to target picture,
        # _timelines: the estimated time line of the button effect.
        times = self.epochs.times
        _lags = paired_lags_timelines['sorted_lags']
        _timelines = paired_lags_timelines['sorted_timelines']

        # Filter too long lags
        _timelines = _timelines[_lags < 0.8]
        _lags = _lags[_lags < 0.8]

        # Compute number of samples
        num_samples = _lags.shape[0]

        # Compute corr between _lags and peak _timelines
        def _corr(_lags, _timelines, num_samples):

            def _smooth_max(a, b=np.ones(10)):
                new_a = np.convolve(a, b, mode='same')
                return np.where(new_a == max(new_a))[0][0]

            _idx_peaks = np.array([_smooth_max(e)
                                   for e in _timelines])

            return np.corrcoef(_lags, _idx_peaks)[0][1]

        _corrcoef = _corr(_lags, _timelines, num_samples)
        title = f'{title} - {_corrcoef}'

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

    def plot_joint(self, event_id, title=None, times='peaks'):
        # Plot joint
        # Make sure we have a title
        if title is None:
            title = event_id

        # Compute evoked
        evoked = self.epochs[event_id].average()

        # Plot evoked in joint plot
        self.drawer.fig = evoked.plot_joint(show=self.show,
                                            times=times,
                                            title=self._title(title))

    def save_figs(self, path):
        # Save plotted figures into [path]
        self.drawer.save(path)
        self.drawer.clear_figures()

# %%
