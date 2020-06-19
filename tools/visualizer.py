# %% Importing
# System
import os
import sys

# Computing
import mne
import numpy as np

# Tools
sys.path.append(os.path.dirname(__file__))  # noqa
from figure_tools import Drawer

DRAWER = Drawer()

# %% Class


class Visualizer():
    def __init__(self, drawer=DRAWER, title_prefix='', show=False):
        self.epochs = None
        self.drawer = drawer
        self.title_prefix = title_prefix
        self.show = show

    def _title(self, title):
        prefix = self.title_prefix
        return f'{prefix}-{title}'

    def load_epochs(self, epochs):
        self.epochs = epochs
        print(f'New epochs are loaded: {epochs}')

    def plot_joint(self, event_id, title=None):
        if title is None:
            title = event_id

        evoked = self.epochs[event_id].average()
        self.drawer.fig = evoked.plot_joint(show=self.show,
                                            title=self._title(title))

    def plot_psd(self, event_id, title=None):
        if title is None:
            title = event_id

        epochs = self.epochs[event_id]

        pass

    def save_figs(self, path):
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
