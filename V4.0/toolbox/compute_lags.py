# Compute and plot lags of signals in epochs

import mne
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .figure_utils import FigureCollection

# Local utils


def mean(x):
    return np.mean(x, axis=0)


def std(x):
    return np.std(x, axis=0)


def norm_std(x):
    return np.std(x, axis=0) / np.mean(x, axis=0)


def zscore(x):
    return (x - np.mean(x)) / np.std(x)


def autocorr(a, b, use_zscore=False):
    assert(len(a.shape) == 1)
    assert(len(b.shape) == 1)
    assert(len(a) == len(b))
    if use_zscore:
        coef = np.correlate(zscore(a), zscore(b), mode='same') / len(a)
    else:
        coef = np.correlate(a, b, mode='same') / len(a)
    return coef


# Main Lags Computer
class LagsComputer(object):
    '''
    # Analysis the lags of signal between epochs
    Use the analysis as in "P300 Speller Performance Predictor Based on RSVP Multi-feature".
    Mainly compute the peak shift in time axis of RSVP target epochs.
    Compute three values:
    - peak_amplitude: The value of peak signal in each epochs
    - peak_latency: The time shift of the peak_amplitude occurs
    - autocorr_lag: The time shift between each epochs and averated evoked signals
    '''

    def __init__(self, epochs):
        '''
        Init the object using epochs
        - @epochs: The epochs of target stimuli
        '''
        self.epochs = epochs
        print(f'Initialized with {epochs}')

    def get_stuff(self):
        '''
        Get useful stuff of the epochs
        - data: Data in shape of (num_trials, num_channels, num_times)
        - info: The info of the epochs
        - mean_data: Data averaged in axis 0
        - times: Times of epochs
        - lags: Lags axis used for auto-correlation lag computing, it is **ZERO** in center, **NEGATIVE** on left and **POSITIVE** on right.
        - shape: The shape of mean_data, (num_channels, num_times)
        '''
        # Get stuff
        data = self.epochs.get_data()
        info = self.epochs.info
        mean_data = self.epochs.average().data
        times = self.epochs.times
        lags = np.array([e-int(len(times)/2) for e in range(len(times))]) / 100
        shape = data.shape[:2]

        # Collect results
        self.data = data
        self.info = info
        self.mean_data = mean_data
        self.times = times
        self.lags = lags
        self.shape = shape

    def compute_lags(self):
        '''
        Compute lag values
        - peak_amplitude: The value of peak signal in each epochs
        - peak_latency: The time shift of the peak_amplitude occurs
        - autocorr_lag: The time shift between each epochs and averated evoked signals
        '''
        # Get stuff firstly
        self.get_stuff()

        # Perform calculation
        peak_amplitude = np.zeros(self.shape)
        peak_latency = np.zeros(self.shape)
        autocorr_lag = np.zeros(self.shape)
        for j in tqdm(range(self.shape[1])):
            m = zscore(self.mean_data[j])
            for i in range(self.shape[0]):
                d = self.data[i, j, :]
                coef = autocorr(zscore(d), m)
                peak_amplitude[i, j] = np.max(d)
                peak_latency[i, j] = self.times[np.argmax(d)]
                autocorr_lag[i, j] = self.lags[np.argmax(coef)]

        # Collect results
        self.peak_amplitude = peak_amplitude
        self.peak_latency = peak_latency
        self.autocorr_lag = autocorr_lag

    def draw(self, suptitle='Sup-Title', fc=FigureCollection(), topo_kwargs=dict()):
        '''
        Draw lags values
        - @suptitle: The suptitle of each fig ('Sup-Title' is used by default)
        - @fc: FigureCollection Object (It is Empty collection by default)
        - @topo_kwargs: The kwargs for topological map
        '''

        # --------------------------------------------------------------------------------
        # Draw as matrix
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))

        im = axes[0].matshow(self.peak_amplitude)
        fig.colorbar(im, ax=axes[0])
        axes[0].set_title('Peak-Amplitude')

        im = axes[1].matshow(self.peak_latency)
        fig.colorbar(im, ax=axes[1])
        axes[1].set_title('Peak-Latency')

        im = axes[2].matshow(self.autocorr_lag)
        fig.colorbar(im, ax=axes[2])
        axes[2].set_title('Autocorr-Lag')

        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0.1, 0.1, 0.9, 0.99])
        fc.fig = fig

        # --------------------------------------------------------------------------------
        # Draw mean and std as topological map
        fig, axes = plt.subplots(4, 2, figsize=(12, 12))

        def draw(topo, ax, title='Title', info=self.info, show=False, fig=fig, cmap='inferno', kwargs=topo_kwargs):
            im = mne.viz.plot_topomap(
                topo, info, axes=ax, show=show, cmap=cmap, **kwargs)
            ax.set_title(title)
            fig.colorbar(im[0], ax=ax)
            return im

        row = 0
        draw(mean(self.peak_amplitude),
             ax=axes[row][0], title='Mean-Amplitude')
        draw(std(self.peak_amplitude), ax=axes[row][1], title='Std-Amplitude')

        row = 2
        draw(mean(self.peak_latency), ax=axes[row][0], title='Mean-Latency')
        draw(std(self.peak_latency), ax=axes[row][1], title='Std-Latency')

        row = 1
        draw(norm_std(self.peak_amplitude),
             ax=axes[row][1], title='NormStd-Amplitude')

        row = 3
        draw(mean(self.autocorr_lag), ax=axes[row][0],
             title='Mean-Lag', cmap='Spectral_r')
        draw(std(self.autocorr_lag), ax=axes[row][1], title='Std-Lag')

        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0.1, 0.1, 0.9, 0.99])
        fc.fig = fig

        return fc
