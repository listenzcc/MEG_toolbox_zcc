# MEG object class
# %% Importing
# System
import os
import sys
import pickle

# Computing
import mne
import sklearn
import numpy as np

# Private settings
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'private'))  # noqa
from dir_settings import RAW_DIR, MEMORY_DIR
from parameter_settings import PARAMETERS

# Local tools
sys.path.append(os.path.dirname(__file__))  # noqa
from file_tools import read_raw_fif, find_files
from remove_button_effect import Button_Effect_Remover

# %% Local settings


def prompt(msg):
    """Prompting method for [msg]

    Args:
        msg ({obj}): Message to be print
    """
    print(f'>> {msg}')


def relabel(events, sfreq):
    """Re-label 2-> 4 when 2 is near to 1

    Arguments:
        events {array} -- The events array, [[idx], 0, [label]],
                          assume the [idx] column has been sorted.
        sfreq {float} -- The sample frequency

    Returns:
        {array} -- The re-labeled events array
    """
    # Init the pointer [j]
    j = 0
    # Repeat for every '1' event, remember as [a]
    for a in events[events[:, -1] == 1]:
        # Do until...
        while True:
            # Break,
            # if [j] is far enough from latest '2' event,
            # it should jump to next [a]
            if events[j, 0] > a[0] + sfreq:
                break
            # Switch '2' into '4' event if it is near enough to the [a] event
            if all([events[j, -1] == 2,
                    abs(events[j, 0] - a[0]) < sfreq]):
                events[j, -1] = 4
            # Add [j]
            j += 1
            # If [j] is out of range of events,
            # break out the 'while True' loop.
            if j == events.shape[0]:
                break
    # Return re-labeled [events]
    return events

# %% Class


class MEG_Worker():
    def __init__(self, running_name,
                 parameters=PARAMETERS,
                 use_memory=True):
        self.running_name = running_name
        self.parameters = parameters
        self.use_memory = use_memory

    def _get_raw(self, dir, ext):
        # Get concatenated raws from [dir]
        # Get good files path
        files_path = find_files(dir, ext=ext)

        # Get concatenated raw from files_path
        raw = mne.concatenate_raws([read_raw_fif(path)
                                    for path in files_path])

        # Report and return
        prompt(f'Got {raw} from {files_path}')
        return raw

    def _filter_raw(self, l_freq, h_freq):
        # Filter self.raw by [l_freq, h_freq],
        # IN-PLACE
        self.raw.load_data()
        self.raw.filter(l_freq=l_freq, h_freq=h_freq)

        # Report
        prompt(f'Filtered {self.raw} using ({l_freq}, {h_freq})')

    def _get_epochs(self, stim_channel):
        # Get epochs from [raw] as [stim_channel]
        # Read and relabel events
        events = mne.find_events(self.raw,
                                 stim_channel=stim_channel)
        sfreq = self.raw.info['sfreq']
        events = relabel(events, sfreq)

        # Get Epochs
        epochs = mne.Epochs(self.raw,
                            events=events,
                            picks=self.parameters['picks'],
                            tmin=self.parameters['tmin'],
                            tmax=self.parameters['tmax'],
                            decim=self.parameters['decim'],
                            detrend=self.parameters['detrend'],
                            baseline=None)

        epochs = epochs[self.parameters['events']]

        # Report and return
        prompt(f'Got {epochs}')
        return epochs

    def _denoise(self, epochs, use_xdawn=True):
        # Denoise [epochs] using Xdawn
        # Prepare epochs
        epochs.load_data()
        epochs.apply_baseline(self.parameters['baseline'])

        # Init and fit xdawn
        if hasattr(self, 'xdawn'):
            xdawn = self.xdawn
        else:
            xdawn = mne.preprocessing.Xdawn(
                n_components=self.parameters['n_components'])
            xdawn.fit(self.epochs)
            self.xdawn = xdawn

        # Apply Xdawn and return
        return xdawn.apply(epochs)[self.parameters['event']]

    def _remove_button_effect(self, e1, e3):
        """Remove button effect from target epochs

        Args:
            e1 ({str}): Name of event 1
            e3 ({str}): Name of event 3

        Returns:
            Clean epochs with button effect removed,
            Paired lags of samples and estimated button effect timeline.
        """
        remover = Button_Effect_Remover(self.denoise_epochs,
                                        sfreq=self.raw.info['sfreq'])
        clean_epochs, paired_lags_timelines = remover.zero_out_button(e1=e1,
                                                                      e3=e3)

        prompt(f'Removed button effect from target epochs')
        return clean_epochs, paired_lags_timelines

    def pipeline(self, band_name,
                 ext='_ica-raw.fif',
                 stim_channel='UPPT001'):
        """Pipeline of standard operations

        Args:
            band_name ({str}): Band name of filter raw
            ext ({str}, optional): The extend name of interest files. Defaults to '_ica-raw.fif'.
            stim_channel ({str}, optional): The channel name of stimuli. Defaults to 'UPPT001'.
        """
        # Prepare memory stuffs ------------------------------------------------------
        # Raw name
        memory_name = f'{self.running_name}-{band_name}-epo.fif'
        memory_path = os.path.join(MEMORY_DIR, memory_name)
        # Denoise name
        memory_denoise_name = f'{self.running_name}-{band_name}-denoise-epo.fif'
        memory_denoise_path = os.path.join(MEMORY_DIR, memory_denoise_name)
        # Clean name
        memory_clean_name = [f'{self.running_name}-{band_name}-clean-epo.fif',
                             f'{self.running_name}-{band_name}-clean-lags.pkl']
        memory_clean_path = [os.path.join(MEMORY_DIR, memory_clean_name[0]),
                             os.path.join(MEMORY_DIR, memory_clean_name[1])]

        # Get raw -------------------------------------------------------------------
        raw_dir = os.path.join(RAW_DIR, self.running_name)
        self.raw = self._get_raw(raw_dir, ext)

        # Raw epochs ----------------------------------------------------------------
        try:
            assert(self.use_memory)

            # Recall epochs from memory
            self.epochs = mne.read_epochs(memory_path)
            prompt(f'Raw epochs are recalled from memory: {self.epochs}')
        except:
            # Filter raw
            l_freq, h_freq = self.parameters['bands'][band_name]
            self._filter_raw(l_freq=l_freq, h_freq=h_freq)

            # Get epochs
            self.epochs = self._get_epochs(stim_channel)

            # Remember if [use_memory]
            if self.use_memory:
                self.epochs.save(memory_path)

        # Denoise epochs ------------------------------------------------------------
        try:
            assert(self.use_memory)

            # Recall denoise epochs from memory
            self.denoise_epochs = mne.read_epochs(memory_denoise_path)
            prompt(
                f'Denoise epochs are recalled from memory: {self.denoise_epochs}')
        except:
            # Denoise epoch
            self.denoise_epochs = self._denoise(self.epochs.copy())

            # Remember if [use_memory]
            if self.use_memory:
                self.denoise_epochs.save(memory_denoise_path)

        # Remove button effect ------------------------------------------------------
        try:
            assert(self.use_memory)

            # Recall clean epochs and lags from memory
            self.clean_epochs = mne.read_epochs(memory_clean_path[0])
            with open(memory_clean_path[1], 'rb') as f:
                self.paired_lags_timelines = pickle.load(f)
            prompt(
                f'Clean epochs are recalled from memory: {self.clean_epochs}')
        except:
            # Remove button effect
            clean_epochs, paired_lags_timelines = self._remove_button_effect(e1='1',
                                                                             e3='3')
            self.clean_epochs = clean_epochs
            self.paired_lags_timelines = paired_lags_timelines

            # Remember if [use_memory]
            if self.use_memory:
                self.clean_epochs.save(memory_clean_path[0])
                with open(memory_clean_path[1], 'wb') as f:
                    pickle.dump(self.paired_lags_timelines, f)


# %%
# running_name = 'MEG_S03'
# band_name = 'U07'
# worker = MEG_Worker(running_name=running_name)
# worker.pipeline(band_name=band_name)

# %%
