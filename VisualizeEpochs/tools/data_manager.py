# File: data_manager.py
# Aim: Data manager for MEG/EEG objects

import os
import mne
from . import config
from .local_tools import find_files, mkdir


def relabel_events(events):
    # Relabel events
    # Relabeled event:
    #  1: Target
    #  2: Far non-target
    #  3: Button motion
    #  4: Near non-target

    print(f'Relabel {len(events)} events')

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
                if count == 10:
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
                if count == 10:
                    break

    return events


def filter_raw(raw, l_freq, h_freq):
    # Filter the [raw] and return the filtered raw
    raw.load_data()
    filtered_raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=48)
    print(f'Filtered {raw} using ({l_freq}, {h_freq})')
    return filtered_raw


class DataManager(object):
    # Build-in functions:
    #   self.load_raws: Load raw .fif and filter them
    #       output: Yield self.raws, a list of raw objects
    #
    #   self.load_epochs: Call self.load_raws and get epochs in them
    #       output: Yield self.epochs_list, a list of epochs, the epochs of self.raws
    #       !!! Will read from memory if [recompute] is False
    #       !!! If [recompute] is True, then the epochs will be saved in [self.memory_dir]

    def __init__(self, subject, config=config, parameters=None):
        self.config = config
        raw_dir = self.config.get('RAW_DIR')
        memory_dir = self.config.get('MEMORY_DIR')

        print(config.get_all())

        self.subject = subject
        self.raw_dir = os.path.join(raw_dir, self.subject)
        self.memory_dir = os.path.join(memory_dir, self.subject)

        self.parameters = parameters
        pass

    def load_raws(self, bandpass=None):
        # Enumerate dir
        names = os.listdir(self.raw_dir)

        # Get raw names
        raw_names = []
        for j in range(20):
            raw_name = f'block_{j:02d}'
            # Try _ica-raw.fif file firstly,
            # if not exist, try _raw.fif files,
            # if still not exist, do nothing and continue
            if f'{raw_name}_ica-raw.fif' in names:
                raw_names.append(f'{raw_name}_ica-raw.fif')
                continue

            if f'{raw_name}_raw.fif' in names:
                raw_names.append(f'{raw_name}_raw.fif')
                continue

        # Read raws
        raws = []
        for name in raw_names:
            print(f'Reading {self.raw_dir}, {name}')
            path = os.path.join(self.raw_dir, name)
            raws.append(mne.io.read_raw_fif(path, verbose=False))
        print(f'Found {len(raws)} raws in {self.subject}.')

        # Filter if [bandpass] is not None
        if bandpass is not None:
            l_freq, h_freq = bandpass
            raws = [filter_raw(r, l_freq, h_freq) for r in raws]

        # Load raws
        self.raws = raws

    def load_epochs(self, recompute=False):
        # Load epochs by new creation or recall memory.
        # if [recompute], read raws and create new Epochs,
        # other wise, recall epochs from memory.
        if recompute:
            # New creation mode
            # Make sure memory_dir exists
            mkdir(self.memory_dir)

            # Read raws
            self.load_raws(bandpass=(self.parameters['l_freq'],
                                     self.parameters['h_freq']))

            # Create new Epochs
            epochs_list = []
            for j, raw in enumerate(self.raws):
                # Get events
                if self.parameters['stim_channel'] == 'from_annotations':
                    events = mne.events_from_annotations(raw)[0]
                else:
                    events = mne.find_events(raw,
                                             stim_channel=self.parameters['stim_channel'])

                events = relabel_events(events)

                # Create Epochs
                epochs = mne.Epochs(raw,
                                    events=events,
                                    picks=self.parameters['picks'],
                                    tmin=self.parameters['tmin'],
                                    tmax=self.parameters['tmax'],
                                    decim=self.parameters['decim'],
                                    detrend=self.parameters['detrend'],
                                    reject=self.parameters['reject'],
                                    baseline=self.parameters['baseline'])

                # Record Epochs
                # epochs = epochs[self.parameters['events']]
                epochs_list.append(epochs)
                print(f'New epochs is created: {epochs}')

                # Solid Epochs
                path = os.path.join(self.memory_dir, f'Session_{j}-epo.fif')
                epochs.save(path, overwrite=True)
                print(f'New epochs is saved: {path}')

            # Load Epochs into [self.epochs_list]
            self.epochs_list = epochs_list
            n = len(self.epochs_list)
            print(f'Created {n} epochs from raws')

        # Recall the epochs
        if not recompute:
            self.epochs_list = [mne.read_epochs(path) for path in
                                find_files(self.memory_dir, '-epo.fif')]
            n = len(self.epochs_list)
            print(f'Recalled {n} epochs')

        # Reject the epochs that is too short
        self.epochs_list = [e for e in self.epochs_list if len(e.events) > 300]

        # def reject_bads(threshold=1e-12):
        #     for epochs in self.epochs_list:
        #         d = epochs.get_data()
        #         drops = []
        #         for j, e in enumerate(epochs.events):
        #             _max = np.max(d[j])
        #             # _min = np.min(d[j])
        #             if _max > threshold:
        #                 drops.append(j)

        #         epochs.drop(drops, reason='Reject big values.')

        # reject_bads()

    def leave_one_session_out(self, includes, excludes):
        # Perform leave one session out on [self.epochs_list]

        def align_epochs():
            # Inner method for align epochs [self.epochs_list]
            dev_head_t = self.epochs_list[0].info['dev_head_t']
            for epochs in self.epochs_list:
                epochs.info['dev_head_t'] = dev_head_t
            pass

        # Align epochs
        align_epochs()

        # Separate [includes] and [excludes] epochs
        include_epochs = mne.concatenate_epochs([self.epochs_list[j]
                                                 for j in includes])
        if len(excludes) == 0:
            exclude_epochs = None
        else:
            exclude_epochs = mne.concatenate_epochs([self.epochs_list[j]
                                                     for j in excludes])

        return include_epochs, exclude_epochs
