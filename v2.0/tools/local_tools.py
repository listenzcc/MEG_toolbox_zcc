# %%
import os
import mne

# %% ----------------------------------


def find_files(dir, ext='.fif'):
    # Find all files in [dir] with extension of [ext]
    # Return a list of full path of found files.

    def join(name):
        return os.path.join(dir, name)

    # Init files_path
    fullpaths = [join(name) for name in os.listdir(dir)
                 if all([name.endswith(ext),
                         not os.path.isdir(join(name))])]

    n = len(fullpaths)

    # Return
    print(f'Found {n} files that ends with {ext} in {dir}')
    return fullpaths


def mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass


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


def filter_raw(raw, l_freq, h_freq):
    raw.load_data()
    filtered_raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=32)
    print(f'Filtered {raw} using ({l_freq}, {h_freq})')
    return filtered_raw

# %% ----------------------------------
# Set and get local configures through environments


class Configure():
    def __init__(self, prefix='_MEG_RSVP_'):
        self.prefix = prefix

    def unique(self, key):
        return f'{self.prefix}{key}'

    def set(self, key, value):
        os.environ[self.unique(key)] = value

    def get(self, key):
        return os.environ.get(self.unique(key))

    def getall(self):
        outputs = dict()
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                outputs[key[len(self.prefix):]] = value
        return outputs

# %% ------------------------------------
# Load files


class FileLoader():
    def __init__(self, subject, parameters, config=Configure()):
        self.config = config
        raw_dir = self.config.get('RAW_DIR')
        memory_dir = self.config.get('MEMORY_DIR')

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

    def load_epochs(self, recompute=True):
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
                events = mne.find_events(raw,
                                         stim_channel=self.parameters['stim_channel'])
                sfreq = raw.info['sfreq']
                events = relabel(events, sfreq)

                # Create Epochs
                epochs = mne.Epochs(raw,
                                    events=events,
                                    picks=self.parameters['picks'],
                                    tmin=self.parameters['tmin'],
                                    tmax=self.parameters['tmax'],
                                    decim=self.parameters['decim'],
                                    detrend=self.parameters['detrend'],
                                    baseline=None)

                # Record Epochs
                epochs = epochs[self.parameters['events']]
                epochs_list.append(epochs)
                print(f'New epochs is created: {epochs}')

                # Solid Epochs
                path = os.path.join(self.memory_dir, f'Session_{j}-epo.fif')
                epochs.save(path)
                print(f'New epochs is saved: {path}')

            # Load Epochs into [self.epochs_list]
            self.epochs_list = epochs_list
            n = len(self.epochs_list)
            print(f'Created {n} epochs from raws')

        if not recompute:
            self.epochs_list = [mne.read_epochs(path) for path in
                                find_files(self.memory_dir, '-epo.fif')]
            n = len(self.epochs_list)
            print(f'Recalled {n} epochs')
