# %%
# System
import os

# Computing
import mne
import numpy as np

# Torch
import torch
import torch.nn as nn
DEVICE = 'cuda'

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
    def __init__(self, subject, parameters=None, config=Configure()):
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

            concatenate_raws = mne.concatenate_raws(self.raws)

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
                                    reject=self.parameters['reject'],
                                    baseline=None)

                # Record Epochs
                epochs = epochs[self.parameters['events']]
                epochs_list.append(epochs)
                print(f'New epochs is created: {epochs}')

                # Solid Epochs
                path = os.path.join(self.memory_dir, f'Session_{j}-epo.fif')
                # epochs.save(path)
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

        def reject_bads(threshold=1e-12):
            for epochs in self.epochs_list:
                d = epochs.get_data()
                drops = []
                for j, e in enumerate(epochs.events):
                    _max = np.max(d[j])
                    # _min = np.min(d[j])
                    if _max > threshold:
                        drops.append(j)

                epochs.drop(drops, reason='Reject big values.')

        reject_bads()

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
        exclude_epochs = mne.concatenate_epochs([self.epochs_list[j]
                                                 for j in excludes])

        return include_epochs, exclude_epochs

# %%


class Enhancer():
    def __init__(self, train_epochs, test_epochs,
                 n_components=6):
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.xdawn = mne.preprocessing.Xdawn(n_components=n_components)

    def fit(self):
        self.xdawn.fit(self.train_epochs)

    def apply(self, target_event_id):
        enhanced_train_epochs = self.xdawn.apply(self.train_epochs)
        enhanced_test_epochs = self.xdawn.apply(self.test_epochs)

        return (enhanced_train_epochs[target_event_id],
                enhanced_test_epochs[target_event_id])

    def fit_apply(self, target_event_id='1'):
        self.fit()
        return self.apply(target_event_id)


# %%
def numpy2torch(array, dtype=np.float32, device=DEVICE):
    # Attach [array] to the type of torch
    return torch.from_numpy(array.astype(dtype)).to(device)


def torch2numpy(tensor):
    # Detach [tensor] to the type of numpy
    return tensor.detach().cpu().numpy()


class CNN_Model(nn.Module):
    # INPUT_SIZE = (1, 141)
    # OUTPUT_SIZE = (272, 141)
    # ----------------------------------------------------------------
    #         Layer (type)               Output Shape         Param #
    # ================================================================
    #             Conv1d-1             [-1, 272, 141]           3,264
    # ================================================================

    def __init__(self, kernel_weights, num_channels=272):
        super(CNN_Model, self).__init__()
        self.kernel_weights = numpy2torch(kernel_weights)
        self.num_channels = num_channels
        self.init_layers()
        self.set_parameters()

    def init_layers(self):
        # Init the conv layer, named as L1
        self.L1 = nn.Conv1d(in_channels=1,
                            out_channels=self.num_channels,
                            kernel_size=11,
                            padding=5)

    def set_parameters(self):
        # Setup the parameter of L1
        # Parameter of weights
        _shape = self.L1.weight.shape
        self.L1.weight = nn.parameter.Parameter(self.kernel_weights)
        self.L1.weight.requires_grad = False

        # Parameter of bias
        _shape = self.L1.bias.shape
        self.L1.bias = nn.parameter.Parameter(torch.zeros(_shape).to(DEVICE))
        self.L1.bias.requires_grad = True

    def forward(self, x):
        # Forward flow
        y = self.L1(x)
        return y

    def fit(self, x, y_true, learning_rate, steps=40):
        # Reset parameters
        self.set_parameters()

        # Convert [x] and [y_true] into torch
        x = numpy2torch(x)
        y_true = numpy2torch(y_true)

        # Set [x] to requires_grad
        x.requires_grad = True

        # Set training issues
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam([x], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=20,
                                                    gamma=0.5)

        loss0 = None
        for step in range(steps):
            # Forward
            y = self.forward(x)

            # Compute loss
            loss = criterion(y, y_true)
            if loss0 is None:
                loss0 = loss.item()

            # Back Pursuit
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        loss1 = loss.item()
        print(f'        Loss: {loss0:0.2e} -> {loss1:0.2e}, {loss0 / loss1}')

        # Return optimized [x] and its [y]
        return torch2numpy(x), torch2numpy(y), loss0 / loss1


class Denoiser():
    def __init__(self):
        pass

    def fit(self, noise_epochs):
        self.noise_epochs = noise_epochs

    def transform(self, epochs, labels=[1, 2], allowed_r_min=0):
        # Get data from [self.epochs]
        data = epochs.get_data()
        events = epochs.events
        num_samples, num_channels, num_times = data.shape

        # Get 11 time points around 0 seconds in noise_epochs
        noise_evoked = self.noise_epochs.average()
        idx = np.where(noise_evoked.times == 0)[0][0]
        _data = noise_evoked.data[:, idx:idx+11]

        # Init cnn
        weights = _data[:, np.newaxis, :]
        cnn = CNN_Model(weights).to(DEVICE)
        learning_rate = num_channels * num_times / np.max(weights)

        # De-noise
        xs = []
        for sample in range(num_samples):
            if not events[sample][2] in labels:
                print(f'    {sample} | {num_samples}, Pass')
                continue

            print(f'    {sample} | {num_samples}')
            y_true = data[sample][np.newaxis, :, :]
            x = np.zeros((1, 1, num_times))

            x, y_estimate, r = cnn.fit(x=x, y_true=y_true,
                                       learning_rate=learning_rate)

            if r < allowed_r_min:
                print(f'    r is too small, {r}')
                continue

            data[sample] -= y_estimate.reshape(num_channels, num_times)
            xs.append(x)

        xs = np.concatenate([x.reshape(1, 141) for x in xs], axis=0)

        return mne.BaseEpochs(epochs.info,
                              data,
                              events=epochs.events,
                              tmin=epochs.times[0],
                              tmax=epochs.times[-1]), xs
