# File: computing_components.py
# Aim: Provide computing components for MEG/EEG data processing

# Computing
import mne
import numpy as np

# Torch
import torch
import torch.nn as nn
DEVICE = 'cuda'


class PreprocessXdawn():
    # !!! I am not sure if the method is necessary
    def __init__(self, train_epochs, test_epochs, n_components=6):
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.n_components = n_components
        self.xdawn = mne.preprocessing.Xdawn(n_components=n_components)

    def fit(self):
        self.xdawn.fit(self.train_epochs)

    def apply(self, target_event_id):
        applied_train_epochs = self.xdawn.apply(self.train_epochs)
        applied_test_epochs = self.xdawn.apply(self.test_epochs)

        return applied_train_epochs, applied_test_epochs

    def transform(self, baseline=(None, 0)):
        train_epochs = self.train_epochs.copy()
        test_epochs = self.test_epochs.copy()

        train_epochs.apply_baseline(baseline)
        test_epochs.apply_baseline(baseline)

        train_data = self.xdawn.transform(train_epochs)
        test_data = self.xdawn.transform(test_epochs)

        return train_data, test_data

    def fit_apply(self, target_event_id='1'):
        self.fit()
        return self.apply(target_event_id)

    def fit_transform(self, target_event_id='1'):
        self.fit()
        return self.transform()


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

    def fit(self, noise_evoked, noise_events=None):
        self.noise_evoked = noise_evoked
        self.noise_events = noise_events

    def transform(self, epochs, labels=[1, 2, 4], force_near=True):
        # Get data from [self.epochs]
        data = epochs.get_data()
        events = epochs.events
        num_samples, num_channels, num_times = data.shape

        # Get 11 time points after 0 seconds in noise_evoked
        noise_evoked = self.noise_evoked
        noise_events = self.noise_events
        idx = np.where(noise_evoked.times == 0)[0][0]
        _data = noise_evoked.data[:, idx:idx+11]

        # Init cnn
        weights = _data[:, np.newaxis, :]
        cnn = CNN_Model(weights).to(DEVICE)
        learning_rate = num_channels * num_times / np.max(weights)

        def is_near(pos, events):
            if force_near:
                return True

            remain = events[events > pos]
            if len(remain) == 0:
                return False

            return remain[0] - pos < 800

        # De-noise
        xs = []
        for sample in range(num_samples):
            if not events[sample][2] in labels:
                print(f'    {sample} | {num_samples}, Pass for [not in label]')
                continue

            try:
                if not is_near(events[sample][0], noise_events[:, 0]):
                    print(f'    {sample} | {num_samples}, Pass for [not near]')
                    continue
            except:
                print('Fail on near block')
                raise Exception('Fail on near block.')

            print(f'    {sample} | {num_samples}')
            y_true = data[sample][np.newaxis, :, :]
            x = np.zeros((1, 1, num_times))

            x, y_estimate, r = cnn.fit(x=x, y_true=y_true,
                                       learning_rate=learning_rate)

            data[sample] -= y_estimate.reshape(num_channels, num_times)
            xs.append(x)

        xs = np.concatenate([x.reshape(1, 141) for x in xs], axis=0)

        return mne.BaseEpochs(epochs.info,
                              data,
                              events=epochs.events,
                              tmin=epochs.times[0],
                              tmax=epochs.times[-1]), xs
