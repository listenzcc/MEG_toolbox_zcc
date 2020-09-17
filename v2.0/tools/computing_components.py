# File: computing_components.py
# Aim: Provide computing components for MEG/EEG data processing

# Computing
import mne
import numpy as np

# Torch
import torch
import torch.nn as nn
DEVICE = 'cuda'


# %%
# Xdawn worker
class MyXdawn(object):
    # Customized Xdawn object
    #   xdawn: The xdawn filter object
    #   train_epochs: The epochs to train xdawn
    #   __init__(train_epochs[, n_components=6]): Init method using [n_components]
    #   fit(): Fit the xdawn train_epochs
    #   transform([, epochs=None]): Transfrom [epochs] or train_epochs if [epochs] is None
    #   apply([, epochs=None]): Apply xdawn to [epochs] or train_epochs if [epochs] is None

    def __init__(self, train_epochs, n_components=6):
        self.xdawn = mne.preprocessing.Xdawn(n_components=n_components)
        self.train_epochs = train_epochs
        print('MyXdawn initialized successfully')

    def fit(self):
        self.xdawn.fit(self.train_epochs)
        print('MyXdawn fitted successfully')

    def transform(self, epochs=None):
        if epochs is None:
            return self.xdawn.transform(self.train_epochs)
        return self.xdawn.transform(epochs)

    def apply(self, epochs=None):
        if epochs is None:
            return self.xdawn.apply(self.train_epochs)
        return self.xdawn.apply(epochs)


# %%
# Local tools for pytorch
def numpy2torch(array, dtype=np.float32, device=DEVICE):
    # Attach [array] to the type of torch
    return torch.from_numpy(array.astype(dtype)).to(device)


def torch2numpy(tensor):
    # Detach [tensor] to the type of numpy
    return tensor.detach().cpu().numpy()


# %%
# Pytorch model
class MyModel(nn.Module):
    # Pytorch linear model for learning ERP temporal and spatial patterns
    # X = D \cdot R \cdot S
    # X: Epoch data, times(100) x sensors(272)
    # D: Toeplitz matrix, times(100) x 600ms(60)
    # R: ERP temporal pattern, 600ms(60) x channels(?)
    # S: ERP spatial pattern, channels(?) x sensors(272)

    def __init__(self, num_channels=6):
        super(MyModel, self).__init__()
        self.num_channels = num_channels
        self.trainables = dict()
        self.init_layers()

    def append_trainable(self, tensor, name='default'):
        while name in self.trainables:
            name += '_1'
        self.trainables[name] = tensor
        print(f'Added new trainable of {name}')

    def list_trainables(self):
        tensors = []
        print('Listing trainables: ')
        for j, key in enumerate(self.trainables):
            print(j, key)
            tensors.append(self.trainables[key])
        print('That is all')
        return tensors

    def init_layers(self):
        # self.R = nn.Conv1d(in_channels=100,
        #                    out_channels=self.num_channels,
        #                    kernel_size=60,
        #                    padding=30)

        # --------------------------------------------------
        self.R1 = nn.Linear(in_features=60,
                            out_features=self.num_channels,
                            bias=False)

        self.S1 = nn.Linear(in_features=self.num_channels,
                            out_features=272,
                            bias=False)

        # --------------------------------------------------
        self.R2 = nn.Linear(in_features=60,
                            out_features=self.num_channels,
                            bias=False)

        self.S2 = nn.Linear(in_features=self.num_channels,
                            out_features=272,
                            bias=False)

        # --------------------------------------------------
        self.R3 = nn.Linear(in_features=60,
                            out_features=self.num_channels,
                            bias=False)

        self.S3 = nn.Linear(in_features=self.num_channels,
                            out_features=272,
                            bias=False)

        for linear, name in zip([self.R1, self.R2, self.R3, self.S1, self.S2, self.S3],
                                ['R1', 'R2', 'R3', 'S1', 'S2', 'S3']):
            self.append_trainable(linear.weight, name)

    def forward(self, D):
        # print(f'D size is {D.shape}')
        D1 = D[:, :, 0:60]
        D2 = D[:, :, 60:120]
        D3 = D[:, :, 120:180]

        # ---------------
        y1 = self.R1(D1)
        y1 = self.S1(y1)

        # ---------------
        y2 = self.R2(D2)
        y2 = self.S2(y2)

        # ---------------
        y3 = self.R3(D3)
        y3 = self.S3(y3)

        y = y1 + y2 + y3
        return y

# %%


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
