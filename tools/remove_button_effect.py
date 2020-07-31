# %% Importing
# Computing
import mne
import numpy as np

# Torch
import torch
import torchsummary
import torch.nn as nn

# Settings
DEVICE = 'cuda'

# %% CNN Model


def numpy2torch(array, dtype=np.float32, device=DEVICE):
    return torch.from_numpy(array.astype(dtype)).to(device)


def torch2numpy(tensor):
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
        self.kernel_weights = kernel_weights
        self.num_channels = num_channels
        self.init_layers()
        self.reset_parameters()

    def init_layers(self):
        # Init the conv layer, named as L1
        self.L1 = nn.Conv1d(in_channels=1,
                            out_channels=self.num_channels,
                            kernel_size=11,
                            padding=5)

    def reset_parameters(self):
        # Setup the parameter of L1
        # Parameter of weights
        _shape = self.L1.weight.shape
        self.L1.weight = nn.parameter.Parameter(self.kernel_weights)
        self.L1.weight.requires_grad = False

        # Parameter of bias
        _shape = self.L1.bias.shape
        self.L1.bias = nn.parameter.Parameter(torch.zeros(_shape))
        self.L1.bias.requires_grad = True

    def init_training(self, timeline, learning_rate):
        # Init training stuffs
        # Input
        self.timeline = timeline

        # Computation of loss
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = torch.optim.Adam([timeline], lr=learning_rate)

    def perform_training(self, y_true, steps=31):
        # Perform training steps
        for step in range(steps):
            # Forward
            y = self.forward(self.timeline)

            # Compute loss
            loss = self.criterion(y, y_true)

            # Back Pursuit
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Report every 10 steps
            if step % 10 == 0:
                print(f'    {step:03d}, Loss: {loss.item()}')

        # Return
        return self.timeline, y

    def forward(self, x):
        # Forward flow
        y = self.L1(x)
        return y


# %% Regress Worker

class Button_Effect_Remover():
    def __init__(self, epochs, sfreq):
        self.epochs = epochs
        self.sfreq = sfreq

    def _set_cnn(self, kernel_weights):
        self.cnn = CNN_Model(kernel_weights)

    def _compute_lags(self, e1, e3):
        # Get events
        events1 = self.epochs[e1].events
        events3 = self.epochs[e3].events

        # Init lags
        lags = np.zeros(events1.shape[0])

        # Compute lag for each event 1
        for j, event in enumerate(events1):
            subs = events3[:, 0] - event[0]
            # Compute nearest lag and change it into seconds
            if len(subs[subs > 0]) == 0:
                # In case there are no positive values
                lag = np.inf
            else:
                lag = np.min(subs[subs > 0]) / self.sfreq
            # Record
            lags[j] = lag

        # Return
        print(f'Computed lags of {len(lags)} samples.')
        return lags

    def zero_out_button(self, e1='1', e2='2', e3='3'):
        # Settings ------------------------------------------------
        # Times
        times = self.epochs.times

        # Data of [e1, e2]
        data12 = self.epochs[[e1, e2]].get_data()
        num_samples, num_channels, num_times = data12.shape
        button_data = data12.copy()
        lags = self._compute_lags(e1=e1, e3=e3)

        # Data of [e3]
        evoked3 = self.epochs[e3].average()
        averaged_data3 = evoked3.data

        # Find 11 time points at 0.0 seconds as kernel
        idx = np.where(times == 0)[0][0]
        kernel = averaged_data3[:, idx-5:idx+6]
        kernel_weights = numpy2torch(kernel[:, np.newaxis, :])

        # CNN Model
        cnn = CNN_Model(kernel_weights).to(DEVICE)
        learning_rate = num_channels * num_times / kernel_weights.max().cpu().numpy()

        # Training -------------------------------------------------
        timelines = np.zeros((num_samples, num_times))
        for sample_idx in range(num_samples):
            # Start
            print(f'{sample_idx} | {num_samples}')

            # Prepare y_true,
            # as the target response
            y_true = numpy2torch(data12[sample_idx][np.newaxis, :, :])

            # Init timeline as x
            x = numpy2torch(np.zeros((1, 1, 141)))
            x.requires_grad = True

            # Init CNN training stuffs
            cnn.init_training(timeline=x,
                              learning_rate=learning_rate)

            # Training and get results
            new_x, y = cnn.perform_training(y_true=y_true)
            timeline = torch2numpy(new_x).reshape((141,))

            # Record timeline in timelines
            timelines[sample_idx] = timeline

            # Record y in button_data
            button_data[sample_idx] = torch2numpy(y).reshape(num_channels,
                                                             num_times)

        print(f'Estimated timeline for {len(timelines)} samples')

        # Fix -------------------------------------------------------
        # Re-order timelines and orders according as orders
        _orders = np.argsort(lags)
        paried_lags_timelines = dict(
            sorted_lags=lags[_orders],
            sorted_timelines=timelines[_orders]
        )

        # Substract button_data from data12
        clean_data12 = data12 - button_data
        epochs = self.epochs[[e1, e2]]
        tmin, tmax = epochs.times[0], epochs.times[-1]
        clean_epochs = mne.BaseEpochs(epochs.info,
                                      clean_data12,
                                      events=epochs.events,
                                      tmin=tmin,
                                      tmax=tmax)

        # Returns ---------------------------------------------------
        return clean_epochs, paried_lags_timelines
