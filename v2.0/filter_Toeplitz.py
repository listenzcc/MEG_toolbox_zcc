# File: filter_toeplitz.py
# Aim: Filter the epochs using toeplitz matrix.
# X = D \cdot R \cdot S
# X: Epoch data, times x sensors(272)
# D: Toeplitz matrix, times x 600ms
# R: ERP temporal pattern, 600ms x channels
# S: ERP spatial pattern, channels x sensors

# %%
import numpy as np
import matplotlib.pyplot as plt
from tools.data_manager import DataManager
from tools.computing_components import MyModel, torch, nn, DEVICE, numpy2torch, torch2numpy
from torchsummary import summary

# %%
# Constants

ERP_length = 80

# %%


def extend_matrix(matrix):
    # Extend the matrix by the value
    # Value 1 on the first 60 columns;
    # Value 2 on the second 60 columns;
    # Value 3 on the third 60 columns.
    extended_matrix = np.zeros((ERP_length, ERP_length*3))
    values = [1, 2, 3]
    for j in range(3):
        _matrix = np.zeros((ERP_length, ERP_length))
        _matrix[matrix == values[j]] = 1
        extended_matrix[:, ERP_length*j:ERP_length*(j+1)] = _matrix
    return extended_matrix


def concatenate_matrix(blocks):
    # Concatenate [blocks],
    # blocks is a list of numpy array, with the same size,
    # 1. add new axis as the first dimension
    # 2. concatenate extended arrays and return the concatenated blocks
    assert(isinstance(blocks, list))
    return np.concatenate([e[np.newaxis, :] for e in blocks], axis=0)


class ToeplitzData(object):
    def __init__(self, epochs, ratio=12):
        # Init
        # epochs: The epochs
        # ratio[=12]: Raw sample frequency is 1200 Hz, epochs sample frequency is 100 Hz

        # Load epochs
        self.epochs = epochs

        # The sample frequency of self.events is 100 Hz
        self.events = epochs.events.copy()
        self.events[:, 0] = self.events[:, 0] / ratio

        # Find and record events
        self.events_collection = dict(
            targets=np.where(self.events[:, -1] == 1)[0],
            others=np.where(self.events[:, -1] == 2)[0],
            buttons=np.where(self.events[:, -1] == 3)[0],
        )

        # Numberic names
        self.names = dict(targets=1,
                          others=2,
                          buttons=3)

        # Init memory for speed up
        self.memory = dict()

        # Report
        print(f'ToeplitzData initialized, Epochs has')
        for e in self.events_collection:
            print(e, len(self.events_collection[e]))

    def random_select(self, name, num):
        # Randomly select [name] events as [num] count
        epochs = self.epochs[name]
        length = len(epochs)
        assert(num < length)
        selects = np.random.permutation(range(length))[:num]
        return epochs.events[selects]

    def remember_toeplitz(self, idx, matrix):
        self.memory[idx] = matrix

    def get_toeplitz(self, event):
        # Make toeplitz matrix
        # event: Input event
        #        !!! Note that the sample frequency of input event is 1200 Hz
        #        !!!   The same as self.epochs.events

        # Get data
        idx = np.where(self.epochs.events[:, 0] == event[0])[0]

        data = self.epochs.get_data()[idx]
        data = data[0].transpose()

        # Check memory, return directly if it has memory
        if str(idx) in self.memory:
            return self.memory[str(idx)], data

        # Init matrix
        matrix = np.zeros((ERP_length, ERP_length))

        # Mark 'diags'
        for name in self.names:
            for other_event in self.events[self.events_collection[name]]:
                # Calculate delay,
                # make sure the sample rates are matched.
                d = int(other_event[0] - event[0] / 12)

                # Continue if delay is too large
                if abs(d) > 100:
                    continue

                # Get pair_idxs, [[x1, y1], [x2, y2], ... ]
                pair_idxs = [[e, e-d] for e in range(d, d+ERP_length)]
                # Make sure every [x, y] is in toeplitz matrix
                pair_idxs = [e for e in pair_idxs if all([e[0] > -1,
                                                          e[0] < ERP_length,
                                                          e[1] > -1,
                                                          e[1] < ERP_length])]

                # Continue if no diag to mark
                if len(pair_idxs) == 0:
                    continue

                # Regulation the pair_idxs, make it into Int type
                pair_idxs = np.array(pair_idxs, dtype=np.int)

                # Mark diag
                matrix[pair_idxs[:, 0], pair_idxs[:, 1]] = self.names[name]

        # print(
            # f'Created matrix {matrix.shape}, Got data {data.shape} of idx {idx}, event {event}')
        self.remember_toeplitz(str(idx), matrix)
        return matrix, data


# %%
name = 'MEG_S02'

parameters_meg = dict(picks='mag',
                      stim_channel='UPPT001',
                      l_freq=0.1,
                      h_freq=20,  # 7,
                      tmin=-0.2,
                      tmax=1.2,
                      decim=12,
                      detrend=1,
                      reject=dict(
                          mag=4e-12,
                      ),
                      baseline=None)

dm = DataManager(name, parameters=parameters_meg)
dm.load_epochs(recompute=True)

# %%
epochs_1, epochs_2 = dm.leave_one_session_out(includes=[1, 3, 5],
                                              excludes=[2, 4, 6])


def relabel(_epochs):
    # Relabel _epochs events from 4 to 2
    _epochs.events[_epochs.events[:, -1] == 4, -1] = 2
    # Remove 4 in event_id pool
    _epochs.event_id.pop('4')


relabel(epochs_1)
relabel(epochs_2)
epochs_1 = epochs_1.apply_baseline((None, 0))
epochs_2 = epochs_2.apply_baseline((None, 0))
# Assume the sample frequency is 100 Hz
epochs_1.crop(tmin=0, tmax=ERP_length / 100, include_tmax=False)
epochs_2.crop(tmin=0, tmax=ERP_length / 100, include_tmax=False)
epochs_1, epochs_2

toeplitz_data = ToeplitzData(epochs_1)

# %%
# Plot known 1 epochs, in 447
# for idx in [446, 447, 448]:
#     matrix, data = toeplitz_data.get_toeplitz(epochs_1.events[idx])
#     fig, ax = plt.subplots(1, 1, figsize=(3, 5))
#     ax.imshow(matrix)
#     ax.set_title(idx)

# %%
# Plot randomly selected epochs
# events = toeplitz_data.random_select('1', 5)
# for e in events:
#     matrix, data = toeplitz_data.get_toeplitz(e)
#     fig, ax = plt.subplots(1, 1, figsize=(3*3, 5))
#     matrix = extend_matrix(matrix)
#     ax.imshow(matrix)
#     ax.set_title(e)


# %%


def get_training_session(toeplitz_data,
                         names=None,
                         num_each_name=100):
    # Get training session,
    # paired matrix and data will be got
    # toeplitz_data: The toeplitz data manager,
    # names[=None]: The name of interest,
    #               if is None, use names in epochs in toeplitz_data
    # num_each_name[=100]: The number of each name.

    # Got matrix and data will be in here
    data_list = dict(
        matrix=[],
        data=[]
    )

    # Select [num_each_name] for each event_id(name)
    if names is None:
        names = toeplitz_data.epochs.event_id
        # print(f'Use default names of {names}')

    for name in names:
        events = toeplitz_data.random_select(name, num_each_name)
        # Get matrix and data under event
        for event in events:
            matrix, data = toeplitz_data.get_toeplitz(event)
            matrix = extend_matrix(matrix)
            data_list['matrix'].append(matrix)
            data_list['data'].append(data)

    # Concatenate selected matrix and data
    selected_matrix = concatenate_matrix(data_list['matrix'])
    selected_data = concatenate_matrix(data_list['data'])

    # Return
    return selected_matrix, selected_data


matrix, data = get_training_session(toeplitz_data)
matrix.shape, data.shape

# %%
# Build a Network to learn R and S

model = MyModel(num_channels=6, bias=True).to(DEVICE)
summary(model, (ERP_length, ERP_length * 3))

# %%
learning_rate = 0.05
criterion = nn.MSELoss()
# criterion = nn.L1Loss()

optimizer = torch.optim.Adam(model.list_trainables(),
                             lr=learning_rate)

# optimizer = torch.optim.SGD(model.list_trainables(),
#                             lr=learning_rate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=50,
                                            gamma=0.8)

num_each_name = 100

for j in range(100):
    X, Y = get_training_session(toeplitz_data,
                                names=['1', '2', '3'],
                                num_each_name=num_each_name)
    Y *= 1e14
    X = numpy2torch(X)
    Y = numpy2torch(Y)
    for _ in range(2):
        y = model.forward(X)
        loss = criterion(y, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f'  {j}, Loss: {loss.item()}')

# %%
# m = torch2numpy(model.R1.weight)
# plt.imshow(m)
# print()

# %%
evoked_template = epochs_1.average().copy()


def plot_joint(y,
               task_id=1,
               evoked=evoked_template.copy(),
               title='No title'):
    y_numpy = torch2numpy(y)
    y_mean = np.mean(
        y_numpy[(task_id-1)*num_each_name:task_id*num_each_name], axis=0)
    evoked.data = y_mean.transpose()
    evoked.plot_joint(title=title)


# -----------------------------------------------------------------------
for j, task_id in enumerate([1, 2, 3]):
    plot_joint(y, task_id=j+1, title=f'Estimation {task_id}')
    plot_joint(Y, task_id=j+1, title=f'GroundTruth {task_id}')

# -----------------------------------------------------------------------
# Single diag
for value in [1, 2, 3]:
    matrix = np.zeros((ERP_length, ERP_length))
    for j in range(ERP_length):
        matrix[j, j] = value

    new_matrix = extend_matrix(matrix)
    new_matrix = new_matrix[np.newaxis, :]

    _y = model.forward(numpy2torch(new_matrix))

    evoked = evoked_template.copy()
    evoked.data = torch2numpy(_y)[0].transpose()
    evoked.plot_joint(title=value)
    print()

# Multiple diags
matrix = np.zeros((ERP_length, ERP_length))
for j in range(ERP_length):
    matrix[j, range(j % 10, ERP_length, 10)] = 2

new_matrix = extend_matrix(matrix)
new_matrix = new_matrix[np.newaxis, :]

_y = model.forward(numpy2torch(new_matrix))

evoked = evoked_template.copy()
evoked.data = torch2numpy(_y)[0].transpose()
evoked.plot_joint(title='m2')
print()


# %%
epochs_1

# %%
