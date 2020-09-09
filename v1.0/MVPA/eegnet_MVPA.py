# %%
# System ----------------------------------------
import os
import sys
import pickle

# Computing -------------------------------------
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Local tools -----------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))  # noqa
from MEG_worker import MEG_Worker

# EEG net ---------------------------------------
import torchsummary
from eegnet import torch, nn, EEGNet, optim, Variable, evaluate
net = EEGNet().cuda()
print(net.forward(Variable(torch.Tensor(np.random.rand(1, 1, 61, 272)).cuda())))
torchsummary.summary(net, input_size=(1, 61, 272))

# Init parameters -------------------------------
# Init results_folder
RESULTS_FOLDER = os.path.join('.', 'results_eegnet')

# Make sure RESULTS_FOLDER is not a file
assert(not os.path.isfile(RESULTS_FOLDER))

# Mkdir if RESULTS_FOLDER does not exist
if not os.path.exists(RESULTS_FOLDER):
    os.mkdir(RESULTS_FOLDER)

# %%


class EEGNet_classifier():
    def __init__(self):
        self.net = EEGNet().cuda()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=20,
                                                   gamma=0.5)

    def fit(self, X, y, quiet=True):
        for epoch in range(100):
            running_loss = 0
            _skf = StratifiedKFold(n_splits=10, shuffle=True)
            for more, less in _skf.split(X, y):
                # Get data and labels
                inputs = torch.from_numpy(X[more])
                labels = torch.FloatTensor(np.array([y[more]]).T*1.0)

                # Make sure positive and negative samples have the same number
                inputs = inputs[:torch.sum(labels).type(torch.int16)*2]
                labels = labels[:torch.sum(labels).type(torch.int16)*2]

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.data

            self.scheduler.step()

            if not quiet:
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}: {running_loss}')

    def predict(self, X):
        inputs = torch.from_numpy(X)
        inputs = Variable(inputs.cuda())
        return self.net(inputs)


# %%
# Set crop
crops = dict(
    a=(0.2, 0.4),
    b=(0.4, 0.6),
    c=(0.6, 0.8),
    d=(0.2, 0.8),
    e=(0.0, 1.0))


def pair_X_y(epochs, label, crop):
    '''pair_X_y Get paired X and y,

    Make sure the epochs has only one class of data,
    the label is the int representation of the class.

    Args:
        epochs ({Epochs}): The epochs object to get data from
        label ({int}): The label of the data

    Returns:
        [type]: [description]
    '''
    X = epochs.copy().crop(crop[0], crop[1]).get_data()
    num = X.shape[0]
    y = np.zeros(num,) + label
    print(f'Got paired X: {X.shape} and y: {y.shape}')
    return X, y


# %%
for idx in range(1, 11):
    # Loading data ------------------------------------------
    running_name = f'MEG_S{idx:02d}'
    band_name = 'U07'

    worker = MEG_Worker(running_name=running_name)
    worker.pipeline(band_name=band_name)

    crop_key = 'e'
    crop = crops[crop_key]
    # Get X and y for class 1
    X1, y1 = pair_X_y(worker.clean_epochs, 1, crop)

    # Get X and y for class 2
    X2, y2 = pair_X_y(worker.denoise_epochs['2'], 2, crop)

    # Concatenate X and y
    X_all = np.concatenate([X1, X2], axis=0)
    X_all = X_all / np.max(X_all)
    y_all = np.concatenate([y1, y2], axis=0)
    y_all = 2 - y_all

    # Estimate n_splits
    n_splits = int(y1.shape[0] / 56)
    print(f'Splitting in {n_splits} splits')

    # Cross validation using sliding window -------------------------------
    # Prepare predicted label matrix
    num_samples, num_times = X_all.shape[0], X_all.shape[2]
    y_pred = np.zeros((num_samples,))

    # Cross validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    for train_index, test_index in skf.split(X_all, y_all):
        # Separate training and testing data
        X_train, y_train = X_all[train_index], y_all[train_index]
        X_test, y_test = X_all[test_index], y_all[test_index]

        X_train = X_train.transpose([0, 2, 1])[:, np.newaxis].astype('float32')
        X_test = X_test.transpose([0, 2, 1])[:, np.newaxis].astype('float32')
        print(X_train.shape, X_test.shape)

        eegnet = EEGNet_classifier()
        eegnet.fit(X_train, y_train, quiet=False)
        output = eegnet.predict(X_test)
        y_pred[test_index] = output.detach().cpu().numpy().ravel()

        params = ['acc', 'recall', 'precision', 'fmeasure']
        print(params)
        print('Train - ', evaluate(eegnet.net, X_train, y_train, params))
        print('Test - ', evaluate(eegnet.net, X_test, y_test, params))
        print('-------------------------------------------------\n')

    # Summary results
    output_dict = dict(y_all=y_all,
                       y_pred=y_pred)

    # Save results
    with open(os.path.join(RESULTS_FOLDER,
                           f'{running_name}_eegnet.pkl'), 'wb') as f:
        pickle.dump(output_dict, f)


# %%
# for epoch in range(100):  # loop over the dataset multiple times
#     # print('\nEpoch ', epoch)
#     scheduler.step()
#     running_loss = 0.0
#     _skf = StratifiedKFold(n_splits=10, shuffle=True)
#     for more, less in _skf.split(X_train, y_train):
#         inputs = torch.from_numpy(X_train[more])
#         labels = torch.FloatTensor(np.array([y_train[more]]).T*1.0)

#         inputs = inputs[:torch.sum(labels).type(torch.int16)*2]
#         labels = labels[:torch.sum(labels).type(torch.int16)*2]

#         # wrap them in Variable
#         inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()

#         optimizer.step()

#         running_loss += loss.data

#     # Validation accuracy
#     if epoch % 10 == 0:
#         print(f'\nEpochs {epoch}')
#         params = ['acc', 'recall', 'precision', 'fmeasure']
#         print(params)
#         print('Training Loss ', running_loss)
#         print('Train - ', evaluate(net, X_train, y_train, params))
#         # print('Validation - ', evaluate(net, X_val, y_val, params))
#         print('Test - ', evaluate(net, X_test, y_test, params))
#         print('-------------------------------------------------\n')
#     print('.', end='', flush=True)

# # %%
# net.forward(Variable(torch.Tensor(X_test / np.max(X_test)).cuda()))

# # %%
# net(X_test / np.max(X_test))
# # %%
