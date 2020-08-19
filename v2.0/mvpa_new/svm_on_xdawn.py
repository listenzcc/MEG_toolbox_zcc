# %%
import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn import metrics
from sklearn import manifold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mne.decoding import Vectorizer

import matplotlib.pyplot as plt
# %%
import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)


def plot(scatters, title='Title'):
    if isinstance(scatters, dict):
        scatters = [scatters]
    layout = go.Layout(title=title)
    data = [go.Scatter(**scatter) for scatter in scatters]
    plotly.offline.iplot(dict(data=data,
                              layout=layout))


# %%
try:
    display('Display is useable.')
    is_notebook = True
except:
    def display(*args):
        print(args)
    is_notebook = False

# %%
data_folder = os.path.join('MVPA_data_xdawn_v2')
xtsne_folder = os.path.join('tsne_xdawn_x3')
names = sorted(os.listdir(xtsne_folder))
# display(names)

# %%
for name in names:
    print(name)

    # Read data -------------------------------------------------
    with open(os.path.join(data_folder, name), 'rb') as f:
        DATA_xraw = pickle.load(f)

    with open(os.path.join(data_folder, f'raw_events_{name}'), 'rb') as f:
        DATA_raw_events = pickle.load(f)

    with open(os.path.join(xtsne_folder, name), 'rb') as f:
        DATA_xtsne = pickle.load(f)

    xtsne = DATA_xtsne['x3']
    train_y = DATA_xtsne['train_y']
    test_y = DATA_xtsne['test_y']
    raw_train_events = DATA_raw_events['raw_train_events']
    raw_test_events = DATA_raw_events['raw_train_events']
    train_xraw = DATA_xraw['train_X']
    test_xraw = DATA_xraw['test_X']
    train_events = DATA_xraw['train_events']
    test_events = DATA_xraw['test_events']

    # Split data -----------------------------------------------

    def split_data(xtsne, n):
        train_xtsne = xtsne[:n]
        test_xtsne = xtsne[n:]
        scaler = StandardScaler()
        scaler.fit(train_xtsne)
        train_xtsne = scaler.transform(train_xtsne)
        test_xtsne = scaler.transform(test_xtsne)

        return train_xtsne, test_xtsne

    train_xtsne, test_xtsne = split_data(xtsne,
                                         n=len(train_y))

    button_train_events = raw_train_events[raw_train_events[:, -1] == 3]
    button_test_events = raw_test_events[raw_test_events[:, -1] == 3]

    train_selects = []
    for j, event in enumerate(train_events):
        if event[-1] == 1:
            lags = button_train_events[:, 0] - event[0]
            lags = lags[lags > 0]
            if len(lags) == 0:
                continue
            if not lags[0] < 0.4 * 1200:
                train_selects.append(j)
        else:
            train_selects.append(j)

    test_selects = []
    for j, event in enumerate(test_events):
        if event[-1] == 1:
            lags = button_test_events[:, 0] - event[0]
            lags = lags[lags > 0]
            if len(lags) == 0:
                continue
            if not lags[0] < 0.4 * 1200:
                test_selects.append(j)
        else:
            test_selects.append(j)

    display('Before',
            train_xtsne.shape, test_xtsne.shape,
            train_xraw.shape, test_xraw.shape,
            train_y.shape, test_y.shape)

    # train_xtsne = train_xtsne[train_selects]
    # test_xtsne = test_xtsne[test_selects]
    # train_xraw = train_xraw[train_selects]
    # test_xraw = test_xraw[test_selects]
    # train_y = train_y[train_selects]
    # test_y = test_y[test_selects]

    display('After',
            train_xtsne.shape, test_xtsne.shape,
            train_xraw.shape, test_xraw.shape,
            train_y.shape, test_y.shape)

    # stophere
    # break

    # TSNE -----------------------------------------------------------
    def tsne_fit_predict(train_xtsne, train_y, test_xtsne):
        # Prepare train data -------------------------------
        dim = train_xtsne.shape[1]
        train_xe = []
        for j in range(len(train_y)):
            d = train_xtsne[j-3:j+4]
            if not len(d) == 7:
                train_xe.append(np.zeros(7 * dim))
                continue
            train_xe.append(np.concatenate(d))
        train_xe = np.array(train_xe)
        target_xe_mean = np.mean(train_xe[train_y == 1], axis=0)
        print(train_xe.shape, train_y.shape, target_xe_mean.shape)
        subs = np.array([target_xe_mean-e for e in train_xe])
        tsne_prob_train = 1 / np.linalg.norm(subs, axis=1)

        # Prepare test data -------------------------------
        pred_y = test_y * 0
        test_xe = []
        for j in range(len(test_xtsne)):
            d = test_xtsne[j-3:j+4]
            if not len(d) == 7:
                test_xe.append(np.zeros(7 * dim))
                continue
            test_xe.append(np.concatenate(d))
        test_xe = np.array(test_xe)

        # Compute prob of tsne
        subs = np.array([target_xe_mean-e for e in test_xe])
        prob = 1 / np.linalg.norm(subs, axis=1)
        return prob

    tsne_prob = tsne_fit_predict(train_xtsne, train_y, test_xtsne)

    # SVM -----------------------------------------------------------
    def svm_fit_predict(train_xraw, train_y, test_xraw):
        clf = svm.SVC(gamma='scale',
                      kernel='rbf',
                      class_weight='balanced',
                      probability=True)
        selects = []
        for j, y in enumerate(train_y):
            if y == 1:
                [selects.append(j-e) for e in [-1, 0, 1]]

        pipeline = make_pipeline(Vectorizer(), clf)
        pipeline.fit(train_xraw[selects, :, 40:80], train_y[selects])
        pred = pipeline.predict(test_xraw[:, :, 40:80])
        prob = pipeline.predict_proba(test_xraw[:, :, 40:80])
        return pred, prob

    svm_pred, svm_prob = svm_fit_predict(train_xraw, train_y, test_xraw)

    outputs = dict(
        svm_pred=svm_pred,
        svm_prob=svm_prob,
        tsne_prob=tsne_prob,
        test_y=test_y,
        name=name
    )

    with open(os.path.join('tmpfile_v2_1',
                           f'{name}.pkl'), 'wb') as f:
        pickle.dump(outputs, f)
        display(f'Saved {f.name}')
        pass


# %%
# y_true = test_y.copy()
# y_pred = svm_pred.copy()
# print(metrics.classification_report(y_pred=y_pred, y_true=y_true))

# plot([dict(y=y_true, name='True'),
#       dict(y=5 + y_pred, name='Pred'),
#       dict(y=5 + svm_prob[:, 0], name='SVM prob'),
#       dict(y=5 - tsne_prob, name='TSNE prob')])

# %%
# y_true = test_y.copy()
# y_true[y_true != 1] = 0

# joint_prob = svm_prob[:, 0] * tsne_prob
# new_joint_prob = joint_prob * 0
# for j in range(2, len(joint_prob)-2):
#     new_joint_prob[j] = np.dot([joint_prob[j+e] for e in [-2, -1, 0, 1, 2]],
#                                [-0.2, -0.3, 2, -0.3, -0.2])

# joint_prob = new_joint_prob.copy()

# y_pred = joint_prob * 0
# y_pred[joint_prob > 0.25] = 1
# print(metrics.classification_report(y_pred=y_pred, y_true=y_true))

# plot([dict(y=y_true, name='True'),
#       dict(y=joint_prob, name='Joint prob')])

# %%
