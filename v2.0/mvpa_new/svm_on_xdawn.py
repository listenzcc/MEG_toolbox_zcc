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
x3_folder = os.path.join('tsne_xdawn_x3')
names = sorted(os.listdir(data_folder))
# display(names)

# %%
for name in names:
    print(name)

    # Read data -------------------------------------------------
    with open(os.path.join(data_folder, name), 'rb') as f:
        DATA_x6 = pickle.load(f)
    with open(os.path.join(x3_folder, name), 'rb') as f:
        DATA_x3 = pickle.load(f)
    xx = DATA_x3['x3']
    xx6 = DATA_x6['x6']
    train_y = DATA_x3['train_y']
    test_y = DATA_x3['test_y']

    # Split data -----------------------------------------------
    def split_data(xx, xx6, n):
        train_xx = xx[:n]
        test_xx = xx[n:]
        scaler = StandardScaler()
        scaler.fit(train_xx)
        train_xx = scaler.transform(train_xx)
        test_xx = scaler.transform(test_xx)

        train_xx6 = xx6[:n]
        test_xx6 = xx6[n:]
        scaler6 = StandardScaler()
        scaler6.fit(train_xx6)
        train_xx6 = scaler6.transform(train_xx6)
        test_xx6 = scaler6.transform(test_xx6)
        train_xx6 = train_xx6.reshape(len(train_xx6), 6, 141)
        test_xx6 = test_xx6.reshape(len(test_xx6), 6, 141)

        return train_xx, test_xx, train_xx6, test_xx6

    train_xx, test_xx, train_xx6, test_xx6 = split_data(xx,
                                                        xx6,
                                                        n=len(train_y))
    display(train_xx.shape, test_xx.shape, train_xx6.shape, test_xx6.shape)

    # TSNE -----------------------------------------------------------
    def tsne_fit_predict(train_xx, train_y, test_xx):
        # Prepare train data -------------------------------
        dim = train_xx.shape[1]
        train_xe = []
        for j in range(len(train_y)):
            d = train_xx[j-3:j+4]
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
        for j in range(len(test_xx)):
            d = test_xx[j-3:j+4]
            if not len(d) == 7:
                test_xe.append(np.zeros(7 * dim))
                continue
            test_xe.append(np.concatenate(d))
        test_xe = np.array(test_xe)

        # Compute prob of tsne
        subs = np.array([target_xe_mean-e for e in test_xe])
        prob = 1 / np.linalg.norm(subs, axis=1)
        return prob

    tsne_prob = tsne_fit_predict(train_xx, train_y, test_xx)

    # SVM -----------------------------------------------------------
    def svm_fit_predict(train_xx6, train_y, test_xx6):
        clf = svm.SVC(gamma='scale',
                      kernel='rbf',
                      class_weight='balanced',
                      probability=True)
        selects = []
        for j, y in enumerate(train_y):
            if y == 1:
                [selects.append(j-e) for e in [-1, 0, 1]]

        pipeline = make_pipeline(Vectorizer(), clf)
        pipeline.fit(train_xx6[selects, :, 40:80], train_y[selects])
        pred = pipeline.predict(test_xx6[:, :, 40:80])
        prob = pipeline.predict_proba(test_xx6[:, :, 40:80])
        return pred, prob

    svm_pred, svm_prob = svm_fit_predict(train_xx6, train_y, test_xx6)

    outputs = dict(
        svm_pred=svm_pred,
        svm_prob=svm_prob,
        tsne_prob=tsne_prob,
        test_y=test_y,
        name=name
    )

    with open(os.path.join('tmpfile',
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
