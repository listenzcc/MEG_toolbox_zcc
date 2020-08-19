
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
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)

# %%
try:
    display('Display is useable.')
    is_notebook = True
except:
    def display(obj):
        print(obj)
    is_notebook = False


def plot_train_test(train_xx, train_y, test_xx, test_y):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(3, 1, figsize=(10, 30))
    for j in [4, 2, 1]:
        print(j)
        axes[0].scatter(train_xx[train_y == j, 0],
                        train_xx[train_y == j, 1],
                        label=j,
                        alpha=0.5)

        axes[1].scatter(train_xx[train_y == j, 0],
                        train_xx[train_y == j, 1],
                        label=j,
                        alpha=0.3)

        axes[1].scatter(test_xx[test_y == j, 0],
                        test_xx[test_y == j, 1],
                        label=10+j,
                        alpha=1)

        axes[2].scatter(test_xx[test_y == j, 0],
                        test_xx[test_y == j, 1],
                        label=10+j,
                        alpha=0.5)

    axes[0].set_title('Training embedding')
    axes[1].set_title('Joint embedding')
    for ax in axes:
        ax.legend()

    fig.tight_layout()

    return fig


# %%
folder_x3 = os.path.join('.', 'tsne_xdawn_x3')
folder_x6 = os.path.join('.', 'MVPA_data_xdawn')
names_x3 = sorted(os.listdir(folder_x3))
print(names_x3)

# %%
# for name in names:
#     with open(os.path.join(results_dir, name), 'rb') as f:
#         DATA = pickle.load(f)
#     break

# x6 = DATA['x6']
# train_y = DATA['train_y']
# test_y = DATA['test_y']

# train_x = x6.reshape(11193, 6, 141)[:len(train_y)]
# test_x = x6.reshape(11193, 6, 141)[len(train_y):]

# fig, axes = plt.subplots(3, 2, figsize=(8, 12))
# for j, event in enumerate([1, 2, 4]):
#     xx = np.mean(train_x[train_y == event], axis=0)
#     axes[j][0].plot(xx.transpose())
#     axes[j][0].set_title(event)

#     xx = np.mean(test_x[test_y== event], axis=0)
#     axes[j][1].plot(xx.transpose())
#     axes[j][1].set_title(event)

# fig.tight_layout()

# %%
frame = pd.DataFrame()
for name in names_x3:
    # name = 'MEG_S03-3.pkl'
    print(name)
    with open(os.path.join(folder_x3, name), 'rb') as f:
        DATA_x3 = pickle.load(f)

    with open(os.path.join(folder_x6, name), 'rb') as f:
        DATA_x6 = pickle.load(f)
    # display(DATA)

    xx = DATA_x3['x3']
    xx6 = DATA_x6['x6']
    train_y = DATA_x3['train_y']
    test_y = DATA_x3['test_y']
    # print(xx.shape, train_y.shape, test_y.shape)

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
    print('Got data {}, {}, {}, {}'.format(train_xx.shape,
                                           test_xx.shape,
                                           train_xx6.shape,
                                           test_xx6.shape))

    # fig = plot_train_test(train_xx, train_y, test_xx, test_y)

    def svm_proba(train_xx6, train_y, test_xx6):
        selects = []
        for j, y in enumerate(train_y):
            if y == 1:
                [selects.append(j-e) for e in [-2, -1, 0, 1, 2]]
        train_xx6 = train_xx6[selects]
        train_xx6 = train_xx6[:, :, 40:70]
        test_xx6 = test_xx6[:, :, 40:70]
        train_y = train_y[selects]
        train_y[train_y != 1] = 0

        # s = train_xx6.shape
        # train_xx6 = train_xx6.reshape((s[0], s[1] * s[2]))
        # s = test_xx6.shape
        # test_xx6 = test_xx6.reshape((s[0], s[1] * s[2]))

        # return test_xx6

        probs = []
        probs_train = []
        for c in range(6):
            clf = make_pipeline(
                # StandardScaler(),
                svm.SVC(gamma='scale',
                        kernel='rbf',
                        class_weight={0: 1, 1: 1},
                        probability=True))
            clf.fit(train_xx6[:, c, :], train_y)
            probs.append(clf.predict_proba(
                test_xx6[:, c, :])[:, 1][:, np.newaxis])

        # return clf.predict_proba(test_xx6)[:, 1]
        return np.concatenate(probs, axis=1)

    # svm_prob = svm_proba(train_xx6, train_y, test_xx6)

    # clf = train_svm(train_xx, train_y, test_xx, test_y)

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
    for j in range(len(test_y)):
        d = test_xx[j-3:j+4]
        if not len(d) == 7:
            test_xe.append(np.zeros(7 * dim))
            continue
        test_xe.append(np.concatenate(d))
    test_xe = np.array(test_xe)

    # Compute prob of tsne
    subs = np.array([target_xe_mean-e for e in test_xe])
    tsne_prob = 1 / np.linalg.norm(subs, axis=1)

    # Get y_true
    y_true = test_y.copy()
    y_true[y_true != 1] = 0

    svc = make_pipeline(
        # StandardScaler(),
        svm.SVC(gamma='scale',
                kernel='rbf',
                class_weight={0: 1, 1: 5},
                probability=True))

    test_prob = np.concatenate([
        svm_prob,  # [:, np.newaxis],
        tsne_prob[:, np.newaxis],
        # test_xe,
    ], axis=1)

    # svc.fit(test_prob, y_true)
    # svc.fit(train_prob, train_y_selects)
    # y_prob = svc.predict_proba(test_prob)[:, 1]
    # Joint prob
    # joint_prob = svm_prob * tsne_prob
    for j in range(len(y_prob)):
        try:
            if y_prob[j] != np.max(y_prob[j-5:j+6]):
                y_prob[j] = 0
        except ValueError:
            pass
    y_pred = y_prob * 0
    y_pred[y_prob > 0.5] = 1

    # joint_prob_1 = joint_prob[y_true == 1]
    # joint_prob_1[joint_prob_1 == 0] = np.inf
    # joint_prob_0 = joint_prob[y_true == 0]
    # threshold = (np.min(joint_prob_1) + np.max(joint_prob_0)) / 2
    # # sorted_joint_prob = sorted(joint_prob)
    # # idx_1 = np.where(sorted_joint_prob == np.min(joint_prob_1))[0][0]
    # # idx_0 = np.where(sorted_joint_prob == np.max(joint_prob_0))[0][0]
    # # threshold = sorted_joint_prob[int((idx_1 + idx_0) / 2)]
    # # threshold = sorted(joint_prob, reverse=True)[56]

    # y_pred = y_true * 0
    # y_pred[joint_prob > threshold] = 1

    # Report ---------------------------------------------------
    print(name)
    print(metrics.classification_report(y_pred=y_pred, y_true=y_true))
    report = metrics.classification_report(y_pred=y_pred,
                                           y_true=y_true,
                                           output_dict=True)

    frame = frame.append(pd.Series(report['1'], name=name[:7]))
    display(frame.describe())

    # Plot -----------------------------------------------------
    if all([is_notebook, True]):
        plotly.offline.iplot({
            'data': [go.Scatter(y=test_y, name='True'),
                     go.Scatter(y=-tsne_prob, name='TSNE'),
                     go.Scatter(y=-1-y_prob, name='JOINT'),
                     ],
            'layout': go.Layout(title=name)
        })
        plotly.offline.iplot({
            'data': [go.Scatter(y=y_true, name='True'),
                     go.Scatter(y=-y_pred, name='Pred'),
                     go.Scatter(y=3+y_true-y_pred, name='Diff')],
            'layout': go.Layout(title=name)
        })
        break


display(frame)
# display(frame.describe())

# %%
display(frame.describe())

# %%
for name in frame.index.unique():
    print(name)
    df = frame.loc[name]
    display(df.describe())

# %%
