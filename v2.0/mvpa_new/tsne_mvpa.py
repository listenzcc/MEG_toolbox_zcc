# %%
import os
import sys
import pickle
import numpy as np
import pandas as pd

import mne
from mne.decoding import Vectorizer

from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import pca
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
        for e in args:
            print(e)
    is_notebook = False


def compute_scores(y_true, y_pred):
    tp = 0
    num_1 = len(np.where(y_true == 1)[0])
    num_pos = 0
    for j, p in enumerate(y_pred):
        if p == 1:
            num_pos += 1
        else:
            continue

        try:
            if 1 in y_true[j-1:j+2]:
                tp += 1
        except:
            pass

    try:
        recall = tp / num_1
    except:
        recall = 0

    try:
        precision = tp / num_pos
    except:
        precision = 0

    return dict(_recall=recall, _precision=precision)


def combine(d1, d2):
    for key in d2:
        assert(key not in d1)
        d1[key] = d2[key]


# %%
folder_x3 = os.path.join('tsne_xdawn_x3')
names_x3 = sorted(os.listdir(folder_x3))
print(names_x3)

folder_data = os.path.join('MVPA_data_xdawn')
# %%
frame = pd.DataFrame()
for name in names_x3:
    # name = 'MEG_S03-3.pkl'
    print(name)
    with open(os.path.join(folder_x3, name), 'rb') as f:
        DATA_x3 = pickle.load(f)

    with open(os.path.join(folder_data, name), 'rb') as f:
        DATA_xx6 = pickle.load(f)

    def split_data(data_x3, data_xx6):
        x3 = data_x3['x3']
        xx6 = data_xx6['x6']
        train_y = DATA_x3['train_y']
        test_y = DATA_x3['test_y']
        n = len(train_y)

        train_x3 = x3[:n]
        test_x3 = x3[n:]
        scaler = StandardScaler()
        scaler.fit(train_x3)
        train_x3 = scaler.transform(train_x3)
        test_x3 = scaler.transform(test_x3)

        train_xx6 = xx6[:n]
        test_xx6 = xx6[n:]
        scaler6 = StandardScaler()
        scaler6.fit(train_xx6)
        train_xx6 = scaler6.transform(train_xx6)
        test_xx6 = scaler6.transform(test_xx6)
        train_xx6 = train_xx6.reshape(len(train_xx6), 6, 141)
        test_xx6 = test_xx6.reshape(len(test_xx6), 6, 141)

        return train_x3, test_x3, train_xx6, test_xx6, train_y, test_y

    train_x3, test_x3, train_xx6, test_xx6, train_y, test_y = split_data(
        DATA_x3, DATA_xx6)

    print('Got data {}, {}, {}, {}, {}, {}'.format(train_x3.shape,
                                                   test_x3.shape,
                                                   train_xx6.shape,
                                                   test_xx6.shape,
                                                   train_y.shape,
                                                   test_y.shape))

    def svm_proba(train_xx6, train_y, test_xx6):
        selects = []
        for j, y in enumerate(train_y):
            if y == 1:
                [selects.append(j-e) for e in [-2, -1, 0, 1, 2]]
        train_xx6 = train_xx6[selects]
        train_xx6 = train_xx6[:, :, 40:80]
        test_xx6 = test_xx6[:, :, 40:80]
        train_y = train_y[selects]
        train_y[train_y != 1] = 0

        # s = train_xx6.shape
        # train_xx6 = train_xx6.reshape((s[0], s[1] * s[2]))
        # s = test_xx6.shape
        # test_xx6 = test_xx6.reshape((s[0], s[1] * s[2]))

        # return test_xx6

        clf = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            pca.PCA(n_components=.95),
            svm.SVC(gamma='scale',
                    kernel='rbf',
                    class_weight={0: 1, 1: 2},
                    probability=True))
        clf.fit(train_xx6, train_y)
        return clf.predict_proba(test_xx6)

    def tsne_proba(train_x3, train_y, test_x3):
        dim = train_x3.shape[1]
        # Prepare train data -------------------------------
        train_xe = []
        for j in range(len(train_y)):
            d = train_x3[j-3:j+4]
            if not len(d) == 7:
                train_xe.append(np.zeros(7 * dim))
                continue
            train_xe.append(np.concatenate(d))
        train_xe = np.array(train_xe)
        target_xe_mean = np.mean(train_xe[train_y == 1], axis=0)
        # print(train_xe.shape, train_y.shape, target_xe_mean.shape)
        subs = np.array([target_xe_mean-e for e in train_xe])
        tsne_prob_train = 1 / np.linalg.norm(subs, axis=1)

        # t1 = max(tsne_prob_train[train_y != 1])
        # t2 = min(tsne_prob_train[train_y == 1])
        # threshold = (t1 + t2) / 2

        # s = np.array(sorted(tsne_prob_train))
        # p1 = np.where(s == t1)[0][0]
        # p2 = np.where(s == t2)[0][0]
        # threshold = s[np.int((p1 + p2) / 2)]

        # Prepare test data -------------------------------
        pred_y = test_y * 0
        test_xe = []
        for j in range(len(test_y)):
            d = test_x3[j-3:j+4]
            if not len(d) == 7:
                test_xe.append(np.zeros(7 * dim))
                continue
            test_xe.append(np.concatenate(d))
        test_xe = np.array(test_xe)

        # Compute prob of tsne
        subs = np.array([target_xe_mean-e for e in test_xe])
        return 1 / np.linalg.norm(subs, axis=1)

    # svm_prob = svm_proba(train_xx6, train_y, test_xx6)[:, 1]
    # plot([dict(y=svm_prob, name='svm'),
    #       dict(y=test_y, name='True')])
    # print(metrics.classification_report(y_pred=svm_pred, y_true=test_y))
    # break

    tsne_prob = tsne_proba(train_x3, train_y, test_x3)
    threshold = 0.5

    y_true = test_y * 0
    y_true[test_y == 1] = 1

    y_pred = test_y * 0
    # for j, _ in enumerate(y_pred):
    #     if tsne_prob[j] > threshold:
    #         p3 = np.add(svm_prob[j-1:j+2], tsne_prob[j-1:j+2])
    #         m = np.where(p3 == max(p3))[0]
    #         y_pred[j-1+m] = 1
    y_pred[tsne_prob > threshold] = 1

    if False:  # True:
        plot([dict(y=tsne_prob / threshold, name='TSNE'),
              dict(y=test_y, name='True'),
              dict(y=y_pred-y_true-1, name='Diff')])

    scores = metrics.classification_report(y_pred=y_pred,
                                           y_true=y_true,
                                           output_dict=True)

    _scores = compute_scores(y_pred=y_pred,
                             y_true=y_true)
    combine(_scores, scores['1'])

    se = pd.Series(_scores,
                   name=name[:9])
    frame = frame.append(se)
    display(frame.describe())
    # break

display(frame.describe())

# %%
print('All done.')

# %%
frame
# %%
frame.to_html('results_tsne.html')
# %%
