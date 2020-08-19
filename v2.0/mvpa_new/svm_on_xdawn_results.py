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


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole...
        else:
            return False
    except:
        return False


if is_notebook():
    import plotly
    import plotly.graph_objs as go
    plotly.offline.init_notebook_mode(connected=True)


def plot(scatters, title='Title'):
    if not is_notebook():
        print(f'Not display {title}')
        return -1

    if isinstance(scatters, dict):
        scatters = [scatters]
    layout = go.Layout(title=title)
    data = [go.Scatter(**scatter) for scatter in scatters]
    plotly.offline.iplot(dict(data=data,
                              layout=layout))


if not is_notebook():
    def display(*args):
        if len(args) == 1:
            print(args[0])
        else:
            print(args)

# %%
results_folder = os.path.join('tmpfile_v2_1')
names = sorted(os.listdir(results_folder))
# display(names)


class Results():
    def __init__(self, obj):
        for key, value in obj.items():
            self._new(key, value)

    def _new(self, key, value):
        self.__setattr__(key, value)
        print(f'New {key} added')


# %%
frame = pd.DataFrame()
for name in names:
    print(name)

    # Read results -------------------------------------------------
    with open(os.path.join(results_folder, name), 'rb') as f:
        results = pickle.load(f)

    res = Results(results)
    res.svm_pred[res.svm_pred == 4] = 0

    # Metrics ------------------------------------------------------
    y_true = res.test_y.copy()
    y_true[y_true != 1] = 0

    # joint_prob = res.svm_prob[:, 0] * res.tsne_prob
    # new_joint_prob = joint_prob * 0
    # for j in range(2, len(joint_prob)-2):
    #     new_joint_prob[j] = np.dot([joint_prob[j+e] for e in [-2, -1, 0, 1, 2]],
    #                                [-0.2, -0.3, 2, -0.3, -0.2])
    # joint_prob = new_joint_prob.copy()

    # Raw ------------------------------------------------------------
    y_pred = res.tsne_prob * 0
    # y_pred[joint_prob > 0.25] = 1
    y_pred[res.tsne_prob > 0.4] = 1
    report = metrics.classification_report(y_pred=y_pred,
                                           y_true=y_true,
                                           output_dict=True)

    # Joint correction -----------------------------------------------
    y_new_pred = y_pred * 0
    for j in range(1, len(y_pred)-1):
        if y_pred[j] == 1:
            # ps = [res.svm_prob[j+e, 0] - res.svm_prob[j+e, 1]
            #       for e in [-1, 0, 1]]
            # f = np.where(ps == np.max(ps))[0]
            # y_new_pred[j-1+f] = 1
            for k in [-1, 0, 1]:
                if res.svm_prob[j+k, 0] > 0.5:
                    y_new_pred[j+k] = 1
    report = metrics.classification_report(y_pred=y_new_pred,
                                           y_true=y_true,
                                           output_dict=True)

    # Selection metrics ---------------------------------------------
    # selects = []
    # for j, y in enumerate(y_true):
    #     if y == 1:
    #         [selects.append(j-e) for e in range(-1, 2)]
    # report = metrics.classification_report(y_pred=res.svm_pred[selects],
    #                                        y_true=y_true[selects],
    #                                        output_dict=True)

    if report.get('1', None) is None:
        report['1'] = report['1.0']

    # Loosely metrics -----------------------------------------------
    TP, FP = 0, 0
    for j in range(1, len(y_pred)-1):
        if y_pred[j] == 1:
            if any([y_true[j+e] == 1 for e in [-1, 0, 1]]):
                TP += 1
            else:
                FP += 1
    report['1']['_recall'] = TP / np.sum(y_true)
    report['1']['_precision'] = TP / (TP + FP)

    se = pd.Series(report['1'], name=name[:-8])
    frame = frame.append(se)

    plot([dict(y=y_true, name='True'),
          dict(y=1+y_pred, name='Pred'),
          dict(y=1+res.tsne_prob, name='Prob'),
          dict(y=2+y_new_pred, name='New')])

    # display(frame.describe())

    if is_notebook():
        pass
        break

display(frame.describe())

# %%
# plot([dict(y=res.svm_prob[selects, 0] + 1, name='Prob'),
#       dict(y=res.svm_pred[selects] - y_true[selects] + 2, name='Diff'),
#       dict(y=y_true[selects], name='True')])

# %%

# %%
