# %% Importing
# System -----------------------------------------
import os
import sys
import json
import pickle
from pprint import pprint
from collections import defaultdict

# Computing --------------------------------------
# numpy
import numpy as np

# sklearn
from sklearn import metrics

# Plotting --------------------------------------
import matplotlib.pyplot as plt

# Local Settings --------------------------------------
# Tools
sys.path.append(os.path.join('..', 'tools'))  # noqa
from figure_tools import Drawer

# Drawer
DRAWER = Drawer()

# Init results_folder
RESULTS_FOLDER = os.path.join('.', 'results_baseline')
RESULTS_NAME = os.path.join('.', 'results_baseline.json')


def fuck_report(report):
    # I don't know why report has to be 2-layers dictionary,
    # this method is to make it human useable.
    fucked_report = dict()
    for key1 in report:
        if isinstance(report[key1], dict):
            for key2 in report[key1]:
                fucked_report[f'{key1}-{key2}'] = report[key1][key2]
        else:
            fucked_report[key1] = report[key1]

    keys = [e for e in fucked_report.keys()]
    for key in keys:
        if key.endswith('support'):
            fucked_report.pop(key)

    return fucked_report


def get_time_report(y_true, y_pred_time):
    report = metrics.classification_report(y_true=y_true,
                                           y_pred=y_pred_time[:, 0],
                                           output_dict=True)
    time_report = fuck_report(report)
    for key in time_report:
        print(key)
        time_report[key] = []

    for j, y_pred in enumerate(y_pred_time.transpose()):
        report = metrics.classification_report(y_true=y_true,
                                               y_pred=y_pred,
                                               output_dict=True)
        report = fuck_report(report)

        for key in time_report:
            time_report[key].append(report[key])

    return time_report


# %%

crops_table = dict(a=(0.2, 0.4),
                   b=(0.4, 0.6),
                   c=(0.6, 0.8),
                   d=(0.2, 0.8),
                   e=(0.0, 1.0))


crop_summary = defaultdict(dict)

for idx in range(1, 11):
    # Loading data ------------------------------------------
    running_name = f'MEG_S{idx:02d}'
    print(running_name)

    # Read pickle
    with open(os.path.join(RESULTS_FOLDER,
                           f'{running_name}_segment.pkl'), 'rb') as f:
        mvpa_dict = pickle.load(f)

    for crop_name in ['a', 'b', 'c', 'd', 'e']:
        # Get y_all, y_pred
        y_all = mvpa_dict['y_all']
        y_pred = mvpa_dict[f'{crop_name}_y_pred']

        # Generate time report ----------------------------------
        report = metrics.classification_report(y_true=y_all,
                                               y_pred=y_pred,
                                               output_dict=True)
        report = fuck_report(report)
        # pprint(report)

        # Record
        for key in report:
            if key in crop_summary[crop_name]:
                crop_summary[crop_name][key].append(report[key])
            else:
                crop_summary[crop_name][key] = [report[key]]


# %%
plt.style.use('ggplot')

fig, axes = plt.subplots(3, 2, figsize=(10, 15))
axes = np.ravel(axes)

for j, crop_name in enumerate(crop_summary):
    ax = axes[j]

    summary = crop_summary[crop_name]

    ax.bar(x=[e for e in summary],
           height=[np.mean(e) for e in summary.values()],
           yerr=[np.std(e) / 2 for e in summary.values()])

    for label in ax.get_xticklabels():
        label.set_ha('left')
        label.set_rotation(-45)

    ax.set_ylim((0.8, 1.0))
    ax.set_title(f'{crop_name} - Metrics')

fig.tight_layout()
DRAWER.fig = fig

# %%
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
axes = np.ravel(axes)

for j, crop_name in enumerate(crop_summary):
    if j >= 4:
        continue
    range_str = crops_table[crop_name]

    ax = axes[j]
    for score_name in ['1.0-recall', '1.0-f1-score', 'accuracy']:
        scores = crop_summary[crop_name][score_name]
        ax.bar(x=score_name.split('-', 1)[-1].title(),
               height=np.mean(scores),
               yerr=np.std(scores) / 2)
        print(score_name, range_str, np.mean(scores))

    ax.set_title(f'Range: {range_str}')
    ax.set_ylim((0.8, 1.0))

fig.tight_layout()
DRAWER.fig = fig

# with open(RESULTS_NAME, 'w') as f:
# json.dump(crop_summary, f)

# %%
# DRAWER.save('segment.pdf')

# %%
