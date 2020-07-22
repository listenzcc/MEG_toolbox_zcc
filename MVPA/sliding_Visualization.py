# %% Importing
# System -----------------------------------------
import os
import sys
import pickle

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
RESULTS_FOLDER = os.path.join('.', 'results')


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


def get_time_report(y_true, y_pred_sliding):
    report = metrics.classification_report(y_true=y_true,
                                           y_pred=y_pred_sliding[:, 0],
                                           output_dict=True)
    time_report = fuck_report(report)
    for key in time_report:
        print(key)
        time_report[key] = []

    for j, y_pred in enumerate(y_pred_sliding.transpose()):
        report = metrics.classification_report(y_true=y_true,
                                               y_pred=y_pred,
                                               output_dict=True)
        report = fuck_report(report)

        for key in time_report:
            time_report[key].append(report[key])

    return time_report


# %%
for idx in range(1, 11):
    # Loading data ------------------------------------------
    running_name = f'MEG_S{idx:02d}'
    print(running_name)

    # Read pickle
    with open(os.path.join(RESULTS_FOLDER,
                           f'{running_name}_sliding.pkl'), 'rb') as f:
        mvpa_dict = pickle.load(f)

    # Get y_all, y_pred_sliding and times
    y_all = mvpa_dict['y_all']
    y_pred_sliding = mvpa_dict['y_pred_sliding']
    times = mvpa_dict['times']

    # Generate time report ----------------------------------
    time_report = get_time_report(y_true=y_all,
                                  y_pred_sliding=y_pred_sliding)

    if idx == 1:
        mean_time_report = time_report
        mean_counting = 1
    else:
        for key in time_report:
            mean_time_report[key] += time_report[key]
        mean_counting += 1

    # Plot --------------------------------------------------
    # Prepare axes
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    axes = np.ravel(axes)

    # Set groups
    groups = ['1.0', '2.0', 'macro', 'weighted', 'accuracy']

    # Plot in curve
    for j, prefix in enumerate(groups):
        for key in [e for e in time_report if e.startswith(prefix)]:
            axes[j].plot(times, time_report[key], label=key)

    # Add legends
    for j in range(5):
        axes[j].set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        axes[j].set_ylim([0.0, 1.0])
        axes[j].legend(loc='lower right', bbox_to_anchor=(1, 0))

    fig.suptitle(f'{running_name}')

    DRAWER.fig = fig

for key in mean_time_report:
    mean_time_report[key] = [e / mean_counting for e in mean_time_report[key]]

# Plot --------------------------------------------------
# Prepare axes
plt.style.use('ggplot')
fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
axes = np.ravel(axes)

# Set groups
groups = ['1.0', '2.0', 'macro', 'weighted', 'accuracy']

# Plot in curve
for j, prefix in enumerate(groups):
    for key in [e for e in time_report if e.startswith(prefix)]:
        axes[j].plot(times, time_report[key], label=key)

# Add legends
for j in range(5):
    axes[j].set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    axes[j].set_ylim([0.0, 1.0])
    axes[j].legend(loc='lower right', bbox_to_anchor=(1, 0))

fig.suptitle(f'MEAN')

DRAWER.fig = fig


# %%
DRAWER.save('sliding.pdf')

# %%
