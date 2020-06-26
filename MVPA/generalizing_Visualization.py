# %% Importing
# System -----------------------------------------
import os
import sys
import pickle
import threading

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
TMP_FOLDER = os.path.join('.', 'tmp')


RE_COMPUTE = False


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


def fill_matrix(j, y_true, y_pred_generalizing, time_report):
    print(f'Start {j}.')

    reports = [e for e in range(141)]
    for k in range(141):
        y_pred = y_pred_generalizing[:, j, k]
        report = metrics.classification_report(y_true=y_true,
                                               y_pred=y_pred,
                                               output_dict=True)
        reports[k] = fuck_report(report)

        if k % 20 == 0:
            print(f'----{j}: {k} | 141')

    for key in time_report:
        for k in range(141):
            time_report[key][j, k] = reports[k][key]

    print(f'Stop {j}.')


def get_time_report(y_true, y_pred_generalizing):
    report = metrics.classification_report(y_true=y_true,
                                           y_pred=y_pred_generalizing[:, 0, 0],
                                           output_dict=True)
    time_report = fuck_report(report)
    for key in time_report:
        print(key)
        time_report[key] = np.zeros((141, 141))

    threading_pool = []

    for j in range(141):
        t = threading.Thread(target=fill_matrix,
                             kwargs=dict(j=j,
                                         y_true=y_true.copy(),
                                         y_pred_generalizing=y_pred_generalizing.copy(),
                                         time_report=time_report))
        threading_pool.append(t)
        t.start()

    t.join()

    print('.')

    for t in threading_pool:
        t.join()

    print('Done.')
    return time_report


def plot_matrix(score, ax, extent, title):
    ax.imshow(score,
              extent=extent,
              cmap='RdBu_r',
              vmin=0,
              vmax=1,
              origin='lower',
              )
    ax.set_title(title)
    ax.set_aspect('equal')


# %%
time_reports = dict()

for idx in range(1, 11):
    # Loading data ------------------------------------------
    running_name = f'MEG_S{idx:02d}'
    print(running_name)

    # Read pickle
    with open(os.path.join(RESULTS_FOLDER,
                           f'{running_name}_generalizing.pkl'), 'rb') as f:
        mvpa_dict = pickle.load(f)

    # Get y_all, y_pred_generalizing and times
    y_all = mvpa_dict['y_all']
    y_pred_generalizing = mvpa_dict['y_pred_generalizing']
    times = mvpa_dict['times']

    # Generate time report ----------------------------------
    if RE_COMPUTE:
        time_report = get_time_report(y_true=y_all,
                                      y_pred_generalizing=y_pred_generalizing)

        with open(os.path.join(TMP_FOLDER,
                               f'time_report_generalizing_{running_name}.pkl'), 'wb') as f:
            pickle.dump(time_report, f)
    else:
        with open(os.path.join(TMP_FOLDER,
                               f'time_report_generalizing_{running_name}.pkl'), 'rb') as f:
            time_report = pickle.load(f)

    time_reports[running_name] = time_report

    # Plot --------------------------------------------------
    # Prepare axes
    plt.style.use('ggplot')
    fig, axes = plt.subplots(4, 4, figsize=(10, 10), constrained_layout=True)

    # Plot in curve
    _min, _max = min(times), max(times)
    for j, s1 in enumerate(['1.0', '2.0', 'macro avg', 'weighted avg']):
        for k, s2 in enumerate(['precision', 'recall', 'f1-score']):
            key = f'{s1}-{s2}'
            plot_matrix(score=time_report[key],
                        ax=axes[k, j],
                        extent=(_min, _max, _min, _max),
                        title=key)

    key = 'accuracy'
    plot_matrix(score=time_report[key],
                ax=axes[3, 0],
                extent=(_min, _max, _min, _max),
                title=key)

    fig.suptitle(f'{running_name}')

    DRAWER.fig = fig

# %%
print('MEAN')

# Plot --------------------------------------------------
# Prepare axes
plt.style.use('ggplot')
fig, axes = plt.subplots(4, 4, figsize=(10, 10), constrained_layout=True)

mean_time_report = dict()
for key in time_report:
    mean_time_report[key] = np.mean(
        [e[key] for e in time_reports.values()], axis=0)

# Plot in curve
_min, _max = min(times), max(times)
for j, s1 in enumerate(['1.0', '2.0', 'macro avg', 'weighted avg']):
    for k, s2 in enumerate(['precision', 'recall', 'f1-score']):
        key = f'{s1}-{s2}'
        plot_matrix(score=mean_time_report[key],
                    ax=axes[k, j],
                    extent=(_min, _max, _min, _max),
                    title=key)

key = 'accuracy'
plot_matrix(score=mean_time_report[key],
            ax=axes[3, 0],
            extent=(_min, _max, _min, _max),
            title=key)

fig.suptitle(f'MEAN')

DRAWER.fig = fig

# %%
DRAWER.save('generalizing.pdf')

# %%
