# %% Importing
# System
import os
import sys

# Computing
import mne
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)  # noqa

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))  # noqa
from MEG_worker import MEG_Worker

TIME_RANGE = (0.2, 0.4)

# %%
epochs_dict = dict()

for idx in range(1, 11):
    running_name = f'MEG_S{idx:02d}'
    band_name = 'U07'

    worker = MEG_Worker(running_name=running_name)
    worker.pipeline(band_name=band_name)

    epochs_dict[running_name] = worker.clean_epochs['1']

# %%
for running_name in epochs_dict:
    # --------------------------------------------------------
    epochs = epochs_dict[running_name]['1']
    evoked = epochs.average()

    # Get info, data and ch_names
    info = evoked.info.copy()
    data = evoked.data.copy()
    ch_names = info['ch_names'].copy()

    # Fine ch_names
    for j, name in enumerate(ch_names):
        ch_names[j] = name.split('-')[0]

    # Reduce data by time_range
    times = evoked.times.copy()
    assert(all([TIME_RANGE[0] > times[0],
                TIME_RANGE[1] < times[-1]]))
    select = np.where((times < TIME_RANGE[1]) & (times > TIME_RANGE[0]))[0]
    mean_data = np.mean(data[:, select], axis=-1)

    # -----------------------------------------------------------
    fig = mne.viz.plot_topomap(mean_data,
                               info,
                               names=ch_names,
                               show_names=True)

    x = ch_names
    y = mean_data
    plotly.offline.iplot({
        "data": [go.Scatter(x=x, y=y)],
        "layout": go.Layout(title=running_name)
    })


# %%
fig = mne.viz.plot_topomap(mean_data,
                           info,
                           names=ch_names,
                           show_names=True)


# %%
help(mne.viz.plot_topomap)

# %%
