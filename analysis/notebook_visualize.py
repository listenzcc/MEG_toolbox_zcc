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
from visualizer import Visualizer
from inverse_solver import Inverse_Solver

# %%
# for idx in range(1, 11):

idx = 3
running_name = f'MEG_S{idx:02d}'
band_name = 'U07'

worker = MEG_Worker(running_name=running_name)
worker.pipeline(band_name=band_name)

# %%
viz = Visualizer()
viz.load_epochs(worker.clean_epochs.copy())

# %%
viz.plot_range('1', (0.2, 0.4))

# %%
help(mne.viz.plot_topomap)

# %%

# %%
time_range = (0.2, 0.4)
evoked = worker.clean_epochs['1'].average()

# Get info, data and ch_names
info = evoked.info.copy()
data = evoked.data.copy()
ch_names = info['ch_names'].copy()

# Fine ch_names
for j, name in enumerate(ch_names):
    ch_names[j] = name.split('-')[0]

# Reduce data by time_range
times = evoked.times.copy()
assert(all([time_range[0] > times[0],
            time_range[1] < times[-1]]))
select = np.where((times < time_range[1]) & (times > time_range[0]))[0]
mean_data = np.mean(data[:, select], axis=-1)
print('-' * 80)
print(mean_data.shape)
print('')


# %%
fig = mne.viz.plot_topomap(mean_data, info, names=ch_names, show_names=True)

x = ch_names
y = mean_data
plotly.offline.iplot({
    "data": [go.Scatter(x=x, y=y)],
    "layout": go.Layout(title='No title')
})

# %%


# %%
