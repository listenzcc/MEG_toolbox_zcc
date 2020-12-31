
# %%
import matplotlib
import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from tools.data_manager import DataManager
from tools.epochs_tools import epochs_get_MVPA_data
from tools.figure_toolbox import Drawer

import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)

plt.style.use('tableau-colorblind10')

# %%
drawer = Drawer()
events_ID = ['1', '2', '4']
n_jobs = 48

# %%
name = 'MEG_S02'
for name in [f'MEG_S{j+1:02d}' for j in range(10)]:
    # -----------------------------------------------
    # Init data manager
    dm = DataManager(name)
    # Load epochs
    dm.load_epochs(recompute=False)

    # -----------------------------------------------
    # Separate epochs
    epochs, epochs_2 = dm.leave_one_session_out(includes=[1, 2, 3, 4, 5, 6, 7],
                                                excludes=[0])
    epochs = epochs[events_ID]
    # Xdawn enhancemen
    xdawn = mne.preprocessing.Xdawn(n_components=12)

    # Fit
    xdawn.fit(epochs)

    # Baseline correction
    epochs.apply_baseline((None, 0))
    epochs = epochs[events_ID]

    # %%
    data = np.mean(xdawn.transform(epochs['1']), axis=0)
    data2 = np.mean(xdawn.transform(epochs['2']), axis=0)
    times = epochs.times
    print(data.shape, data2.shape, times.shape)

    # %%
    M = np.max(data)
    m = np.min(data)
    fig, axes = plt.subplots(12, 2, figsize=(12, 24))

    def black(ax):
        # The style will stroke the outline of the head as 'white' color,
        # so we have to paint it as 'dark gray' (like #444444) manually,
        # ! it seems the outline is of and only of the type of Line2D,
        # ! it can be changed, SO BE CAREFUL.
        for e in ax.get_children():
            if type(e) in [matplotlib.lines.Line2D]:
                e.set_color('#444444')
        return ax

    for j in range(12):
        pattern = xdawn.patterns_['1'][j]
        im = mne.viz.plot_topomap(pattern, epochs.info,
                                  axes=axes[j, 0],
                                  show=False)
        black(axes[j, 0])
        axes[j, 0].set_title(j)

        axes[j, 1].plot(times, data[j], label='Target')
        axes[j, 1].plot(times, data2[j], label='Non-Target')
        axes[j, 1].set_ylim([m, M])
        axes[j, 1].legend()

    fig.suptitle(name)
    fig.tight_layout(rect=[0.1, 0.1, 0.9, 0.99])
    drawer.fig = fig
    print()

# %%

drawer.save('xdawn-Count.pdf')
