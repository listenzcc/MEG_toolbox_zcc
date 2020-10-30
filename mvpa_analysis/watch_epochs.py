# File: watch_epochs.py
# Aim: Plot epochs with Xdawn enhancement
# %%
import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from tools.data_manager import DataManager
from tools.epochs_tools import epochs_get_MVPA_data
from tools.figure_toolbox import Drawer

# %%
drawer = Drawer()
events_ID = ['1', '2', '4']
n_components = 6
n_jobs = 48

# %%
for name in ['MEG_S02', 'EEG_S02']:
    # -----------------------------------------------
    # Init data manager
    dm = DataManager(name)
    # Load epochs
    dm.load_epochs(recompute=False)

    # -----------------------------------------------
    # Separate epochs
    epochs_1, epochs_2 = dm.leave_one_session_out(includes=[1, 3, 5],
                                                  excludes=[2, 4, 6])

    # Xdawn enhancemen
    xdawn = mne.preprocessing.Xdawn(n_components=n_components)

    # Fit
    xdawn.fit(epochs_1)

    # Baseline correction
    epochs_1.apply_baseline((None, 0))
    epochs_2.apply_baseline((None, 0))
    epochs_1 = epochs_1[events_ID]
    epochs_2 = epochs_2[events_ID]

    # Apply using xdawn
    epochs_1_xdawn = xdawn.apply(epochs_1)
    epochs_2_xdawn = xdawn.apply(epochs_2)

    # -----------------------------------------------
    # Plot waveform
    for e in events_ID:
        # Draw raw evoked
        title = 'Raw-{}'
        drawer.fig = epochs_1[e].average().plot_joint(
            title=title.format(e), show=False)
        drawer.fig = epochs_2[e].average().plot_joint(
            title=title.format(e), show=False)

        # Draw xdawn evoked
        title = 'Xdawn-{}'
        drawer.fig = epochs_1_xdawn['1'][e].average().plot_joint(
            title=title.format(e), show=False)
        drawer.fig = epochs_2_xdawn['1'][e].average().plot_joint(
            title=title.format(e), show=False)

    # -----------------------------------------------
    # Transfrom using xdawn
    data_1 = xdawn.transform(epochs_1)[:, :n_components]
    data_2 = xdawn.transform(epochs_2)[:, :n_components]

    # Get data
    X, y, z = epochs_get_MVPA_data([epochs_1, epochs_2])
    Xd = np.concatenate([data_1, data_2], axis=0)
    X.shape, Xd.shape, y.shape, z.shape

    # -----------------------------------------------
    # Calculate TSNE manifold
    vectorizer = mne.decoding.Vectorizer()
    tsne = manifold.TSNE(n_components=2, n_jobs=n_jobs)
    X2 = tsne.fit_transform(vectorizer.fit_transform(Xd))
    X2.shape

    # -----------------------------------------------
    # Plot in TSNE manifold
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    yy = y + z * 10
    for j in np.unique(yy):
        print(j)
        ax.scatter(X2[yy == j, 0], X2[yy == j, 1], alpha=0.5, label=j)
    ax.legend()
    drawer.fig = fig

    # -----------------------------------------------
    drawer.save(f'Epochs_{name}.pdf')
    drawer.clear()

# %%
