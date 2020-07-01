# %% Importing
# System
import os
import sys
import time
import pickle

# Computing
import mne
import numpy as np
from scipy.stats import entropy

# Plotting
import tqdm
import matplotlib.pyplot as plt

# Local tools
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))  # noqa
from MEG_worker import MEG_Worker
from figure_tools import Drawer

# Settings
BAND_NAME = 'U07'

COLORS_272 = np.load(os.path.join(os.path.dirname(__file__),
                                  '..', 'tools', 'colors_272.npy'))

DRAWER = Drawer()
SHOW = False

# %%
idx = 3

# Setting -------------------------------------------
running_name = f'MEG_S{idx:02d}'

plt.style.use('ggplot')

# Worker pipeline -----------------------------------
worker = MEG_Worker(running_name=running_name)
worker.pipeline(band_name=BAND_NAME)

# %%
info = worker.clean_epochs.info.copy()
for j, name in enumerate(info['ch_names']):
    info['ch_names'][j] = name.split('-')[0]

fig = mne.viz.plot_sensors(info, show_names=True)

# %%
