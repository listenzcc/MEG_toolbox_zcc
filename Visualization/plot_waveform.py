# %% Importing
# System
import os
import sys

# Computing
import mne
import numpy as np

# Plotting
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))  # noqa
from MEG_worker import MEG_Worker
from visualizer import Visualizer
from inverse_solver import Inverse_Solver

# %%
for idx in range(1, 11):
    # Setting -------------------------------------------
    running_name = f'MEG_S{idx:02d}'
    band_name = 'U07'

    # Worker pipeline -----------------------------------
    worker = MEG_Worker(running_name=running_name)
    worker.pipeline(band_name=band_name)

    # Viz plotting --------------------------------------
    # Init
    viz = Visualizer(title_prefix=running_name)

    # Plotting evoked
    viz.load_epochs(worker.denoise_epochs)
    viz.plot_joint('1')
    viz.plot_joint('2')
    viz.plot_joint('3')

    # Plotting evoked of clean
    viz.load_epochs(worker.clean_epochs)
    viz.plot_joint('1', title='clean-1')

    # Plotting lags
    viz.plot_lags(worker.paired_lags_timelines)

    # Save figures into pdf
    viz.save_figs(f'{running_name}.pdf')
