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
# for idx in range(1, 11):

idx = 3
running_name = f'MEG_S{idx:02d}'
band_name = 'U07'

worker = MEG_Worker(running_name=running_name)
worker.pipeline(band_name=band_name)

# %%
viz = Visualizer(title_prefix=running_name)

viz.load_epochs(worker.denoise_epochs)
viz.plot_joint('1')
viz.plot_joint('2')
viz.plot_joint('3')

viz.load_epochs(worker.clean_epochs)
viz.plot_joint('1', title='clean-1')

viz.plot_lags(worker.paired_lags_timelines)

viz.save_figs(f'{running_name}.pdf')

# %%
solver = Inverse_Solver(running_name=running_name)
solver.pipeline(obj=worker.clean_epochs.average(),
                epochs=worker.clean_epochs,
                raw_info=worker.raw.info)

# %%
solver.stc

# %%
solver.stc_fsaverage

# %%
mne.viz.set_3d_backend('pyvista')

alldata = sorted(solver.stc_fsaverage.data.ravel(), reverse=True)
n = len(alldata)
surfer_kwargs = dict(hemi='both',
                     clim=dict(kind='value',
                               lims=[alldata[int(n * r)] for r in [0.05, 0.01, 0]]),
                     views='lateral',
                     initial_time=0.4,
                     time_unit='s',
                     size=(800, 800),
                     smoothing_steps=10)

# This can not be operated using VS code
brain = solver.stc_fsaverage.plot(**surfer_kwargs)


# %%
input('Press enter to escape.')
