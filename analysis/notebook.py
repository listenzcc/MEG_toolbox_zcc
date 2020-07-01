# %% Importing
# System
import os
import sys

# Computing
import mne
import numpy as np

# Plotting
import matplotlib.pyplot as plt
from surfer import Brain

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
# epochs = worker.denoise_epochs['3']
epochs = worker.clean_epochs
solver = Inverse_Solver(running_name=running_name)
solver.pipeline(epochs=epochs,
                raw_info=worker.raw.info)

# %%
stc, stc_fsaverage = solver.estimate(obj=epochs.average())

# %%
labels = mne.read_labels_from_annot('fsaverage', 'aparc', 'lh')

brain = Brain('fsaverage', 'lh', 'inflated',
              cortex='low_contrast', background='white', size=(800, 600))
brain.add_annotation('aparc')
# aud_label = [label for label in labels if label.name == 'L_A1_ROI-lh'][0]
# brain.add_label(aud_label, borders=False)

# %%
# mne.viz.set_3d_backend('pyvista')

# alldata = sorted(solver.stc_fsaverage.data.ravel(), reverse=True)
# n = len(alldata)
# surfer_kwargs = dict(hemi='both',
#                      clim=dict(kind='value',
#                                lims=[alldata[int(n * r)] for r in [0.05, 0.01, 0]]),
#                      views='lateral',
#                      initial_time=0.4,
#                      time_unit='s',
#                      size=(800, 800),
#                      smoothing_steps=10)

# # This can not be operated using VS code
# brain = solver.stc_fsaverage.plot(**surfer_kwargs)

# %%
input('Press enter to escape.')
