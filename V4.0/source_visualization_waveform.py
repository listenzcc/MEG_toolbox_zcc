#!/usr/bin/env python
# coding: utf-8
# Visualize Source Activity in Wave Form

# %%
import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from toolbox.estimate_source import get_stuff

import plotly.offline as py
import plotly.graph_objs as go

# %%
# Setup Freesurfer Environ Variables

if not '__file__' in dir():
    __file__ = os.path.join(os.path.abspath(''), '__fake__file__')


def set_freesurfer_environ():
    freesurfer_folder = os.path.join(os.path.dirname(__file__),
                                     '_link_freesurfer')
    mne.utils.set_config('SUBJECTS_DIR',
                         os.path.join(freesurfer_folder, 'subjects'))


set_freesurfer_environ()


# %%
# Read Stc and Extract Time Series
# Settings
subject_name = 'MEG_S02'
freesurfer_name = 'RSVP_MRI_S02'
freesurfer_template = 'fsaverage'
stc_folder = os.path.join('MiddleResults', 'SourceEstimation')
subject_folder = os.path.join(os.path.dirname(__file__),
                              '_link_preprocessed', subject_name)

# Read objs from disk


def stuff_path(name, freesurfer_name=freesurfer_name, folder=subject_folder):
    return os.path.join(folder, f'{freesurfer_name}-{name}.fif')


src = get_stuff('src', stuff_path('src'))

# Read stc from disk
stc = mne.read_source_estimate(os.path.join(stc_folder, subject_name))

# Extract label_ts in subject's space
labels = mne.read_labels_from_annot(freesurfer_name, 'aparc')
label_ts = mne.extract_label_time_course(
    stc, labels, src, mode='mean_flip', return_generator=False)

label_ts

# %%
# Display using plotly

data = []
for j in range(68):
    data.append(go.Scatter(y=label_ts[j],
                           name=labels[j].name))

layout = go.Layout(
    title=subject_name
)

fig = go.Figure(data=data, layout=layout)
fig.show()

# %%
