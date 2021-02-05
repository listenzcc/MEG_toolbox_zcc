# MVPA on temporal generation manner

# %%
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

import mne
from mne.decoding import Scaler
from mne.decoding import Vectorizer
from mne.decoding import SlidingEstimator
from mne.decoding import LinearModel
from mne.decoding import cross_val_multiscore
from mne.decoding import get_coef

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from read_epochs import read_all_epochs
from toolbox.preprocessing import denoise_projs
from toolbox.fix_known_issue import custom_montage
from toolbox.figure_utils import FigureCollection

# %%
subject_name = 'MEG_S02'
# subject_name = sys.argv[1]  # 'MEG_S02' for example

dest_folder = os.path.join('Visualization', 'MVPA_Generation')
if not os.path.isdir(dest_folder):
    os.mkdir(dest_folder)
assert(os.path.isdir(dest_folder))

pdf_path = os.path.join(dest_folder, f'{subject_name}.pdf')
html_path = os.path.join(dest_folder, f'{subject_name}.html')

fc = FigureCollection()
# %%
epochs_list = read_all_epochs(subject_name, use_list=True)
epochs_list = [e[['1', '2']] for e in epochs_list]
ns = [len(e) for e in epochs_list]

# %%
epochs = mne.concatenate_epochs(epochs_list)
denoise_projs(epochs)
if subject_name.startswith('EEG'):
    epochs = custom_montage(epochs)
epochs.crop(0, 1)

times = epochs.times
info = epochs.info
data = epochs.get_data()
label = epochs.events[:, 2]
session = label - label
c = 0
for j, n in enumerate(ns):
    session[c:c+n] = j+1
    c += n

# _data = []
# _label = []
# _session = []
# for j, epochs in enumerate(epochs_list):
#     _data.append(epochs_list[j].get_data())
#     _label.append(epochs_list[j].events[:, 2])
#     _session.append(epochs_list[j].events[:, 2] * 0 + j + 1)

# data = np.concatenate(_data, axis=0)
# label = np.concatenate(_label, axis=0)
# session = np.concatenate(_session, axis=0)


def crash(d, method=np.max):
    '''
    Crash the last two dimensions of the 3d matrix

    args:
    - d: The 3d matrix to be crashed, the shape is (x, y, z)
    - method: The crashing method

    outs:
    - d: The matrix after crashing, the shape is (x,)
    '''
    d = method(d, axis=-1)
    d = method(d, axis=-1)
    return d


df = pd.DataFrame()
df['label'] = label
df['session'] = session
df['max'] = crash(data)
df['min'] = crash(data, np.min)

df

# %%
n_jobs = 48
clf = make_pipeline(
    Scaler(info),
    Vectorizer(),
    StandardScaler(),
    LinearModel(LogisticRegression(solver='liblinear')),
)

time_decoder = SlidingEstimator(
    clf,
    scoring='roc_auc',
    n_jobs=n_jobs,
)

y = df['label'].values.copy()
y[y == 2] = 0

scores = cross_val_multiscore(
    time_decoder,
    X=data,
    y=y,
    groups=df['session'],
    cv=len(df['session'].unique()),
    n_jobs=n_jobs,
)

time_decoder.fit(data, y)

# %%
# Plot

kwargs = dict(
    x=times,
    y=np.mean(scores, axis=0),
    error_y=np.std(scores, axis=0),
    title=f'Sensor Space Decoding (Auc of Roc) of {subject_name}',
)
fig = px.line(**kwargs)
fig.show()

coef = get_coef(time_decoder, 'patterns_', inverse_transform=True)[0]
evoked_time_gen = mne.EvokedArray(coef, info, tmin=times[0])

joint_kwargs = dict(
    ts_args=dict(time_unit='s'),
    topomap_args=dict(time_unit='s'),
    times=np.arange(0.2, 0.6, 0.1),
    show=False,
)

fc.fig = evoked_time_gen.plot_joint(
    title='patterns',
    **joint_kwargs,
)

fc.fig = epochs['1'].average().plot_joint(
    title='Evoked',
    **joint_kwargs,
)


def fig_save(fig, path=html_path):
    html = fig.to_html()
    with open(path, 'w') as f:
        f.write(html)


fig_save(fig)
fc.save(pdf_path)

# %%
print('All Done')

# %%
# Plot sessions and their max and min values


# def plot_sessions(df):
#     fig1 = px.scatter(data_frame=df,
#                       y=['label', 'session'],
#                       title='Label & Session')
#     fig1.show()

#     kwargs = dict(
#         histnorm='percent'
#     )
#     trace = [
#         go.Histogram(x=df.query('label == 0')['max'], name='0 max', **kwargs),
#         go.Histogram(x=df.query('label == 1')['max'], name='1 max', **kwargs),
#         go.Histogram(x=df.query('label == 0')['min'], name='0 min', **kwargs),
#         go.Histogram(x=df.query('label == 1')['min'], name='1 min', **kwargs),
#     ]
#     fig2 = go.Figure(data=trace, layout=go.Layout(title='Histogram'))
#     fig2.show()

#     trace = []
#     for s in df['session'].unique():
#         _df = df.query(f'session == {s}')
#         for label in [0, 1]:
#             for method in ['max', 'min']:
#                 trace.append(go.Box(
#                     y=_df.query(f'label == {label}')[method],
#                     name=f'{chr(ord("A")+s-1)} {label} {method}',
#                 ))

#     layout = go.Layout(title='Values in Sessions in BoxGraph')
#     fig3 = go.Figure(data=trace, layout=layout)
#     fig3.show()

# plot_sessions(df)

# %%
