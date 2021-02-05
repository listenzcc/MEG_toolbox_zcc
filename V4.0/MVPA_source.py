# MVPA on source estimation

# %%
import plotly.figure_factory as ff
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

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
from toolbox.estimate_source import SourceEstimator, get_stuff

from set_mne_freesurfer import set_freesurfer_environ

set_freesurfer_environ()

# %%
subject_name = 'MEG_S02'
freesurfer_name = 'RSVP_MRI_S02'
# subject_name, freesurfer_name = sys.argv[1:3]

# %%
# Prepare epochs
epochs_list = read_all_epochs(subject_name, use_list=True)
epochs_list = [e[['1', '2']] for e in epochs_list]
ns = [len(e) for e in epochs_list]

dest_folder = os.path.join('Visualization', 'MVPA_Source')
if not os.path.isdir(dest_folder):
    os.mkdir(dest_folder)
assert(os.path.isdir(dest_folder))

html_path = dict(
    plot=os.path.join(dest_folder, f'{subject_name}-plot.html'),
    table=os.path.join(dest_folder, f'{subject_name}-table.html')
)

# %%
epochs = mne.concatenate_epochs(epochs_list)
denoise_projs(epochs)

times = epochs.times
info = epochs.info
data = epochs.get_data()
label = epochs.events[:, 2]
session = label - label
c = 0
for j, n in enumerate(ns):
    session[c:c+n] = j+1
    c += n


epochs_df = pd.DataFrame()
epochs_df['label'] = label
epochs_df['session'] = session

epochs_df

# %%
# Source estimation

# Prepare parameters
subject_folder = os.path.join(os.path.dirname(__file__),
                              '_link_preprocessed', subject_name)


def stuff_path(name, freesurfer_name=freesurfer_name, folder=subject_folder):
    ''' Get stuff for source estimation '''
    return os.path.join(folder, f'{freesurfer_name}-{name}.fif')


src = get_stuff('src', stuff_path('src'))
bem = get_stuff('bem', stuff_path('bem'))
sol = get_stuff('bem-sol', stuff_path('bem-sol'))
trans = get_stuff('trans', stuff_path('trans'))

# Init estimator
srcEst = SourceEstimator(freesurfer_name)
srcEst.pre_estimation(src, sol, trans, epochs, epochs.info)

stcs, morph = srcEst.estimate(epochs)
stcs = [morph.apply(s) for s in stcs]
stcs

# %%
annot_name = freesurfer_name
annot_name = 'fsaverage'
labels_name = 'aparc'
labels_name = 'PALS_B12_Visuotopic'

src = mne.read_source_spaces(os.path.join(
    os.environ['SUBJECTS_DIR'], 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))

labels = mne.read_labels_from_annot(annot_name, labels_name)
labels = [e for e in labels if not 'unknown' in e.name]
labels = [e for e in labels if not '?' in e.name]
label_ts = mne.extract_label_time_course(
    stcs, labels, src, mode='mean_flip', return_generator=False)
label_ts = np.array(label_ts)
label_ts


# %%
n_jobs = 48
clf = make_pipeline(
    # Scaler(info),
    Vectorizer(),
    StandardScaler(),
    LinearModel(LogisticRegression(solver='liblinear')),
)

time_decoder = SlidingEstimator(
    clf,
    scoring='roc_auc',
    n_jobs=n_jobs,
)

y = epochs_df['label'].values.copy()
y[y == 2] = 0

scores = cross_val_multiscore(
    time_decoder,
    X=label_ts,
    y=y,
    groups=epochs_df['session'],
    cv=len(epochs_df['session'].unique()),
    n_jobs=n_jobs,
)

time_decoder.fit(label_ts, y)
coef = get_coef(time_decoder, 'patterns_', inverse_transform=True)

# %%
# Plot


def fig_save(fig, path):
    html = fig.to_html()
    with open(path, 'w') as f:
        f.write(html)


p = np.max(np.abs(coef), axis=1)
o = np.argsort(p)[::-1]
p = p[o]
lp = [labels[j].name for j in o]

df = pd.DataFrame(lp)
df.columns = ['label']
df['peak_value'] = p

# kwargs = dict(
#     # mode='lines',
#     showlegend=False
# )

fig = make_subplots(
    rows=4,
    cols=1,
    subplot_titles=['Ts', 'Roc', 'Coef', 'Coef (Peak)']
)

# Ts
ts = np.mean(label_ts[epochs_df['label'] == 1], axis=0)
data = []
for s in ts:
    data.append(go.Scatter(x=times, y=s, showlegend=False))
for d in data:
    fig.add_trace(d, row=1, col=1)

# Roc
m = np.mean(scores, axis=0)
s = np.std(scores, axis=0)
data = [
    go.Scatter(x=times, y=m, showlegend=False),
    go.Scatter(x=times, y=m+s, showlegend=False),
    go.Scatter(x=times, y=m-s, showlegend=False),
]
for d in data:
    fig.add_trace(d, row=2, col=1)

# Coef
data = []
for j, c in enumerate(coef):
    data.append(go.Scatter(x=times, y=c, name=labels[j].name, mode='lines'))
for d in data:
    fig.add_trace(d, row=3, col=1)

# Coef (Peak)
data = go.Bar(x=lp, y=p, showlegend=False)
fig.add_trace(data, row=4, col=1)

# Show
fig.update_layout(dict(
    height=800
))

fig.show()

# fig_save(fig, html_path['plot'])

df

# fig = ff.create_table(df)
# fig.show()
# fig_save(fig, html_path['table'])

# %%
print('All Done.')

# %%
