# Source estimation

# %% ----------------------------------------------------------------
import os
import sys
import mne

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from toolbox.estimate_source import SourceEstimator, get_stuff
from read_epochs import read_all_epochs
from set_mne_freesurfer import set_freesurfer_environ

set_freesurfer_environ()

# Results will be stored here
save_to_folder = os.path.join('MiddleResults', 'SourceEstimation')
if not os.path.isdir(save_to_folder):
    os.mkdir(save_to_folder)
assert(os.path.isdir(save_to_folder))

# If overwrite existing -stc.fif file
# If False, it will stop on error when existing files are found
overwrite = True

# %%
# Input parameters
subject_name = 'MEG_S02'
freesurfer_name = 'RSVP_MRI_S02'

subject_name, freesurfer_name = sys.argv[1:3]

# Read epochs and formulate evoked
epochs = read_all_epochs(subject_name)
epochs = epochs['1', '2']
evoked = epochs['1'].average()
epochs, evoked

# Prepare parameters
subject_folder = os.path.join(os.path.dirname(__file__),
                              '_link_preprocessed', subject_name)

# Get stuff for source estimation


def stuff_path(name, freesurfer_name=freesurfer_name, folder=subject_folder):
    return os.path.join(folder, f'{freesurfer_name}-{name}.fif')


src = get_stuff('src', stuff_path('src'))
bem = get_stuff('bem', stuff_path('bem'))
sol = get_stuff('bem-sol', stuff_path('bem-sol'))
trans = get_stuff('trans', stuff_path('trans'))

# Init estimator
srcEst = SourceEstimator(freesurfer_name)
srcEst.pre_estimation(src, sol, trans, epochs['1'], epochs.info)

# Source estimation
stc, morph = srcEst.estimate(evoked)
stc_morph = morph.apply(stc)
stcs, morph = srcEst.estimate(epochs)
stcs_morph = [morph.apply(e) for e in stcs]

# Finally, following objects have been computed
output = dict(
    stc='Source activity of evoked in individual space',
    stcs='Source activity of epochs in individual space',
    morph='Morph method from individual to "fsaverage"',
    stc_morph='Source activity of evoked in "fsaverage" space',
)
for k in output:
    print(k, output[k])

# %%
# Extract time course of labels
# labels are specified by [labels_name]
labels_name = 'PALS_B12_Visuotopic'

# Get [src] of ioc-5 spacing in 'fsaverage' subject
src = mne.read_source_spaces(os.path.join(
    os.environ['SUBJECTS_DIR'], 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))

# Get [labels] and remove *missing and wrong* labels
labels = mne.read_labels_from_annot('fsaverage', labels_name)
labels = [e for e in labels if not 'unknown' in e.name]
labels = [e for e in labels if not '?' in e.name]

# Extract time course into [label_ts]
label_ts = mne.extract_label_time_course(
    stcs_morph, labels, src, mode='mean_flip', return_generator=False)

# Convert [label_ts] into 3-D numpy array
# shape is [n_epochs x n_labels x n_times]
label_ts = np.array(label_ts)
label_ts

# %%
# Save [label_ts] into DataFrame
# The operation speed is about 400 it/s
# [df] has columns of
# -@ts: Time course
# -@label: Label object of "mne.read_labels_from_annot('fsaverage', labels_name)"
# -@event: Event label
# -@epochs: Id of epochs

event_ids = epochs.events[:, 2]

df_list = []
for j, (mat, evt) in tqdm(enumerate(zip(label_ts, event_ids))):
    d = pd.DataFrame()
    d['ts'] = [e for e in mat]
    d['label'] = labels
    d['event'] = evt
    d['epochs'] = j
    df_list.append(d)

df = pd.concat(df_list, axis=0)
df.index = range(len(df))
df

# %%
# Save results to disk
stc.save(os.path.join(save_to_folder, subject_name))
stc_morph.save(os.path.join(save_to_folder, f'{subject_name}-morph'))
morph.save(os.path.join(save_to_folder, subject_name), overwrite=overwrite)
df.to_json(os.path.join(save_to_folder,
                        f'{subject_name}-{labels_name}-df.json'))

# %%
print('All Done')

# %%
