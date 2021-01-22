# Source estimation

# %% ----------------------------------------------------------------
import os
import sys

from toolbox.estimate_source import SourceEstimator, get_stuff
from read_epochs import read_all_epochs
from set_mne_freesurfer import set_freesurfer_environ

set_freesurfer_environ()

# Results will be stored here
# Options are: SourceEstimation-norm | SourceEstimation
# Make sure the -norm refers using 'normal' value in 'pick_ori' option in 'estimate_source.py' in toolbox
save_to_folder = os.path.join('MiddleResults', 'SourceEstimation-norm')
if not os.path.isdir(save_to_folder):
    os.mkdir(save_to_folder)
assert(os.path.isdir(save_to_folder))

# %%
# Input parameters
subject_name = 'MEG_S02'
freesurfer_name = 'RSVP_MRI_S02'

subject_name, freesurfer_name = sys.argv[1:3]

# Read epochs and formulate evoked
epochs = read_all_epochs(subject_name)['1']
evoked = epochs.average()
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
srcEst.pre_estimation(src, sol, trans, epochs, epochs.info)

# Source estimation
stc, morph = srcEst.estimate(evoked)
stc_morph = morph.apply(stc)
# stcs, _ = srcEst.estimate(epochs)

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
# Save results to disk
stc.save(os.path.join(save_to_folder, subject_name))
stc_morph.save(os.path.join(save_to_folder, f'{subject_name}-morph'))
overwrite = True
morph.save(os.path.join(save_to_folder, subject_name), overwrite=overwrite)

# %%
print('All Done')

# %%
