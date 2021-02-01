# Set up Freesurfer Environ for MNE

import os
import mne


def set_freesurfer_environ():
    freesurfer_folder = os.path.join(os.path.dirname(__file__),
                                     '_link_freesurfer')
    mne.utils.set_config('SUBJECTS_DIR',
                         os.path.join(freesurfer_folder, 'subjects'))
