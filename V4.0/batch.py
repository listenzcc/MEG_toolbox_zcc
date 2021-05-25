# On-time batch workload runner
# ! Only for temporal use
# ! Do not trust any word in this script

import os
import subprocess

# Python script file name
# script = 'visualize_epochs.py'
script = 'source_estimation.py'
# script = 'MVPA_generation.py'
# script = 'MVPA_source.py'

# The latest segment of cmd,
# use background if use '&',
# not use background if use ' '.
# background_ext = ' & '
background_ext = '   '

for j in [5, 6, 7, 8, 9]:  # range(10):
    subject_id = j+1
    subject_name = f'MEG_S{subject_id:02d} RSVP_MRI_S{subject_id:02d}'
    # subject_name = f'MEG_S{subject_id:02d}'
    cmd = f'python {script} {subject_name} {background_ext}'
    print(cmd)

    # Operate [cmd] one after another
    os.system(cmd)

    # Operate [cmd] as background job
    # subprocess.call(cmd, shell=True)


# subprocess.call('git --version', shell=True)
