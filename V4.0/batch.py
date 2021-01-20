# On-time batch workload runner
# ! Only for temporal use
# ! Do not trust any word in this script

import os
import subprocess

# Python script file name
script = 'visualize_epochs.py'
script = 'source_estimation.py'

# The latest segment of cmd,
# use background if use '&',
# not use background if use ' '.
background_ext = ' & '
background_ext = '   '

for j in range(10):
    subject_id = j+1
    subject_name = f'MEG_S{subject_id:02d}'
    cmd = f'python {script} {subject_name} {background_ext}'
    print(cmd)
    subprocess.call(cmd, shell=True)


# subprocess.call('git --version', shell=True)
