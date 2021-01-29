# Visualize epochs

# %%
import os
import sys
import mne
import pandas as pd

from toolbox.figure_utils import FigureCollection
from toolbox.compute_lags import LagsComputer
from toolbox.fix_known_issue import custom_montage
from toolbox.preprocessing import denoise_projs

from read_epochs import read_all_epochs

# %%
subject_name = sys.argv[1]  # 'MEG_S02' for example
# subject_name = 'MEG_S02'
dest_folder = os.path.join('Visualization', 'Evoked_and_TimeShift')
assert(os.path.isdir(dest_folder))
pdf_path = os.path.join(dest_folder, f'{subject_name}.pdf')

# %%
fc = FigureCollection()

epochs_inventory = pd.read_json('inventory-epo.json')
epochs_inventory

assert(subject_name in epochs_inventory.subject.to_list())


def fetch_subject(subject, frame=epochs_inventory):
    return frame.query(f'subject == "{subject}"')

# %%


epochs = read_all_epochs(subject_name)

denoise_projs(epochs)

if subject_name.startswith('EEG'):
    epochs = custom_montage(epochs)

epochs_1 = epochs['1'].copy()
epochs_1.apply_baseline((None, 0))
fc.fig = epochs_1.average().plot_joint(show=False, title=subject_name)

# %%
lags_computer = LagsComputer(epochs_1)
lags_computer.compute_lags()

fc = lags_computer.draw(fc=fc, suptitle=subject_name)

fc.save(pdf_path)

# %%
print('All Done')

# %%
