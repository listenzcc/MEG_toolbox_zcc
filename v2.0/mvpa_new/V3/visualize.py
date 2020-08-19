# %%
import os
import mne
import matplotlib.pyplot as plt

data_folder = '../MVPA_data_xdawn_v3'
raw_data_folder = '../MVPA_data_xdawn_raw_v3'

uid = 'MEG_S03-0'

# %%
epochs = mne.read_epochs(os.path.join(data_folder, f'{uid}-train-epo.fif'))
raw_epochs = mne.read_epochs(os.path.join(
    raw_data_folder, f'{uid}-train-epo.fif'))

epochs.apply_baseline((None, 0))
raw_epochs.apply_baseline((None, 0))
display(epochs, raw_epochs)

# %%
plt.style.use('ggplot')
times = {'1': [0.2, 0.3, 0.4, 0.5, 0.6],
         '3': [0]}
for event in ['1', '3']:
    epochs[event].average().plot_joint(times=times[event],
                                       title=f'Denoise')
    raw_epochs[event].average().plot_joint(times=times[event],
                                           title=f'Raw')

print('-')

# %%
