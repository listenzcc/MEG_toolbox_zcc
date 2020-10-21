# %%
import mne
from tools.data_manager import DataManager

# %%
parameters_meg = dict(picks='mag',
                      stim_channel='UPPT001',
                      l_freq=0.1,
                      h_freq=15.0,
                      tmin=-0.2,
                      tmax=1.2,
                      decim=12,
                      detrend=1,
                      reject=dict(
                          mag=4000e-15,  # 4000 fT
                      ),
                      baseline=None)

parameters_meg = dict(picks='eeg',
                      stim_channel='from_annotations',
                      l_freq=0.1,
                      h_freq=15.0,
                      tmin=-0.2,
                      tmax=1.2,
                      decim=10,
                      detrend=1,
                      reject=dict(
                          eeg=150e-6,  # 150 $mu$V
                      ),
                      baseline=None)

# %%

name = 'EEG_S02'
loader = DataManager(name)
loader.load_raws()
print('Done')

# %%
raw = loader.raws[0]
raw
# %%
mne.find_events(raw)
# %%
mne.events_from_annotations(raw)
# %%
