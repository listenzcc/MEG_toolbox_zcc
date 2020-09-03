# %%
import mne
import numpy as np
import matplotlib.pyplot as plt
from tools import Drawer
from data_manager import get_objects

# %%
raws = get_objects.get_raws('MEG_S02')
raw = get_objects.concatenate_raws(raws)

# %%
# epochs = get_objects.get_epochs(raws[3])
epochs = get_objects.get_epochs(raw)
print(epochs.info)
epochs

# %%
freqs = np.logspace(0.5, 1.5, 20)
freqs = np.array([3, 5, 7, 10, 15, 20, 30])
n_cycles = np.ceil(freqs / 2).astype(np.int)
evoked = epochs['1'].average()

for event in ['1', '2', '3']:
    print(event)
    power, itc = mne.time_frequency.tfr_morlet(epochs[event],
                                               freqs,
                                               n_cycles,
                                               use_fft=False,
                                               return_itc=True,
                                               decim=1,
                                               n_jobs=48,
                                               picks='mag',
                                               zero_mean=True,
                                               average=True,
                                               output='power',
                                               verbose=None)

    drawer = Drawer()
    for j, freq in enumerate(freqs):
        _power = power.data[:, j, :]
        evoked.data = _power
        drawer.fig = evoked.plot_joint(title=freq, show=False)

    drawer.save(f'event_{event}.pdf')

# %%
epochs['1'].average().plot_joint()
epochs['2'].average().plot_joint()
print()

# %%
evoked1 = epochs['1'].average()
evoked2 = epochs['2'].average()
for e in [evoked1, evoked2]:
    e.crop(tmin=0, tmax=1)

# Compute filter
D = evoked2.data
B = np.zeros((10, D.shape[1]))
for j in range(10):
    idxs = [e for e in range(j, 101, 10)]
    B[j, idxs] = 1
b = np.linalg.pinv(B)
A = np.matmul(D, b)
print(D.shape, B.shape, A.shape, b.shape)

# Filter data
D1 = evoked1.data
a = np.linalg.pinv(A)
B1 = np.matmul(a, D1)
plt.imshow(B1)

# Plot filtered evoked
evoked2.plot_joint(title='Raw 2')
_evoked2 = evoked2.copy()
_evoked2.data = np.matmul(A, B)
_evoked2.plot_joint(title='Filter 2')

evoked1.plot_joint(title='Raw 1')
_evoked1 = evoked1.copy()
_evoked1.data = np.matmul(A, B1)
_evoked1.plot_joint(title='Filter 1')

print()
# %%
evoked_diff = evoked1.copy()
evoked_diff.data = evoked1.data - _evoked1.data
evoked_diff.plot_joint(title='Diff')
print()

# %%
