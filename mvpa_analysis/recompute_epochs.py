# %%
# This script will recompute epochs and restore them in memory dir (see deploy.py).
# It uses multiprocessing to operate, and it requires a long time to complete.
# You can run the script as following.
# python recompute_epochs.py >> running.log

# %%
import multiprocessing

from tools.data_manager import DataManager

# %%
reject_criteria = dict(mag=4000e-15,     # 4000 fT
                       grad=4000e-13,    # 4000 fT/cm
                       eeg=150e-6,       # 150 µV
                       eog=250e-6)       # 250 µV

parameters_meg = dict(picks='mag',
                      stim_channel='UPPT001',
                      l_freq=0.1,
                      h_freq=15.0,
                      tmin=-0.2,
                      tmax=1.2,
                      decim=12,
                      detrend=1,
                      reject=dict(mag=4000e-15),
                      baseline=None)

parameters_eeg = dict(picks='eeg',
                      stim_channel='from_annotations',
                      l_freq=0.1,
                      h_freq=15.0,
                      tmin=-0.2,
                      tmax=1.2,
                      decim=10,
                      detrend=1,
                      reject=dict(eeg=150e-6),
                      baseline=None)


# %%


def run_subject(name, parameters, recompute=True):
    loader = DataManager(name, parameters=parameters)
    loader.load_epochs(recompute=recompute)
    print(f'Done {name}.')


# %%
# # pool = []
for idx in range(1, 11):
    name = f'EEG_S{idx:02d}'

    # Run in sequence
    run_subject(name, parameters_eeg)

    # Run in parallel
    # p = multiprocessing.Process(target=run_subject,
    #                             args=(name, parameters_eeg))
    # p.start()

# %%
# idx = 3
# name = f'MEG_S{idx:02d}'
# loader = FileLoader(name, parameters=parameters_meg)
# loader.load_epochs(recompute=False)
# print(loader.epochs_list)
# t = 5
# includes = [e for e in range(len(loader.epochs_list)) if not e == t]
# excludes = [t]
# a, b = loader.leave_one_session_out(includes, excludes)
# print(a, b)


# %%
# for eid in a.event_id:
#     print(eid)
#     a[eid].average().plot_joint()
#     b[eid].average().plot_joint()

# %%
