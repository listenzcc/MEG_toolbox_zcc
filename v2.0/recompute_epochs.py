# %%
# This script will recompute epochs and restore them in memory dir (see deploy.py).
# It uses multiprocessing to operate, and it requires a long time to complete.
# You can run the script as following.
# python recompute_epochs.py >> running.log

# %%
import multiprocessing

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
                          mag=4e-12,
                      ),
                      baseline=None)

# %%


def run_subject(name):
    loader = DataManager(name, parameters=parameters_meg)
    loader.load_epochs(recompute=True)
    print(f'Done {name}.')


# %%
# # pool = []
for idx in range(1, 11):
    name = f'MEG_S{idx:02d}'
    # run_subject(name)
    p = multiprocessing.Process(target=run_subject, args=(name,))
    p.start()

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
