# %%
import os
import sys
import time
import multiprocessing

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))  # noqa
import deploy
from local_tools import FileLoader

# %%
parameters_meg = dict(picks='mag',
                      stim_channel='UPPT001',
                      l_freq=0.1,
                      h_freq=7,
                      tmin=-0.2,
                      tmax=1.2,
                      decim=12,
                      detrend=1,
                      event='1',
                      events=['1', '2', '3'],
                      baseline=(None, 0))

# %%


def run_subject(name):
    loader = FileLoader(name, parameters=parameters_meg)
    loader.load_epochs(recompute=False)
    print(f'Done {name}.')


# %%
pool = []
for idx in range(1, 11):
    name = f'MEG_S{idx:02d}'
    run_subject(name)
    # p = multiprocessing.Process(target=run_subject, args=(name,))
    # p.start()
