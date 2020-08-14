# %%
import os
import sys
import pickle
import numpy as np
import multiprocessing

import mne
import sklearn.manifold as manifold
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # noqa
import deploy
from local_tools import FileLoader, Enhancer


# %%


results_dir = os.path.join('.', 'MVPA_data_xdawn_v2')
try:
    os.mkdir(results_dir)
except:
    pass
finally:
    assert(os.path.isdir(results_dir))

# %%


def prepare(name):
    # Load epochs
    loader = FileLoader(name)
    loader.load_epochs(recompute=False)
    print(loader.epochs_list)

    # Cross validation
    num_epochs = len(loader.epochs_list)
    for exclude in range(num_epochs):
        # Start on separate training and testing dataset
        print(f'---- {name}: {exclude} | {num_epochs} ----------------------')
        includes = [e for e in range(
            len(loader.epochs_list)) if not e == exclude]
        excludes = [exclude]
        train_epochs, test_epochs = loader.leave_one_session_out(includes,
                                                                 excludes)

        train_epochs = train_epochs[['1', '2', '4']]
        test_epochs = test_epochs[['1', '2', '4']]

        train_events = train_epochs.events.copy()
        test_events = test_epochs.events.copy()

        # Xdawn
        print('Xdawn 1 ------------------------------')
        enhancer1 = Enhancer(train_epochs=train_epochs,
                             test_epochs=test_epochs)
        train_epochs, test_epochs = enhancer1.fit_apply()

        # Relabel
        print('Relabel -----------------------------')
        selects = []
        for j, event in enumerate(train_events):
            if event[2] == 1:
                for k in [-1, 0, 1]:
                    selects.append(j + k)

        # new_epochs = train_epochs.copy()
        # new_epochs.event_id = {'1': 1, '4': 4}
        # new_epochs.events = train_epochs.events[selects]
        train_epochs.load_data()
        new_epochs = mne.BaseEpochs(train_epochs.info,
                                    train_epochs.get_data()[selects],
                                    events=train_epochs.events[selects],
                                    tmin=train_epochs.times[0],
                                    tmax=train_epochs.times[-1],
                                    baseline=None)

        print('Xdawn 2 ------------------------------')
        failed = False
        try:
            xdawn = mne.preprocessing.Xdawn(n_components=6)
            xdawn.fit(new_epochs)
            train_epochs = xdawn.apply(train_epochs)['1']
            test_epochs = xdawn.apply(test_epochs)['1']
        except:
            failed = True

        print('Baseline -----------------------------------')
        train_epochs.apply_baseline((None, 0))
        test_epochs.apply_baseline((None, 0))

        # break

        # Get train/test x/y
        print('Get data -----------------------------')
        train_X = train_epochs.get_data()
        test_X = test_epochs.get_data()

        # Save
        print('Save -------------------------------')
        data_name = f'{name}-{exclude}.pkl'
        tmpdata = dict(
            train_X=train_X,
            train_events=train_events,
            test_X=test_X,
            test_events=test_events,
            failed=failed
        )
        with open(os.path.join(results_dir, data_name), 'wb') as f:
            pickle.dump(tmpdata, f)
            print(f'Saved in {f.name}')

    # break

# %%
# for key in train_epochs.event_id:
#     train_epochs[key].average().plot_joint(title=f'Before {key}', show=True)
#     _train_epochs[key].average().plot_joint(title=f'After {key}', show=True)


# %%
for idx in range(1, 11):
    # Load epochs
    name = f'MEG_S{idx:02d}'
    print(name)
    prepare(name)
    # p = multiprocessing.Process(target=prepare, args=(name,))
    # p.start()

# %%
