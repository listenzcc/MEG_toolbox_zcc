# File: get_objects.py
# Version: 0.1
# Aim: Methods of getting raw, epochs and so on.

# %%
import os
import mne
import time
import numpy as np
from .settings import DataFolder


# %% Raw


def get_raw(subject, session):
    # Prepare possible pathes
    ica_path = os.path.join(DataFolder, subject,
                            f'block_{session:02d}_ica-raw.fif')
    raw_path = os.path.join(DataFolder, subject,
                            f'block_{session:02d}_raw.fif')

    # Get ica_path or raw_path if it exists
    path = None
    if os.path.exists(ica_path):
        path = ica_path
    else:
        if os.path.exists(raw_path):
            path = raw_path

    # Return if path is None
    if path is None:
        return None

    return mne.io.read_raw_fif(path)


def get_raws(subject):
    raws = []
    for session in range(20):
        raw = get_raw(subject, session)
        if raw is None:
            continue
        else:
            raws.append(raw)

    print(f'Found {len(raws)} raws in {subject}.')
    return raws


def concatenate_raws(raws):
    print(f'Concatenate {len(raws)} raws')
    return mne.concatenate_raws(raws)

# %% Epochs


def relabel_events(events):
    # Relabel events
    # Relabeled event:
    #  1: Target
    #  2: Far non-target randomly selected matching with target
    #  3: Button motion
    #  4: Far non-target non selected
    #  5: Near non-target

    print(f'Relabel {len(events)} events')
    t0 = time.time()

    for j, event in enumerate(events):
        if event[2] == 1:
            count, k = 0, 0
            while True:
                k += 1
                try:
                    if events[j+k, 2] == 2:
                        events[j+k, 2] = 5
                        count += 1
                except IndexError:
                    break
                if count == 5:
                    break

            count, k = 0, 0
            while True:
                k += 1
                try:
                    if events[j-k, 2] == 2:
                        events[j-k, 2] = 5
                        count += 1
                except IndexError:
                    break
                if count == 5:
                    break

    n1 = len(np.where(events[:, 2] == 1)[0])
    n2 = len(np.where(events[:, 2] == 2)[0])

    pos = np.where(events[:, 2] == 2)[0]
    np.random.shuffle(pos)
    events[pos[:n2-n1], 2] = 4

    print('Passed {} seconds.'.format(time.time() - t0))
    return events


def get_epochs(raw, params_name='MEG'):
    filter_params = dict(l_freq=0.1,
                         h_freq=50)

    if params_name == 'MEG':
        events = mne.find_events(raw, stim_channel='UPPT001')
        events = relabel_events(events)

        params = dict(events=events,
                      picks='mag',
                      tmin=-0.2,
                      tmax=1.2,
                      decim=12,
                      detrend=1,
                      reject=dict(mag=4e-12),
                      baseline=(None, 0))

        epochs = mne.Epochs(raw, **params)
        epochs.drop_bad()

        print(f'Got epochs: {epochs}')

        return epochs
