# Compute epochs based on inventory

# %%
import os
import mne
import pandas as pd
import configparser

from tqdm.auto import tqdm

# %%
n_jobs = 48
config = configparser.ConfigParser()
config.read('settings.ini')
config['Path']['home'] = os.path.join(os.environ['HOME'], 'RSVP_dataset')

if not os.path.isdir(config['Path']['epochs']):
    os.mkdir(config['Path']['epochs'])

# %%
inventory = pd.read_json('inventory.json')
inventory['epochsPath'] = '-'
inventory

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


def relabel_events(events):
    # Relabel events
    # Relabeled event:
    #  1: Target
    #  2: Far non-target
    #  3: Button motion
    #  4: Near non-target

    print(f'Relabel {len(events)} events')

    for j, event in enumerate(events):
        if event[2] == 1:
            count, k = 0, 0
            while True:
                k += 1
                try:
                    if events[j+k, 2] == 2:
                        events[j+k, 2] = 4
                        count += 1
                except IndexError:
                    break
                if count == 10:
                    break

            count, k = 0, 0
            while True:
                k += 1
                try:
                    if events[j-k, 2] == 2:
                        events[j-k, 2] = 4
                        count += 1
                except IndexError:
                    break
                if count == 10:
                    break

    return events


# %%
for i in tqdm(range(len(inventory))):
    # ------------------------------------
    # Get series of the frame
    series = inventory.iloc[i]
    print(series.rawPath)

    # ------------------------------------
    # Prepare parameters
    parameters = None

    if series.subject.startswith('EEG'):
        parameters = parameters_eeg

    if series.subject.startswith('MEG'):
        parameters = parameters_meg

    assert(parameters is not None)

    # ------------------------------------
    # Get and filter raw
    raw = mne.io.read_raw(series.rawPath)
    raw.load_data()

    if series.subject.startswith('MEG'):
        events = mne.find_events(raw,
                                 stim_channel=parameters['stim_channel'])
    else:
        events = mne.events_from_annotations(raw)[0]

    events = relabel_events(events)

    filtered_raw = raw.filter(l_freq=parameters['l_freq'],
                              h_freq=parameters['h_freq'],
                              n_jobs=n_jobs)

    # ------------------------------------
    # Get epochs
    epochs = mne.Epochs(
        filtered_raw,
        events=events,
        picks=parameters['picks'],
        tmin=parameters['tmin'],
        tmax=parameters['tmax'],
        decim=parameters['decim'],
        detrend=parameters['detrend'],
        reject=parameters['reject'],
        baseline=parameters['baseline'],
    )

    # ------------------------------------
    # Prepare epochs path
    epochs_name = os.path.basename(series.rawPath)[
        :len('block_00')] + '-epo.fif'

    subject_dir = os.path.join(config['Path']['epochs'], series.subject)
    if not os.path.isdir(subject_dir):
        os.mkdir(subject_dir)

    epochs_path = os.path.join(subject_dir, epochs_name)

    # ------------------------------------
    # Save epochs
    epochs.save(epochs_path)
    inventory['epochsPath'].iloc[i] = epochs_path

    # break

# %%
inventory.to_json('inventory-epo.json')
print('All Done.')
