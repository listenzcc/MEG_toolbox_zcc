# Compute epochs based on inventory
# Will COST VERY LONG TIME, like a day, depending on your computer

# %%
import os
import mne
import pandas as pd
import configparser

from tqdm.auto import tqdm

# %%
# If the inventory-epo.json already exists,
# it should better stop and check the process,
# to prevent wasting of time
assert(not os.path.exist('inventory-epo.json'))

# %%
n_jobs = 48
config = configparser.ConfigParser()
config.read('auto-settings.ini')


# %%
inventory = pd.read_json('inventory.json')
inventory['epochsPath'] = '-'
inventory

# %%
reject_criteria = eval(config['epochs']['reject_criteria'])
params_meg = eval(config['epochs']['params_meg'])
params_eeg = eval(config['epochs']['params_eeg'])
reject_criteria, params_meg, params_eeg

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
cccc = 0
destinations = dict()

for i in tqdm(range(len(inventory))):
    # ------------------------------------
    # Get series of the frame
    series = inventory.iloc[i]
    print(series.rawPath)

    # ------------------------------------
    # Prepare parameters
    parameters = None

    if series.subject.startswith('EEG'):
        # continue
        parameters = params_eeg

    if series.subject.startswith('MEG'):
        parameters = params_meg

    assert(parameters is not None)

    # ------------------------------------
    # Get and filter raw
    raw = mne.io.read_raw(series.rawPath)
    raw.load_data()

    if series.subject.startswith('MEG'):
        raw.apply_gradient_compensation(0)

        chpi_locs = mne.chpi.extract_chpi_locs_ctf(raw)

        head_pos = mne.chpi.compute_head_pos(raw.info,
                                             chpi_locs=chpi_locs,
                                             verbose=0)

        dest = destinations.get(series.subject, raw.info['dev_head_t'])
        destinations[series.subject] = dest

        raw = mne.preprocessing.maxwell_filter(
            raw, origin=[0, 0, 0], destination=dest)

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

    subject_dir = os.path.join(config['path']['epochs'], series.subject)
    if not os.path.isdir(subject_dir):
        os.mkdir(subject_dir)

    epochs_path = os.path.join(subject_dir, epochs_name)

    # ------------------------------------
    # Save epochs
    epochs.load_data()
    epochs.save(epochs_path, overwrite=True)
    inventory['epochsPath'].iloc[i] = epochs_path

    # cccc += 1
    # if cccc == 2:
    #     break
    # break

# %%
assert(not os.path.exist('inventory-epo.json'))
inventory.to_json('inventory-epo.json')
print('All Done.')


# %%
