# Inventory the MEG/EEG RSVP dataset

# %%
import os
import pandas as pd
import configparser

from tqdm.auto import tqdm

# %%
config = configparser.ConfigParser()
config.read('auto-settings.ini')

# %%
frame = pd.DataFrame()

path0 = config['path']['preprocessed']
for subject in tqdm(os.listdir(path0)):
    if not any([subject.startswith(e) for e in ['EEG', 'MEG']]):
        continue

    path1 = os.path.join(path0, subject)
    files = os.listdir(path1)
    for i in range(20):
        raw_name = None

        if f'block_{i:02d}_raw.fif' in files:
            raw_name = f'block_{i:02d}_raw.fif'

        if f'block_{i:02d}_ica-raw.fif' in files:
            raw_name = f'block_{i:02d}_ica-raw.fif'

        if raw_name is None:
            continue

        series = pd.Series(dict(
            subject=subject,
            rawPath=os.path.join(path1, raw_name),
        ))

        frame = frame.append(series, ignore_index=True)

assert(not os.path.exists('inventory.json'))
frame.to_json('inventory.json')
frame


# %%
