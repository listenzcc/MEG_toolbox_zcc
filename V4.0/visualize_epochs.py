# Visualize epochs

# %%
import os
import mne
import pandas as pd
import configparser

from tqdm.auto import tqdm

# %%
epochs_inventory = pd.read_json('inventory-epo.json')
epochs_inventory

# %%


def fetch_subject(subject, frame=epochs_inventory):
    return frame.query(f'subject == "{subject}"')


# %%
epochs_list = [mne.read_epochs(e) for e in
               fetch_subject('MEG_S02')['epochsPath'].tolist()]


# %%
epochs = mne.concatenate_epochs(epochs_list)
epochs['1'].average().plot_joint()
epochs

# %%
