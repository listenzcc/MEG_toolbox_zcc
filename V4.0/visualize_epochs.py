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
