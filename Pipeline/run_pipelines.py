# %% Importing
# Local tools -----------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))  # noqa
from MEG_worker import MEG_Worker


# %%
idx = 3
# Loading data ------------------------------------------
running_name = f'MEG_S{idx:02d}'
band_name = 'U07'

worker = MEG_Worker(running_name=running_name)
worker.pipeline(band_name=band_name)


# %%
worker.epochs.events.shape

# %%
