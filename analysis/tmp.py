# %%
import os
import sys

# Local tools -----------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))  # noqa
from MEG_worker import MEG_Worker


# %%
worker = MEG_Worker(running_name='MEG_S01')
worker.pipeline(band_name='U07')
