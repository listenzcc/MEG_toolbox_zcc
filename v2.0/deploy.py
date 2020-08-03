# %%
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))  # noqa
from local_tools import Configure

# %%
config = Configure()

home = os.environ['HOME']
config.set('RAW_DIR', os.path.join(home, 'RSVP_dataset', 'processed_data'))
config.set('MEMORY_DIR', os.path.join(home, 'RSVP_dataset', 'memory'))
config.set('SUBJECTS_DIR', os.path.join(home, 'RSVP_dataset', 'subjects'))

# %%
