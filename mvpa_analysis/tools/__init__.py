# File: __init__.py
# Aim: Default __init__ file for package

import os
from .configure import Configure

config = Configure()
home = os.environ['HOME']
config.set('RAW_DIR', os.path.join(home, 'RSVP_dataset', 'processed_data'))
config.set('MEMORY_DIR', os.path.join(home, 'RSVP_dataset', 'memory'))
config.set('SUBJECTS_DIR', os.path.join(home, 'RSVP_dataset', 'subjects'))

print(config.get_all())
