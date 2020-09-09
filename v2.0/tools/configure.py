# File: configure.py
# Aim: Provide tools for setting and getting environ variables

# %%
import os

# %% ----------------------------------
# Set and get local configures through environments


class Configure():
    def __init__(self, prefix='_MEG_RSVP_'):
        self.prefix = prefix

    def unique(self, key):
        return f'{self.prefix}{key}'

    def set(self, key, value):
        os.environ[self.unique(key)] = value

    def get(self, key):
        return os.environ.get(self.unique(key))

    def getall(self):
        outputs = dict()
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                outputs[key[len(self.prefix):]] = value
        return outputs
