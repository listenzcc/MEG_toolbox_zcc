# File: configure.py
# Aim: Provide tools for setting and getting environ variables

# %%
import os
import warnings

# %% ----------------------------------
# Set and get local configures through environments


class Configure(object):
    def __init__(self, prefix='_MEG_RSVP_'):
        # prefix is used to find the customer settings
        self.prefix = prefix

    def unique(self, key):
        # Make up unique key
        return f'{self.prefix}{key}'

    def set(self, key, value):
        # Setup key as value
        # The settings will be restored in os.environ
        unique = self.unique(key)
        v = os.environ.get(unique)
        if v is not None:
            warnings.warn(
                f'Found "{key}" in env, overwriting "{v}" with "{value}"', RuntimeWarning)
        os.environ[unique] = value

    def get(self, key):
        # Get key
        return os.environ.get(self.unique(key))

    def getall(self):
        # Get all keys
        outputs = dict()
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                outputs[key[len(self.prefix):]] = value
        return outputs

# %%
