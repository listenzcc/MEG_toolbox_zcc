# File: configure.py
# Aim: Provide tools for setting and getting environ variables

# %%
# import os
import warnings
import configparser

# %% ----------------------------------
# Set and get local configures through environments


class Configure(object):
    def __init__(self, section_name='RSVP-dataset'):
        parser = configparser.ConfigParser()
        parser.add_section(section_name)
        self.parser = parser
        self.section_name = section_name

    def set(self, key, value):
        # Setup entry of [key] as [value]
        if self.parser.has_option(self.section_name, key):
            v = self.parser.get_option(self.section_name, key)
            warnings.warn(
                f'Found "{key}" in config, overwriting "{v}" with "{value}"', UserWarning)

        self.parser[self.section_name][key] = value
        print(f'Added "{key}" as "{value}"')

    def get(self, key):
        # Get key
        if not self.parser.has_option(self.section_name, key):
            warnings.warn(
                f'Can not find "{key}" in config, using "None" instead', UserWarning)
            return None
        return self.parser[self.section_name][key]

    def get_all(self):
        # Get all keys
        return dict(self.parser[self.section_name])


# %%
