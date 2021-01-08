# Set up default settings for analysis

import configparser

config = configparser.ConfigParser()
config.add_section('path')
config.add_section('epochs')

# Path settings
config['path'] = dict(
    preprocessed='_link_preprocessed',
    epochs='_link_epochs',
)

# Epochs settings
reject_criteria = dict(mag=4000e-15,     # 4000 fT
                       grad=4000e-13,    # 4000 fT/cm
                       eeg=150e-6,       # 150 µV
                       eog=250e-6)       # 250 µV

parameters_meg = dict(picks='mag',
                      stim_channel='UPPT001',
                      l_freq=0.1,
                      h_freq=15.0,
                      tmin=-0.2,
                      tmax=1.2,
                      decim=12,
                      detrend=1,
                      reject=dict(mag=4000e-15),
                      baseline=None)

parameters_eeg = dict(picks='eeg',
                      stim_channel='from_annotations',
                      l_freq=0.1,
                      h_freq=15.0,
                      tmin=-0.2,
                      tmax=1.2,
                      decim=10,
                      detrend=1,
                      reject=dict(eeg=150e-6),
                      baseline=None)

config['epochs'] = dict(
    reject_criteria=reject_criteria.__str__(),
    params_meg=parameters_meg.__str__(),
    params_eeg=parameters_eeg.__str__(),
)

# Write to the disk
config.write(open('auto-settings.ini', 'w'))
