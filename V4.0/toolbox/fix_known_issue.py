# Fix known issues

import mne


def custom_montage(epochs):
    '''
    Rename channels in montage of standard_1020, to match our EEG setting-up.
    - @epochs: The epochs's montage will be set as fixed montage
    '''
    # Replacement table, change {key} to {value} in standard montage
    dm = mne.channels.make_standard_montage('standard_1020')

    # ! Not sure the replacement is correct, but the position looks alright
    replace = dict(
        A1='CB1',  # Not sure
        A2='CB2',  # Not sure
    )

    # Replace name using table
    for j, n in enumerate(dm.ch_names):
        if n in replace:
            dm.ch_names[j] = replace[n]

    # Change ch_names in dm based on the ch_names in info
    # Make sure the ch_names between dm and info match
    # ! It is perhaps SLOW, but EFFECTIVE
    for j, name in enumerate(epochs.info.ch_names):
        for k, n in enumerate(dm.ch_names):
            if name.lower() == n.lower():
                dm.ch_names[k] = name

    # Set montage
    epochs.set_montage(dm)

    return epochs
