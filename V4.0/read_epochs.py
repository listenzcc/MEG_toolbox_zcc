# Easy-to-use python functions for reading epochs
import pandas as pd
import mne


def read_all_epochs(subject_name, auto_baseline=True, use_list=False):
    '''
    # Read all epochs of [subject_name] on the disk
    - @subject_name: The name of subject
    - @auto_baseline: Whether apply baseline of (None, 0) to the found epochs, default by True
    - @use_list: Whether to return the epochs list, default by False,
                 if equaling to True, the epochs_list will be returned,
                 it is designed for MVPA analysis, since it separates the epochs from different sessions
    '''
    epochs_inventory = pd.read_json('inventory-epo.json')

    assert(subject_name in epochs_inventory.subject.to_list())

    def fetch_subject(subject, frame=epochs_inventory):
        return frame.query(f'subject == "{subject}"')

    epochs_list = [mne.read_epochs(e) for e in
                   fetch_subject(subject_name)['epochsPath'].tolist()]

    epochs_list = [e for e in epochs_list if len(e) > 100]

    # If [use_list],
    # it will stop here and return the epochs_list
    if use_list:
        if auto_baseline:
            epochs_list = [e.apply_baseline((None, 0)) for e in epochs_list]
        return epochs_list

    # If not [use_list],
    # the projs are concatenated

    # Concatenate
    epochs = mne.concatenate_epochs(epochs_list)

    if auto_baseline:
        epochs = epochs.apply_baseline((None, 0))

    return epochs
