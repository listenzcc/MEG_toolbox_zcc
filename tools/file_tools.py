# File operations

import os
import mne


def read_raw_fif(path):
    """Read raw from [path]

    Args:
        path ({str}): The path of MEG raw file, commonly endswith raw.fif

    Returns:
        The MEG object of raw 
    """
    return mne.io.read_raw_fif(path, verbose=False)


def find_files(dir, ext='.fif'):
    """Find .fif files in [dir]

    Args:
        dir ({str}): The target directory
        ext ({str}}, optional): The extend of files to be found. Defaults to '.fif'.

    Returns:
        The list of found path of files
    """
    # Legal assert
    assert(os.path.isdir(dir))

    # Init files_path
    files_path = []

    # Try all files
    for name in os.listdir(dir):
        # Ignore dir
        if os.path.isdir(os.path.join(dir, name)):
            continue

        # Found a file
        if name.endswith(ext):
            files_path.append(os.path.join(dir, name))

    # Return
    print(f'Found {len(files_path)} files that ends with {ext}')
    return files_path
