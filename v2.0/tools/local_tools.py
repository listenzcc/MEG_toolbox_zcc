# File: local_tools.py
# Aim: Provide easy-to-use local toolbox

import os


def find_files(dir, ext='.fif'):
    # Find all files in [dir] with extension of [ext]
    # Return a list of full path of found files.

    def join(name):
        return os.path.join(dir, name)

    # Init files_path
    fullpaths = [join(name) for name in os.listdir(dir)
                 if all([name.endswith(ext),
                         not os.path.isdir(join(name))])]

    n = len(fullpaths)

    # Return
    print(f'Found {n} files that ends with {ext} in {dir}')
    return fullpaths


def mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass
