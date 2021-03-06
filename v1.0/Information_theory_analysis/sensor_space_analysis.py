# %% Importing
# System
import os
import sys
import time
import pickle
import multiprocessing

# Computing
import mne
import numpy as np
from scipy.stats import entropy

# Plotting
import tqdm
import matplotlib.pyplot as plt

# Local tools
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))  # noqa
from MEG_worker import MEG_Worker
from figure_tools import Drawer

# Settings
BAND_NAME = 'U07'

SEGMENT_RANGES = dict(
    a=dict(crop=(0.15, 0.25), center=0.2),
    b=dict(crop=(0.25, 0.35), center=0.3),
    c=dict(crop=(0.35, 0.45), center=0.4),
    d=dict(crop=(0.45, 0.55), center=0.5),
    e=dict(crop=(0.55, 0.65), center=0.6),
    f=dict(crop=(0.65, 0.75), center=0.7),
)

COLORS_272 = np.load(os.path.join(os.path.dirname(__file__),
                                  '..', 'tools', 'colors_272.npy'))

RE_COMPUTE_ENTROPY = False
NUM_BINS = 10
TMP_DIR = os.path.join('.', 'tmp_mutual')
if not os.path.exists(TMP_DIR):
    os.mkdir(TMP_DIR)

DRAWER = Drawer()
SHOW = False

# %% Tools


def hist(x, y, num_bins=NUM_BINS):
    _range = (min((min(x), min(y))), max((max(x), max(y))))
    x_hist, edges = np.histogram(x, bins=num_bins, range=_range)
    y_hist, _ = np.histogram(y, bins=edges)
    xy_hist, _, _ = np.histogram2d(x, y, bins=(edges, edges), range=_range)
    return (x_hist / num_bins,
            y_hist / num_bins,
            np.ravel(xy_hist) / num_bins)


def compute_entropy(x, y, num_bins=NUM_BINS):
    # Compute histograms, x, y, and joint x and y
    x_hist, y_hist, xy_hist = hist(x, y, num_bins)

    # Compute entropy
    _entropy = entropy(x_hist) + entropy(y_hist) - entropy(xy_hist)

    return _entropy


def compute_entropy_matrix(data1, data2, num_sensors, compare_name):
    print(f'Working on {compare_name}')
    entropy_matrix = np.zeros((num_sensors, num_sensors))

    for j in range(num_sensors):
        for k in range(num_sensors):
            _entropy = compute_entropy(data1[:, j], data2[:, k])
            entropy_matrix[j, k] = _entropy
        if j % 50 == 0:
            print(f'{compare_name} - {j}')

    np.save(os.path.join(TMP_DIR,
                         f'{compare_name}.npy'),
            entropy_matrix)

    print(f'Done {compare_name}')

# %%


for idx in range(1, 11):
    # Setting -------------------------------------------
    running_name = f'MEG_S{idx:02d}'

    plt.style.use('ggplot')

    # Worker pipeline -----------------------------------
    worker = MEG_Worker(running_name=running_name)
    worker.pipeline(band_name=BAND_NAME)

    # Compute segments ----------------------------------
    # Init empty segments and segment_data
    segments = dict()
    segment_data = dict()

    # Plot evoked
    DRAWER.fig = worker.clean_epochs.average().plot_joint(
        times=[e['center'] for e in SEGMENT_RANGES.values()],
        title=running_name,
        show=SHOW)

    # Set values
    for key in SEGMENT_RANGES:
        crop = SEGMENT_RANGES[key]['crop']
        segments[key] = worker.clean_epochs.copy().crop(crop[0], crop[1])
        segment_data[key] = np.mean(segments[key].get_data(), axis=-1)

    # %% ---------------------------------------------------------------------
    # Set up number of sensors
    num_sensors = 272
    names = []

    if RE_COMPUTE_ENTROPY:
        # Compute entropy matrixes
        # Travel all comparison
        for n1 in SEGMENT_RANGES:
            for n2 in SEGMENT_RANGES:
                # Only compute entropy between time two ranges once,
                # by resticking the conditions that n2 not less than n1
                if n2 < n1:
                    continue

                # Set up compare_name
                compare_name = f'{running_name}-{n1}-{n2}'
                print(compare_name)
                names.append(compare_name)

                process = multiprocessing.Process(target=compute_entropy_matrix,
                                                  args=(segment_data[n1],
                                                        segment_data[n2],
                                                        num_sensors,
                                                        compare_name))

                process.start()

        while True:
            if not all([os.path.exists(os.path.join(TMP_DIR, f'{e}.npy')) for e in names]):
                time.sleep(1)
            else:
                break

    # %% ----------------------------------------------------------------------
    entropy_matrixes = dict()

    for n1 in SEGMENT_RANGES:
        for n2 in SEGMENT_RANGES:
            if n2 < n1:
                continue
            compare_name = f'{running_name}-{n1}-{n2}'
            entropy_matrixes[compare_name] = np.load(os.path.join(TMP_DIR,
                                                                  f'{compare_name}.npy'))

    # %% Plot ----------------------------------------------------
    plt.style.use('default')
    names = [e for e in SEGMENT_RANGES]

    fig, axes = plt.subplots(6, 6, figsize=(12, 12))
    for j, n1 in enumerate(names):
        for k, n2 in enumerate(names):
            if n2 < n1:
                continue
            compare_name = f'{running_name}-{n1}-{n2}'
            print(compare_name)
            ax = axes[j][k]
            ax.imshow(entropy_matrixes[compare_name],
                      origin='lower', vmin=0, vmax=1)
            ax.set_title(compare_name)

    fig.tight_layout()

    DRAWER.fig = fig

# %%
DRAWER.save('sensor_space_mutual_information.pdf')

# %%
