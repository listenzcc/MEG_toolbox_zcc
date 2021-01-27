# Analysis RSVP data of MEG and EEG signal

- [Analysis RSVP data of MEG and EEG signal](#analysis-rsvp-data-of-meg-and-eeg-signal)
  - [Structure](#structure)
  - [Settings](#settings)
  - [Epochs Generation](#epochs-generation)
    - [Raw File Inventory](#raw-file-inventory)
    - [Epochs Segment and Computation](#epochs-segment-and-computation)
    - [Read Epochs](#read-epochs)
  - [Sensor Space Computation](#sensor-space-computation)
  - [Source Estimation](#source-estimation)

## Structure

I would like to organize the scripts in a flatten manner.
Which means the scripts and other files (like files of .json, .ini, ...) will **almost** all being listed on the top-level folder.
To make them understandable, a file table is used.

Table 1 File Table

| File name                                                                      | Function                                       | Description           | Section                                               |
| ------------------------------------------------------------------------------ | ---------------------------------------------- | --------------------- | ----------------------------------------------------- |
| Settings                                                                       |                                                |                       |
| [auto_settings.py](./auto_settings.py)                                         | Generate settings of MEG and EEG data          | Python Script         | [Settings](#settings)                                 |
| [auto-settings.ini](./auto-settings.ini)                                       | Settings of MEG and EEG data                   | Auto Generated        | [Settings](#settings)                                 |
| Epochs Generation                                                              |                                                |                       |
| [inventory.py](./inventory.py)                                                 | Detect raw .fif files of MEG and EEG data      | Python Script         | [Epochs Generation](#epochs-generation)               |
| [inventory.json](./inventory.json)                                             | Inventory of raw files, in .json format        | Auto Generated        | [Epochs Generation](#epochs-generation)               |
| [compute_epochs.py](./compute_epochs.py)                                       | Generate epochs based on the inventory         | Python Script         | [Epochs Generation](#epochs-generation)               |
| [inventory-epo.json](./inventory-epo.json)                                     | Inventory of epochs files, in .json format     | Auto Generated        | [Epochs Generation](#epochs-Generation)               |
| [read_epochs.py](./read_epochs.py)                                             | The method of read epochs based on inventory   | Python Script         | [Epochs Generation](#epochs-generation)               |
| Visualization Epochs                                                           |                                                |                       |
| [visualize_epochs.py](./visualize_epochs.py)                                   | Visualize the evoked and epochs in time-shifts | Python Script         | [Sensor Space Computation](#sensor-space-computation) |
| Source Estimation                                                              |                                                |                       |
| [source_estimation.py](./source_estimation.py)                                 | Estimate neural activity in cortex             | Python Script         | [Source Estimation](#source-estimation)               |
| [source_visualization_3DSurface.ipynb](./source_visualization_3DSurface.ipynb) | Plot activity in cortex                        | Notebook Script       | [Source Estimation](#source-estimation)               |
| [source_visualization_corr.ipynb](./source_visualization_corr.ipynb)           | Plot source corr in circle graph               | Notebook Script       | [Source Estimation](#source-estimation)               |
| [source_visualization_waveform.py](./source_visualization_waveform.py)         | Plot source waveform in plotly                 | Notebook Script       | [Source Estimation](#source-estimation)               |
| Stand Alone Tools                                                              |                                                |                       |
| [toolbox](./toolbox)                                                           | Python package of local utils                  | Python Package Folder | Stand alone                                           |
| [batch.py](./batch.py)                                                         | Automatic batch script                         | Python Script         | Stand alone                                           |

---

## Settings

The settings are stored in the file of [auto-settings.ini](./auto-settings.ini).
It is generated by the python script of [auto_settings.py](./auto_settings.py).

Example usage in shell:

```sh
# Make sure auto-settings.ini does not exist
python auto_settings.py
```

The generated ini file contains

- The path of processed and epochs files, in section of 'path'
- The parameters of generating MEG and EEG epochs, in section of 'epochs'

---

## Epochs Generation

I assume the raw MEG and EEG data has been preprocessed.
The thing is read the raw files and generate epochs.

### Raw File Inventory

The raw MEG and EEG files were in the folder of 'preprocessed' option in the setting.
The python script of [inventory.py](./inventory.py) is to detect the raw .fif file.

Example usage in shell:

```sh
# Make sure inventory.json does not exist
python inventory.py
```

The file of [inventory.json](./inventory.json) will be generated accordingly.
There are two columns in the DataFrame

- rawPath: full path of raw .fif file
- subject: subject name, like 'MEG_S02' or 'EEG_S02'.

### Epochs Segment and Computation

The epochs will be generated based on the inventory of each raw files.
The python script of [compute_epochs.py](./compute_epochs.py) is to perform the generation.
It will read [inventory.json](./inventory.json) for raw files and generate epochs accordingly,
during the epochs generation, the parameters in settings are used.

Eventually, it will generate [inventory-epo.json](./inventory-epo.json),
by adding columns of

- epochsPath: full path of epochs file

Note before run: **Make sure everything is OK before start it, since the generation process costs much CPU resource.**

Example usage in shell:

```sh
# Make sure everything is OK before start it, since the generation process costs much CPU resource
# Make sure inventory-epo.json does not exist
python compute_epochs.py
```

### Read Epochs

For users to easily use the inventory, I provide the python script of [read_epochs.py](./read_epochs.py).
It contains easy-to-use python function to read epochs.

Example usage in python:

```python
# Import method of read all available epochs
from read_epochs import read_all_epochs
# Read all epochs for 'MEG_S02'
all_epochs = read_all_epochs('MEG_S02')
```

After the epochs reading, the de-noise projection is necessary and on-time.
In the current practice, the Signal-space projection (SSP) method is used.
See [MNE document](https://mne.tools/stable/auto_tutorials/preprocessing/plot_45_projectors_background.html#computing-projectors) for detail.

Example usage in python:

```python
# Import method of denoise using SSP projector
from toolbox.preprocessing import denoise_projs
# Compute SSP projector and apply it to the [epochs]
# - @epochs: The epochs with type of mne.BaseEpochs or something likes
denoise_projs(epochs)
```

---

## Sensor Space Computation

The first phase of analysis is to visualize the **epochs** and **evoked** responses.

The waveforms of the multi-channel MEG or EEG dataset is plotted using [visualize_epochs.py](./visualize_epochs.py).
Additionally, the time-shifts among the epochs are also plotted using [visualize_epochs.py](./visualize_epochs.py) too.
The figures will be plotted in one single .pdf file for each subject (like 'MEG_S02', 'EEG_S02', ... as long as they are in the subject column of [inventory-epo.json](./inventory-epo.json)).

The .pdf file will be stored in [Visualization/Evoked_and_TimeShift](./Visualization/Evoked_and_TimeShift).
One example can be found in [MEG_S02.pdf](./Visualization/Evoked_and_TimeShift/MEG_S02.pdf).

Users can see the [compute_lags.py](./toolbox/compute_lags.py) for computation detail.

Example usage in shell:

```sh
# Make sure the results folder of Visualization/Evoked_and_TimeShift exists
# To analysis the data of 'MEG_S02', use the following command
# MEG_S02.pdf will be stored in the results folder
python visualize_epochs.py MEG_S02
```

## Source Estimation

The MEG data analysis requires source estimation analysis.
I use several scripts to perform the analysis.
I use [source_estimation.py](./source_source.py) to operate estimation for each subject.
After operating the script, the source activity will be estimated and stored in [MiddleResults/SourceEstimation](./MiddleResults/SourceEstimation-norm/).

To perform multi-subject analysis, the activity are also morphed to template source model, which is currently **fsaverage** subject in freesurfer.
The morphed activity and morph file are also stored in the folder.
Additionally, the '-norm' suffix of **SourceEstimation** folder refers to restrict the micro current direction to orthogonal to surface.

Users can see the [estimate_source.py](./toolbox/estimate_source.py) for computation detail.
To use freesurfer correctly, the script of [set_mne_freesurfer.py](./set_mne_freesurfer.py) is also provided to set environ variables.

Example usage in shell

```sh
# MEG_S02 and RSVP_MRI_S02 refer the subject name of experiment and freesurfer respectively
# The script requires the two names to link MEG and MRI data of the subject
python source_estimation.py MEG_S02 RSVP_MRI_S02
```

Additionally, the jupyter notebook scripts of
[source_visualization_3DSurface.ipynb](./source_visualization_3DSurface.ipynb), [source_visualization_corr.ipynb](./source_visualization_corr.ipynb) and [source_visualization_waveform.py](./source_visualization_waveform.py)
are used to visualize the output of source estimation.
The reason I apply notebook scripts is because it allows **pyvista 3D backend** to be correctly used as plotting.
Users have to run the scripts to see the results.

---
