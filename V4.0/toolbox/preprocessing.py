
import mne


def denoise_projs(epochs):
    '''
    # Estimate SSP denoise projectors for the [epochs]
    - @epochs: The epochs whose SSP projectors are computed and applied
    see 'https://mne.tools/stable/auto_tutorials/preprocessing/plot_45_projectors_background.html#computing-projectors' for detail
    '''
    projs = mne.compute_proj_epochs(epochs)
    for p in projs:
        epochs.add_proj(p)

    return epochs
