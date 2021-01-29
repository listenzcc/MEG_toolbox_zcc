
import mne


def denoise_projs(epochs, apply=True):
    '''
    # Estimate SSP denoise projectors for the [epochs]
    - @epochs: The epochs whose SSP projectors are computed and applied
    - @apply: Whether apply_proj to the epochs, default by True
    see 'https://mne.tools/stable/auto_tutorials/preprocessing/plot_45_projectors_background.html#computing-projectors' for detail
    '''
    projs = mne.compute_proj_epochs(epochs)
    for p in projs:
        epochs.add_proj(p)

    if apply:
        epochs.apply_proj()

    return epochs
