# %% Importing
# System
import os
import sys

# Computing
import mne

# Private settings
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'private'))  # noqa
from dir_settings import RAW_DIR, MEMORY_DIR, SUBJECTS_DIR

# Setup Freesurfer SUBJECTS_DIR
mne.utils.set_config('SUBJECTS_DIR', SUBJECTS_DIR, set_env=True)

# %%


class Inverse_Solver():
    def __init__(self, running_name, use_memory=True):
        self.running_name = running_name
        self.subject_name = running_name.replace('MEG_S', 'RSVP_MRI_S')
        self.use_memory = use_memory
        print(f'Inversing on {self.subject_name}')

    def _get_source_stuff(self):
        # Get source stuffs
        # Get path of subject folder
        subject_path = os.path.join(RAW_DIR, self.running_name)

        def _path(ext):
            # Method to get file path that ends with [ext]
            return os.path.join(subject_path,
                                f'{self.subject_name}{ext}')

        # Read source space
        src = mne.read_source_spaces(_path('-src.fif'))

        # Read bem surfaces
        model = mne.read_bem_surfaces(_path('-bem.fif'))

        # Read bem solution
        bem_sol = mne.read_bem_solution(_path('-bem-sol.fif'))

        # Read trans matrix
        trans = mne.read_trans(_path('-trans.fif'))

        # Return
        return dict(src=src,
                    model=model,
                    bem_sol=bem_sol,
                    trans=trans)

    def _compute_inv(self, epochs, raw_info, loose, depth, n_jobs):
        # Compute inverse operator
        # Prepare forward solution
        fwd = mne.make_forward_solution(raw_info,
                                        self.source_stuff['trans'],
                                        self.source_stuff['src'],
                                        self.source_stuff['bem_sol'],
                                        eeg=False,
                                        n_jobs=n_jobs)

        # Compute covariance
        cov = mne.compute_covariance(epochs,
                                     tmax=.0,
                                     method='empirical',
                                     n_jobs=n_jobs)

        # Compute inverse operator
        inv = mne.minimum_norm.make_inverse_operator(raw_info,
                                                     fwd,
                                                     cov,
                                                     loose=loose,
                                                     depth=depth)

        # Report and return
        print(f'Computed inverse operator: {inv}')
        self.fwd = fwd
        self.cov = cov

        return inv

    def pipeline(self,
                 epochs,
                 re_compute_inv=False,
                 loose=0.2,
                 depth=0.8,
                 lambda2=1/9,
                 n_jobs=48,
                 raw_info=None):
        """Inverse opeartion pipeline,
        if [re_compute_inv], it will re-compute the inverse operator ignoring the memory.

        Args:
            epochs: The epochs to compute the inverse operator. Defaults to None.
            re_compute_inv ({bool}, optional): If recompute the inverse operator. Defaults to False.
            loose ({float}, optional): Parameter to compute the inverse operator. Defaults to 0.2.
            depth ({float}, optional): Parameter fo compute the inverse operator. Defaults to 0.8.
            lambda2 ({float}, optional): Parameter to compute the stc. Defaults to 1/9.
            n_jobs ({int}, optional): n_jobs of parall computation. Defaults to 48.
            raw_info ({Info}, optional): The raw_info to compute the inverse operator. Defaults to None.
        """

        # Prepare memory stuffs ---------------------------------------------
        memory_name = f'{self.subject_name}-inv.fif'
        memory_path = os.path.join(MEMORY_DIR, memory_name)

        # Get source stuffs -------------------------------------------------
        self.source_stuff = self._get_source_stuff()

        # Compute or Recall inv ---------------------------------------------
        if re_compute_inv:
            # Compute inverse solution
            self.inv = self._compute_inv(epochs,
                                         raw_info,
                                         loose,
                                         depth,
                                         n_jobs)
        else:
            try:
                assert(self.use_memory)

                # Recall inv
                self.inv = mne.minimum_norm.read_inverse_operator(memory_path)
                print(f'Inverse operator is recalled: {self.inv}')
            except:
                # Re-compute inverse solution
                self.inv = self._compute_inv(epochs,
                                             raw_info,
                                             loose,
                                             depth,
                                             n_jobs)

        # Save if use_memory
        if self.use_memory:
            mne.minimum_norm.write_inverse_operator(memory_path, self.inv)

    def estimate(self, obj, subject_to='fsaverage', spacing=6, lambda2=1/9):
        # Compute SourceEstimate --------------------------------------------
        # obj ({obj}): Epochs or Evoked
        # SourceEstimate in individual space
        stc = mne.minimum_norm.apply_inverse(obj,
                                             self.inv,
                                             lambda2=lambda2)
        print(f'Estimated stc: {stc}')

        # SourceMorph to fsaverage space
        morph = mne.compute_source_morph(src=self.inv['src'],
                                         subject_from=stc.subject,
                                         subject_to=subject_to,
                                         spacing=spacing)

        # SourceEstimate in fsaverage space
        stc_morph = morph.apply(stc)
        print(f'Estimated morphed stc: {stc_morph}')

        # Return
        return stc, stc_morph


# %%
# solver = Inverse_Solver('MEG_S03')
# d = solver._get_source_stuff()
# d

# %%
