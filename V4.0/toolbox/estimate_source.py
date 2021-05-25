# Source estimation of the MEG

import os
import mne

# The table of source stuff we want
# Keys are the latest of the .fif file
# Values are the readers
stuff_table = {
    'bem-sol': mne.read_bem_solution,
    'bem': mne.read_bem_surfaces,
    'src': mne.read_source_spaces,
    'trans': mne.read_trans
}


def get_stuff(stuff_name, stuff_path, table=stuff_table):
    '''
    Get the stuffs for source estimation
    Read the file of [stuff_name] based on [cls_table] from [folder]

    Args:
    - @stuff_name: The stuff_name of the [stuff_path]
    - @stuff_path: The full path of the target file to be read
    - @table: The table of source stuff

    Out:
    - The object of the stuff
    '''
    reader = table[stuff_name]
    obj = reader(stuff_path)
    print(f'Got {stuff_name} from {stuff_path}')
    return obj


# Default params for source estimation


def generate_default_parameters():
    '''
    # Explains of pick_ori parameter pick_ori: None | “normal” | “vector”
    Options:

    None
    Pooling is performed by taking the norm of loose/free orientations. In case of a fixed source space no norm is computed leading to signed source activity.

    "normal"
    Only the normal to the cortical surface is kept. This is only implemented when working with loose orientations.

    "vector"
    No pooling of the orientations is done, and the vector result will be returned in the form of a mne.VectorSourceEstimate object.

    It turns out that the "normal" option is better, according to the MVPA analysis on source space
    '''

    params = dict(
        # Overall
        n_jobs=48,
        # Forward solution
        eeg=False,
        # Covariance matrix
        tmax=.0,
        method='empirical',
        # Inverse operator
        loose=0.2,
        depth=0.8,
        # Inverse computation
        lambda2=1/9,
        pick_ori='normal',  # None | 'normal'
        # Source morph
        # spacing=6,
    )

    return params


params = generate_default_parameters()


def subset(keys, params=params):
    '''
    Get subset form params by keys

    Args:
    - @keys: The keys to fetch
    - @params: The keys are found from params

    Out:
    - Dict with found items
    '''
    found = dict()
    for k in keys:
        found[k] = params.get(k, None)
    print(f'Got {keys} from params')
    return found


class SourceEstimator(object):
    '''
    Source estimator of MEG dataset
    '''

    def __init__(self, freesurfer_name):
        '''
        Initialization The Source Estimator
        - @freesurfer_name: The freesurfer name of the source estimator
        '''
        self.freesurfer_name = freesurfer_name
        print(
            f'Source estimator is initialized of "{freesurfer_name}"')

    def pre_estimation(self, src, sol, trans, epochs, info):
        '''
        Compute fwd, cov and inv before estimation

        Args:
        - @src: The source space src
        - @sol: The bem solution
        - @trans: The trans parameters of solid motion
        - @epochs: The epochs to compute the covariance matrix of baseline
        - @info: The info of obj, like raw/epochs/evoked, to compute the inverse operator

        Outs:
        - @fwd: The forward solution
        - @cov: The covariance matrix of baseline
        - @inv: The inverse operator
        '''

        # Forward solution
        kwargs = subset(['eeg', 'n_jobs'])
        fwd = mne.make_forward_solution(info, trans, src, sol, **kwargs)

        # Covariance matrix
        kwargs = subset(['tmax', 'method', 'n_jobs'])
        cov = mne.compute_covariance(epochs, **kwargs)

        # Inverse operator
        kwargs = subset(['loose', 'depth'])
        inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, **kwargs)

        self.src = src
        self.fwd = fwd
        self.cov = cov
        self.inv = inv

        return src, fwd, cov, inv

    def estimate(self, obj, subject_to='fsaverage'):
        '''
        Source estimation
        ! **Make Sure** use pre_estimation before estimation
        The Stc will be not be morphed, and the morph method will be returned.

        Args:
        - @obj: The MEG object to perform estimation, it can be Epochs or Evoked
        - @subject_to: The morph-to template name in freesurfer folder, default by 'fsaverage',
                       *users should be ware that the morph function works ONLY when 'fsaverage' is used.*

        Outs:
        - @stc: The estimated source activity in the space of [freesurfer_name]
        - @morph: The morph method from [subject_from] to [subject_to],
                  users have to morph the stc in after as required.
                  *The morph output will be None if subject_to is not 'fsaverage'.*
        '''

        kwargs = subset(['lambda2', 'pick_ori'])
        inv = self.inv

        stc = None

        if isinstance(obj, mne.Evoked):
            stc = mne.minimum_norm.apply_inverse(obj, inv, **kwargs)

        if isinstance(obj, mne.epochs.BaseEpochs):
            stc = mne.minimum_norm.apply_inverse_epochs(obj, inv, **kwargs)

        assert(not stc is None)

        if subject_to == 'fsaverage':
            src = mne.read_source_spaces(os.path.join(
                os.environ['SUBJECTS_DIR'], 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
            kwargs = dict(
                src=self.src,
                subject_from=self.freesurfer_name,
                subject_to=subject_to,
                src_to=src,
            )
            morph = mne.compute_source_morph(**kwargs)
        else:
            morph = None

        self.stc = stc
        self.morph = morph

        return stc, morph
