# %%
import os
import sys
import pickle
import multiprocessing

import mne
from mne.decoding import Vectorizer
from sklearn import svm
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # noqa
import deploy
from local_tools import FileLoader, Enhancer, Denoiser

# %%


results_dir = os.path.join('.', 'SVM_xdawn_denoise_1')
try:
    os.mkdir(results_dir)
except:
    pass
finally:
    assert(os.path.isdir(results_dir))


def get_X_y(epochs):
    # Get data [X] and label [y] from [epochs]
    X = epochs.get_data()
    y = epochs.events[:, -1]
    return X, y


def mvpa(name):
    # Perform MVPA
    # Setting
    BASELINE = (None, 0)
    CROP = (0, 0.8)
    EVENTS = ['1', '2']

    # Load epochs
    loader = FileLoader(name)
    loader.load_epochs(recompute=False)
    print(loader.epochs_list)

    # Prepare [predicts] for results
    predicts = []

    # Cross validation
    num_epochs = len(loader.epochs_list)
    for exclude in range(num_epochs):
        # Start on separate training and testing dataset
        print(f'---- {name}: {exclude} | {num_epochs} ----------------------')
        includes = [e for e in range(
            len(loader.epochs_list)) if not e == exclude]
        excludes = [exclude]
        train_epochs, test_epochs = loader.leave_one_session_out(includes,
                                                                 excludes)
        print(train_epochs, test_epochs)

        def prepare_epochs(epochs):
            # A tool for prepare epochs
            epochs = epochs[EVENTS]
            epochs.apply_baseline(BASELINE)
            return epochs.crop(CROP[0], CROP[1])

        # Xdawn enhance
        print('Xdawn --------------------------------')
        # print('Doing nothing')
        enhancer = Enhancer(train_epochs=train_epochs,
                            test_epochs=test_epochs)
        train_epochs, test_epochs = enhancer.fit_apply()

        # CNN denoise
        print('Denoise ------------------------------')

        # 3
        denoiser = Denoiser()
        noise_epochs = train_epochs['3'].copy()
        noise_epochs.apply_baseline(BASELINE)
        denoiser.fit(noise_epochs=noise_epochs)
        train_epochs, _ = denoiser.transform(train_epochs[EVENTS].copy(),
                                             labels=[1])

        noise_epochs = test_epochs['3'].copy()
        noise_epochs.apply_baseline(BASELINE)
        denoiser.fit(noise_epochs=noise_epochs)
        test_epochs, _ = denoiser.transform(test_epochs[EVENTS].copy(),
                                            labels=[1, 2],
                                            allowed_r_min=1.3)

        # for e in ['1', '2']:
        #     print(e)
        #     train_epochs[e].average().plot_joint()
        # stophere

        # Prepare epochs
        print('Prepare ------------------------------')
        train_epochs = prepare_epochs(train_epochs)
        test_epochs = prepare_epochs(test_epochs)

        # Get array
        print('Get array ----------------------------')
        X_train, y_train = get_X_y(train_epochs)
        X_test, y_test = get_X_y(test_epochs)

        # Training
        print('Training -----------------------------')
        clf = svm.SVC(gamma='scale', kernel='rbf', class_weight='balanced')
        pipeline = make_pipeline(Vectorizer(), StandardScaler(), clf)
        pipeline.fit(X_train, y_train)

        # Testing
        print('Testing ------------------------------')
        y_pred = pipeline.predict(X_test)
        # y_pred = y_test

        # Record
        print('Recording ----------------------------')
        predicts.append(dict(y_test=y_test,
                             y_pred=y_pred))

    with open(os.path.join(results_dir,
                           f'{name}.json'), 'wb') as f:
        pickle.dump(predicts, f)

        pass


# %%

for idx in range(1, 11):
    name = f'MEG_S{idx:02d}'
    process = multiprocessing.Process(target=mvpa, args=(name,))
    process.start()
    # mvpa(name)

# %%
