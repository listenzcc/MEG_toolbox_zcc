# %%
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

# %%


def watch_folder(folder):
    try:
        frame = pd.read_json(f'./{folder}/{name}.json')
    except:
        return

    y_true = np.concatenate(frame.y_true.to_list())
    y_pred = np.concatenate(frame.y_pred.to_list())
    print('Classification report\n',
          metrics.classification_report(y_pred=y_pred, y_true=y_true))
    print('Confusion matrix\n',
          metrics.confusion_matrix(y_pred=y_pred, y_true=y_true))


for name in ['MEG_S02', 'MEG_S03', 'MEG_S04', 'MEG_S05']:
    print('-' * 80)
    print(name)

    watch_folder('svm_3classes')
    watch_folder('xdawn_eegnet')
    watch_folder('no_xdawn_eegnet')

# %%
