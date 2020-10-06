# %%
import numpy as np
import pandas as pd
import traceback
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# %%


def watch_folder(folder, method='svm'):
    print(folder)
    try:
        frame = pd.read_json(f'./{folder}/{name}.json')
    except:
        traceback.print_exc()
        return

    y_true = np.concatenate(frame.y_true.to_list())
    y_true[y_true == 4] = 2

    if method == 'svm':
        y_pred = np.concatenate(frame.y_pred.to_list())
        y_pred[y_pred == 4] = 2
        print('Classification report\n',
              metrics.classification_report(y_pred=y_pred, y_true=y_true))
        print('Confusion matrix\n',
              metrics.confusion_matrix(y_pred=y_pred, y_true=y_true))
    else:
        y_prob = np.concatenate(frame.y_prob.to_list())

        y_pred = y_true * 0 + 2
        y_pred[y_prob[:, 0] > 0.8] = 1
        # y_pred[y_prob[:, 0] - y_prob[:, 1] - y_prob[:, 2] > 0] = 1

        print('Classification report\n',
              metrics.classification_report(y_pred=y_pred, y_true=y_true))
        print('Confusion matrix\n',
              metrics.confusion_matrix(y_pred=y_pred, y_true=y_true))
        y_true[y_true != 1] = 0
        print('ROC score\n',
              metrics.roc_auc_score(y_true, y_prob[:, 0]))

    return frame


# %%

for name in ['MEG_S01', 'MEG_S02', 'MEG_S03', 'MEG_S04', 'MEG_S05', 'MEG_S06', 'MEG_S07', 'MEG_S08', 'MEG_S09', 'MEG_S10']:
    print('-' * 80)
    print(name)

    watch_folder('svm_3classes', 'svm')
    # watch_folder('no_xdawn_eegnet_2classes', 'eegnet')
    watch_folder('no_xdawn_eegnet_3classes', 'eegnet')

# %%
