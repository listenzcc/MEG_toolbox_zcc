# %%
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

# %%
for name in ['MEG_S02', 'MEG_S03', 'MEG_S04', 'MEG_S05']:
    print('-' * 80)
    print(name)

    try:
        frame = pd.read_json(f'./svm_2classes/{name}.json')
    except:
        continue

    y_true = np.concatenate(frame.y_true.to_list()[0:1])
    y_pred = np.concatenate(frame.y_pred.to_list()[0:1])
    print('Classification report\n',
          metrics.classification_report(y_pred=y_pred, y_true=y_true))
    print('Confusion matrix\n',
          metrics.confusion_matrix(y_pred=y_pred, y_true=y_true))

    try:
        frame = pd.read_json(f'./svm_3classes/{name}.json')
    except:
        continue

    y_true = np.concatenate(frame.y_true.to_list()[0:1])
    y_pred = np.concatenate(frame.y_pred.to_list()[0:1])
    print('Classification report\n',
          metrics.classification_report(y_pred=y_pred, y_true=y_true))
    print('Confusion matrix\n',
          metrics.confusion_matrix(y_pred=y_pred, y_true=y_true))

    break
# %%
