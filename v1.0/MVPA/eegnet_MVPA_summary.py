# %%
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
from pprint import pprint
from sklearn import metrics

# %%


def fuck_report(report):
    # I don't know why report has to be 2-layers dictionary,
    # this method is to make it human useable.
    fucked_report = dict()
    for key1 in report:
        if isinstance(report[key1], dict):
            for key2 in report[key1]:
                fucked_report[f'{key1}-{key2}'] = report[key1][key2]
        else:
            fucked_report[key1] = report[key1]

    keys = [e for e in fucked_report.keys()]
    for key in keys:
        if key.endswith('support'):
            fucked_report.pop(key)

    return fucked_report


def get_scores(y_true, y_pred):
    report = metrics.classification_report(y_true=y_true,
                                           y_pred=y_pred,
                                           output_dict=True)
    return fuck_report(report)


def clean(table):
    for col in table.columns:
        if any([col.startswith(e) for e in ['2.0', '0.0']]):
            table.pop(col)


# %%
table_eegnet = pd.DataFrame()
table_svm = pd.DataFrame()

for idx in range(1, 11):
    running_name = f'MEG_S{idx:02d}'
    with open(os.path.join('results_eegnet',
                           f'{running_name}_eegnet.pkl'), 'rb') as f:
        content_eegnet = pickle.load(f)

    with open(os.path.join('results', f'{running_name}_segment.pkl'), 'rb') as f:
        content_svm = pickle.load(f)

    scores_eegnet = get_scores(y_true=content_eegnet['y_all'],
                               y_pred=np.round(content_eegnet['y_pred']))

    scores_svm = get_scores(y_true=content_svm['y_all'],
                            y_pred=np.round(content_svm['e_y_pred']))

    # display(scores_eegnet, scores_svm)

    table_eegnet = table_eegnet.append(pd.Series(scores_eegnet,
                                                 name=running_name))
    table_svm = table_svm.append(pd.Series(scores_svm,
                                           name=running_name))

clean(table_eegnet)
clean(table_svm)

display(table_eegnet)
display(table_svm)

# %%

print('EEGNet')
display(table_eegnet.describe())
print('SVM')
display(table_svm.describe())

# %%
for col in table_eegnet.columns:
    s, p = stats.ttest_rel(a=table_svm[col], b=table_eegnet[col], axis=0)
    print(col, s, p)

# %%
