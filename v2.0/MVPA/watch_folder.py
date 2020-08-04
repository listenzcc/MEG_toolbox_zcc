# %%
import os
import pickle
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

# %%


def parse_dict(report, table):
    groups = ['1', 'macro avg', 'weighted avg']
    scores = ['recall', 'precision', 'f1-score']
    output = dict()
    for grp in groups:
        for scr in scores:
            output[f'{grp}-{scr}'] = report[grp][scr]

    output['accuracy'] = report['accuracy']

    return table.append(output, ignore_index=True)


# %%

table = pd.DataFrame()

results_dir = os.path.join('.', 'SVM_baseline')

for name in os.listdir(results_dir):
    with open(os.path.join(results_dir, name), 'rb') as f:
        predicts = pickle.load(f)

    for pred in predicts:
        report = metrics.classification_report(y_true=pred['y_test'],
                                               y_pred=pred['y_pred'],
                                               output_dict=True)
        table = parse_dict(report, table)

# table

display(table.describe())

# %%
predicts

# %%
