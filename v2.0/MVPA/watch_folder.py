# %%
import os
import pickle
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

import statsmodels.api as sm
from statsmodels.formula.api import ols

# %%


def parse_dict(report, table, name, method):
    groups = ['1', 'macro avg', 'weighted avg']
    scores = ['recall', 'precision', 'f1-score']
    output = dict()
    for grp in groups:
        for scr in scores:
            output[f'{grp}-{scr}'] = report[grp][scr]

    output['accuracy'] = report['accuracy']
    output['Name'] = name.split('.')[0]
    output['Method'] = method

    return table.append(output, ignore_index=True)


def write(content, fname='Report.md', refresh=False):
    # Write [content] into file of [fname]
    # Refresh [fname] if [refresh]
    if refresh:
        with open(fname, 'w') as f:
            f.writelines([])
        print(f'{fname} has been emptied.')

    # Make sure contents is a list
    if isinstance(content, str):
        # When [content] is a str
        contents = [content]
    else:
        # When [content] is iterable
        contents = [e for e in content]

    # Write
    with open(fname, 'a') as f:
        f.writelines(contents)
        f.writelines(['\n', '\n'])

    print(f'{fname} has been added.')

# %%


main_table = pd.DataFrame()

for folder in ['SVM_baseline',
               'SVM_xdawn',
               'SVM_denoise',
               'SVM_xdawn_denoise',
               'SVM_xdawn_denoise_1',
               'SVM_xdawn_denoise_2']:
    results_dir = os.path.join('.', folder)

    for name in os.listdir(results_dir):
        with open(os.path.join(results_dir, name), 'rb') as f:
            predicts = pickle.load(f)

        y_true = np.concatenate([e['y_test'] for e in predicts], axis=0)
        y_pred = np.concatenate([e['y_pred'] for e in predicts], axis=0)

        for pred in predicts:
            y_true = pred['y_test']
            y_pred = pred['y_pred']
            report = metrics.classification_report(y_true=y_true,
                                                   y_pred=y_pred,
                                                   output_dict=True)
            main_table = parse_dict(report, main_table, name, method=folder)

# %%
write('<h1>MVPA comparison</h1>', refresh=True)

for method in main_table['Method'].unique():
    print(method)
    display(main_table.loc[main_table['Method'] == method].describe())
    write([f'<h2>{method}</h2>',
           main_table.loc[main_table['Method'] == method].describe().to_html()])


# %%
main_table

# %%
anova_tables = dict()
for column in main_table.columns:  # ['1-recall', '1-f1-score', 'accuracy']:
    if column in ['Name', 'Method']:
        continue
    print(column)

    table = main_table[['Name', 'Method', column]]
    table.columns = ['Name', 'Method', 'Value']

    model = ols('Value ~ C(Name) + C(Method) + C(Name):C(Method)',
                data=table).fit()
    anova_tables[column] = sm.stats.anova_lm(model, typ=2)

for key in anova_tables:
    print(f'\n{key} ------------------------------')
    display(anova_tables[key])
    write([f'<h2>{key}</h2>',
           anova_tables[key].to_html()])


# %%
