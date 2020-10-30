# %%
import seaborn as sns
import os
import numpy as np
import pandas as pd
import traceback
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# %%
# Load and generate useful dataset
# Summary of scores
summary = pd.DataFrame(
    columns=['name', 'folder', 'method', 'recall', 'precision', 'score', 'accuracy', 'confusion'])

# Dataframe of roc curve
df_roc = pd.DataFrame()


MODEL = 'EEG'
for subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
    name = f'{MODEL}_{subject}'

    # Load and generate for 10 names
    print('-' * 80)
    print(name)

    # Read json file
    path_svm = os.path.join('svm_3classes', f'{name}.json')
    frame_svm = pd.read_json(path_svm)

    # Get y lists from frames
    y_true_list = frame_svm.y_true.to_list()
    y_pred_svm_list = frame_svm.y_pred.to_list()

    for folder, ys in enumerate(zip(y_true_list, y_pred_svm_list)):
        # Get y_true, y_pred and y_prob
        y_true = np.array(ys[0])
        y_pred = np.array(ys[1])

        confusion = metrics.confusion_matrix(
            y_true, y_pred, normalize='true')

        report = metrics.classification_report(
            y_pred=y_pred, y_true=y_true, output_dict=True
        )
        summary = summary.append(pd.Series(dict(
            name=name,
            method='SVM',
            folder=folder,
            recall=report['1']['recall'],
            precision=report['1']['precision'],
            score=report['1']['f1-score'],
            accuracy=report['accuracy'],
            confusion=confusion,
        )), ignore_index=True)

summary

summary.describe()
# %%
# Compare scores of SVM and Net
# [ss] is the temporal dict for SVM and Net scores
ss = dict()
# Calculate averaged confusion matrix for latter use
confusion_matrixes = dict()
# Restore the anova table in type of dataframe
anovas = dict()

# Prepare data
for method in ['Net', 'SVM']:
    ss[method] = summary.loc[summary.method == method]
    print(method)
    print(ss[method].describe())
    print()

    confusion_matrixes[method] = np.mean(
        ss[method].confusion.to_list(), axis=0)

# Anova analysis
for name in ['recall', 'precision', 'score', 'accuracy']:
    print('-' * 80)
    print(name)
    formula = f'{name} ~ name + method + folder'
    model = ols(formula, data=summary).fit()
    anova = anova_lm(model)
    anovas[name] = anova
    display(anova)
    print()

# T-test
for col in ['recall', 'precision', 'score', 'accuracy']:
    print(col)
    print(stats.ttest_rel(ss['Net'][col], ss['SVM'][col]))
    print()

# Write scores and anova tables into html
with open('anova_tables.html', 'w') as f:
    for method in ss:
        describe = ss[method].describe()
        f.writelines(['<div>',
                      f'<h2>{method}</h2>'])
        describe.to_html(f)
        f.writelines(['</div><br>'])

    for name in anovas:
        anova = anovas[name]
        f.writelines(['<div>',
                      f'<h2>{name}</h2>'])
        anova.to_html(f)
        f.writelines(['</div><br>'])

# %% Plot scatters chart
df = summary.copy()
df['name'] = df['name'].map(lambda x: x[4:])

# Plot scatters across names
fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=300)
axes = np.ravel(axes)
for j, name in enumerate(['recall', 'precision', 'score', 'accuracy']):
    ax = axes[j]
    sns.boxplot(x='name', y=name, hue='method', data=df, ax=ax)
    # sns.swarmplot(x='name', y=name, hue='method', data=summary, ax=ax)
    ax.set_title(name.title())
    ax.set_ylabel('')

fig.tight_layout()
# fig.savefig('scatters_names.png')


# %%
# Generate and save confusion matrixes
cmap = 'Blues'
ticklabels = ['Target', 'Far', 'Near']
fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)

# for j, item in enumerate(zip([confusion_svm, confusion_net], ['SVM', 'Net'])):
for j, method in enumerate(confusion_matrixes):
    cmat = confusion_matrixes[method]  # Confuse matrix
    ax = axes[j]  # Specify axis

    # Formulate DataFrame
    df = pd.DataFrame(cmat)
    df.index = ticklabels
    df.columns = ticklabels

    # Draw on axis
    hmap = sns.heatmap(df, annot=True, cmap=cmap, ax=ax)
    ax.set_title(f'{method} confusion matrix')
    ax.set_xlabel('Predict')
    ax.set_ylabel('Truth')

fig.tight_layout()
# fig.savefig('confusion_matrix.png')
