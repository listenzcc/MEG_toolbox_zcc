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

summary = pd.DataFrame(
    columns=['subject', 'folder', 'method', 'recall', 'precision', 'score', 'accuracy', 'confusion'])

for subject in ['MEG_S01', 'MEG_S02', 'MEG_S03', 'MEG_S04', 'MEG_S05', 'MEG_S06', 'MEG_S07', 'MEG_S08', 'MEG_S09', 'MEG_S10']:
    print('-' * 80)
    print(subject)

    # Read json file
    path_svm = os.path.join('svm_3classes', f'{subject}.json')
    path_net = os.path.join('no_xdawn_eegnet_3classes', f'{subject}.json')
    frame_svm = pd.read_json(path_svm)
    frame_net = pd.read_json(path_net)

    # Get y lists from frames
    y_true_list = frame_svm.y_true.to_list()
    y_pred_svm_list = frame_svm.y_pred.to_list()
    y_prob_net_list = frame_net.y_prob.to_list()

    for folder, ys in enumerate(zip(y_true_list, y_pred_svm_list, y_prob_net_list)):
        # Get y_true, y_pred and y_prob
        y_true = np.array(ys[0])
        y_pred_svm = np.array(ys[1])
        y_prob_net = np.array(ys[2])

        # Calculate y_pred of eegnet
        y_pred_net = y_true * 0 + 2
        y_pred_net[y_prob_net[:, 0] > 0.9] = 1
        y_pred_net[np.all([y_prob_net[:, 0] < 0.9,
                           y_prob_net[:, 1] < y_prob_net[:, 2]],
                          axis=0)] = 4

        confusion_net = metrics.confusion_matrix(
            y_true, y_pred_net, normalize='true')
        confusion_svm = metrics.confusion_matrix(
            y_true, y_pred_svm, normalize='true')

        # Calcute y_pred of joint
        # y_pred_joint = y_true * 0 + 2
        # for j in range(y_pred_joint.shape[0]):
        #     if all([
        #         y_prob_net[j, 0] > 0.9,
        #     ]):
        #         y_pred_joint[j] = 1

        #     if all([
        #         _y_pred_svm[j] == 1,
        #     ]):
        #         y_pred_joint[j] = 1

        for method, y_pred, confusion in zip(['SVM', 'Net'],
                                             [y_pred_svm, y_pred_net],
                                             [confusion_svm, confusion_net]):
            _y_true = y_true.copy()
            _y_true[_y_true == 4] = 2
            _y_pred = y_pred.copy()
            _y_pred[_y_pred == 4] = 2

            report = metrics.classification_report(
                y_pred=_y_pred, y_true=_y_true, output_dict=True
            )
            summary = summary.append(pd.Series(dict(
                subject=subject,
                method=method,
                folder=folder,
                recall=report['1']['recall'],
                precision=report['1']['precision'],
                score=report['1']['f1-score'],
                accuracy=report['accuracy'],
                confusion=confusion,
            )), ignore_index=True)

summary

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
    formula = f'{name} ~ subject + method + folder'
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
df['subject'] = df['subject'].map(lambda x: x[4:])

# Plot scatters across subjects
fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=300)
axes = np.ravel(axes)
for j, name in enumerate(['recall', 'precision', 'score', 'accuracy']):
    ax = axes[j]
    sns.boxplot(x='subject', y=name, hue='method', data=df, ax=ax)
    # sns.swarmplot(x='subject', y=name, hue='method', data=summary, ax=ax)
    ax.set_title(name.title())
    ax.set_ylabel('')
fig.tight_layout()
fig.savefig('scatters_subjects.png')


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
    fig.savefig('confusion_matrix.png')

# %%
help(anova.to_html)
# %%
