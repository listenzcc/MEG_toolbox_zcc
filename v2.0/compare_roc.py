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
summary = pd.DataFrame()

dfs = []
for subject in ['MEG_S01', 'MEG_S02', 'MEG_S03', 'MEG_S04', 'MEG_S05', 'MEG_S06', 'MEG_S07', 'MEG_S08', 'MEG_S09', 'MEG_S10']:
    # Load and generate for 10 subjects
    print('-' * 80)
    print(subject)

    # Read json file
    path_net2 = os.path.join('no_xdawn_eegnet_2classes',
                             f'{subject}.json')
    path_net3 = os.path.join('no_xdawn_eegnet_3classes',
                             f'{subject}.json')
    frame_net2 = pd.read_json(path_net2)
    frame_net3 = pd.read_json(path_net3)

    # Get y lists from frames

    y_true_list = frame_net2.y_true.to_list()
    y_prob_net2_list = frame_net2.y_prob.to_list()
    y_prob_net3_list = frame_net3.y_prob.to_list()

    for y_prob_net_list, method in zip([y_prob_net2_list, y_prob_net3_list],
                                       ['Binary', 'Ternary']):

        y_prob_net = np.concatenate(y_prob_net_list, axis=0)[:, 0]
        y_true = np.concatenate(y_true_list, axis=0)
        y_true[y_true != 1] = 0

        y_pred = y_prob_net > 0.9
        kwargs = dict(
            y_true=y_true,
            y_pred=y_pred
        )
        recall = metrics.recall_score(**kwargs)
        precision = metrics.precision_score(**kwargs)
        f1_score = metrics.f1_score(**kwargs)

        roc_score = metrics.roc_auc_score(y_true=y_true, y_score=y_prob_net)
        summary = summary.append(
            pd.Series(dict(subject=subject,
                           method=method,
                           roc=roc_score,
                           recall=recall,
                           precision=precision,
                           f1_score=f1_score)), ignore_index=True)

        fpr, tpr, thresholds = metrics.roc_curve(
            y_true=y_true, y_score=y_prob_net)

        df = pd.DataFrame()
        df['tpr'] = tpr
        df['fpr'] = fpr
        df['subject'] = subject
        df['thresholds'] = thresholds
        df['method'] = method
        dfs.append(df)

# Dataframe of roc curve
df_roc = pd.concat(dfs)
display(df_roc)
display(summary)

with open('compare_eegnets.html', 'w') as f:
    for method in ['Binary', 'Ternary']:
        print(method)
        description = summary.loc[summary.method == method].describe()
        display(description)
        f.writelines(['<div>'])
        f.writelines(['<h2>', method, '</h2>'])
        description.to_html(f)
        f.writelines(['</div>'])

# %%
plt.style.use('ggplot')
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

sns.lineplot(data=df_roc, y='tpr',
             x='thresholds', hue='subject', style='method', ax=axes[0, 0])

sns.lineplot(data=df_roc, y='fpr',
             x='thresholds', hue='subject', style='method', ax=axes[0, 1])

sns.lineplot(data=df_roc, y='tpr',
             x='thresholds', hue='subject', style='method', ax=axes[1, 0], legend=False)

sns.lineplot(data=df_roc, y='fpr',
             x='thresholds', hue='subject', style='method', ax=axes[1, 1], legend=False)

# sns.boxplot(data=summary, y='recall',
#             x='subject', hue='method', ax=axes[1])

ax = axes[0, 0]
ax.set_title('TPR')
ax.set_xlim((0, 1))
ax.invert_xaxis()
ax.set_ylim((0, 1))

ax = axes[0, 1]
ax.set_title('FPR')
ax.set_xlim((0, 1))
ax.invert_xaxis()
ax.set_ylim((0, 1))

ax = axes[1, 0]
ax.set_title('TPR')
ax.set_xlim((0, 1))
ax.invert_xaxis()
ax.set_ylim((0.6, 1))

ax = axes[1, 1]
ax.set_title('FPR')
ax.set_xlim((0, 1))
ax.invert_xaxis()
ax.set_ylim((0, 0.2))

fig.tight_layout()
fig.savefig('ROC.png')
# %%
# Draw an empty plot only for legends
plt.style.use('ggplot')
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
df = df_roc.iloc[:3]
df.subject = ['Target', 'Far', 'Near']
sns.lineplot(data=df, y='fpr',
             x='thresholds', hue='subject', ax=ax)
fig.savefig('legend.png')
# %%
