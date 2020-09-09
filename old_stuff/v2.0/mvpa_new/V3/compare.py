# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# %%
frame1 = pd.read_html('results_denoise.html')[0]
frame2 = pd.read_html('../results_tsne.html')[0]
frame2['method'] = 'manifolder'
frame2['recall'] = frame2['_recall']
frame2['precision'] = frame2['_precision']
frame2['f1-score'] = 2 / (1 / frame2['recall'] + 1 / frame2['precision'])
frame2['f1score'] = 2 / (1 / frame2['recall'] + 1 / frame2['precision'])
frame2['cv'] = frame2['Unnamed: 0'].map(lambda e: e[-2:])
frame2['subject'] = frame2['Unnamed: 0'].map(lambda e: e[:7])

frame1 = frame1[['subject', 'cv', 'method',
                 'recall', 'precision', 'f1score', 'f1-score']]
frame2 = frame2[['subject', 'cv', 'method',
                 'recall', 'precision', 'f1score', 'f1-score']]
display(frame1, frame2)

# %%
frame = pd.concat([frame1, frame2])
frame.index = range(len(frame))
frame['cv'] = frame['cv'].map(lambda e: str(e))
display(frame)

# %%
for value in ['f1score', 'recall', 'precision']:
    print('-' * 80)
    print(value)
    formula = f'{value} ~ subject + method + cv'
    model = ols(formula, data=frame).fit()
    anova = anova_lm(model)
    anova.to_html(f'anova_{value}.html')
    display(anova)

# %%
newframe = frame.groupby(['method', 'subject']).mean()
newframe.pop('f1score')

display(newframe)

plt.style.use('ggplot')
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for j, value in enumerate(['recall', 'precision', 'f1-score']):
    ax = axes[j]
    ax.boxplot([newframe.loc['raw'][value],
                newframe.loc['manifolder'][value]],
               labels=['Denoise', 'Manifolder'],
               widths=0.4)
    ax.set_title(value)
fig.tight_layout()
print('')

# %%
frame
# %%
