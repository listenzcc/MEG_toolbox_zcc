# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
import traceback


# %%


class Summary(object):
    def __init__(self):
        columns = ['Folder', 'Subject', 'Model',
                   'Precision', 'Recall', 'F1score', 'Accuracy']
        self.frame = pd.DataFrame(columns=columns)

    def append(self, folder, subject, model, y_true, y_pred):
        report = metrics.classification_report(y_true=y_true,
                                               y_pred=y_pred,
                                               output_dict=True)
        Accuracy = metrics.balanced_accuracy_score(y_true=y_true,
                                                   y_pred=y_pred)

        if '1.0' in report:
            report['1'] = report['1.0']

        Precision = report['1']['precision']
        Recall = report['1']['recall']
        F1score = report['1']['f1-score']

        self.frame = self.frame.append(dict(Folder=folder,
                                            Subject=subject,
                                            Model=model,
                                            Precision=Precision,
                                            Recall=Recall,
                                            F1score=F1score,
                                            Accuracy=Accuracy
                                            ),
                                       ignore_index=True)

    def get(self):
        return self.frame.copy()


summary = Summary()
# %%
folder = 'svm_3classes'
subject = 'S01'
model = 'EEG'

# %%
for model in ['EEG', 'MEG']:
    for folder in ['svm_2classes',
                   'svm_3classes',
                   'eegnet_2classes',
                   'eegnet_3classes',
                   'svm_2classes_meg64',
                   'svm_3classes_meg64',
                   'eegnet_2classes_meg64',
                   'eegnet_3classes_meg64']:
        for subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
            filepath = os.path.join(folder, f'{model}_{subject}.json')

            try:
                df = pd.read_json(filepath)
            except:
                continue

            y_true = np.concatenate(df.y_true.to_list())
            y_pred = np.concatenate(df.y_pred.to_list())

            if 'y_prob' in df:
                y_prob = np.concatenate(df.y_prob.to_list())[:, 0]
                y_pred = y_pred * 0 + 2
                y_pred[y_prob > 0.9] = 1

            y_true[y_true != 1] = 2
            y_pred[y_pred != 1] = 2

            summary.append(folder, subject, model, y_true, y_pred)

model = 'EEG'
folder = 'eegnet_3classes_mix'
for subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
    filepath = os.path.join(folder, f'{subject}_mix.json')
    df = pd.read_json(filepath)

    y_true = np.concatenate(df.y_true_eeg.to_list())
    y_prob = np.concatenate(df.y_prob_eeg.to_list())[:, 0]
    y_pred = y_prob * 0 + 2
    y_pred[y_prob > 0.9] = 1
    y_true[y_true != 1] = 2
    y_pred[y_pred != 1] = 2

    summary.append(folder, subject, model, y_true, y_pred)


# %%
df = summary.get()
df

# %%
for e in df.Folder.unique():
    print(e)
    print(df.query(f'Folder=="{e}"').describe())
    print()

# %%
plt.style.use('ggplot')
fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
axes = axes.ravel()
for j, name in enumerate(['Precision', 'Recall', 'F1score', 'Accuracy']):
    ax = axes[j]
    sns.boxplot(data=df, x='Folder', y=name, hue='Model', ax=ax)
    ax.set_title(name)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha='left')
fig.tight_layout()

# %%
