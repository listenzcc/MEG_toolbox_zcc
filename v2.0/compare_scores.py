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
    columns=['subject', 'folder', 'method', 'recall', 'precision', 'score', 'accuracy', 'confusion'])


for subject in ['MEG_S01', 'MEG_S02', 'MEG_S03', 'MEG_S04', 'MEG_S05', 'MEG_S06', 'MEG_S07', 'MEG_S08', 'MEG_S09', 'MEG_S10']:
    # Load and generate for 10 subjects
    print('-' * 80)
    print(subject)

    # Read json file
    path_svm2 = os.path.join('svm_2classes', f'{subject}.json')
    path_svm3 = os.path.join('svm_3classes', f'{subject}.json')
    path_net2 = os.path.join('no_xdawn_eegnet_2classes', f'{subject}.json')
    path_net3 = os.path.join('no_xdawn_eegnet_3classes', f'{subject}.json')

    frame_svm2 = pd.read_json(path_svm2)
    frame_svm3 = pd.read_json(path_svm3)
    frame_net2 = pd.read_json(path_net2)
    frame_net3 = pd.read_json(path_net3)

    # Get y lists from frames
    y_true_list = frame_svm3.y_true.to_list()
    y_pred_svm2_list = frame_svm2.y_pred.to_list()
    y_pred_svm3_list = frame_svm3.y_pred.to_list()
    y_prob_net2_list = frame_net2.y_prob.to_list()
    y_prob_net3_list = frame_net3.y_prob.to_list()

    for folder, ys in enumerate(zip(y_true_list,
                                    y_pred_svm2_list,
                                    y_pred_svm3_list,
                                    y_prob_net2_list,
                                    y_prob_net3_list)):

        # Get y_true, y_pred and y_prob
        y_true = np.array(ys[0])
        y_pred_svm2 = np.array(ys[1])
        y_pred_svm3 = np.array(ys[2])
        y_prob_net2 = np.array(ys[3])
        y_prob_net3 = np.array(ys[4])

        # Calculate y_pred of eegnet
        y_pred_net2 = y_true * 0 + 2
        y_pred_net3 = y_true * 0 + 2
        y_pred_net2[y_prob_net2[:, 0] > 0.9] = 1
        y_pred_net3[y_prob_net3[:, 0] > 0.9] = 1

        for method, y_pred in zip(['SVM2', 'SVM3', 'Net2', 'Net3'],
                                  [y_pred_svm2, y_pred_svm3,
                                   y_pred_net2, y_pred_net3],
                                  ):
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
            )), ignore_index=True)

summary

# %%
with open('compare_scores.html', 'w') as f:
    for method in ['SVM2', 'SVM3', 'Net2', 'Net3']:
        print(method)
        s = summary.loc[summary.method == method]
        display(s.describe())
        f.writelines(['<div>'])
        f.writelines(['<h2>', method, '</h2>'])
        s.describe().to_html(f)
        f.writelines(['</div>'])

print('Done')

# %%
