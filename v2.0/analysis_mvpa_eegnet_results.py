# %%
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
    columns=['subject', 'folder', 'method', 'recall', 'precision', 'score', 'accuracy', 'confuse'])

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

        confuse_net = metrics.confusion_matrix(
            y_true, y_pred_net, normalize='true')
        confuse_svm = metrics.confusion_matrix(
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

        for method, y_pred, confuse in zip(['svm', 'net'],
                                           [y_pred_svm, y_pred_net],
                                           [confuse_svm, confuse_net]):
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
                confuse=confuse,
            )), ignore_index=True)

summary

# %%
ss = dict()
# for method in ['net', 'svm', 'joint']:
for method in ['net', 'svm']:
    ss[method] = summary.loc[summary.method == method]
    print(method)
    print(ss[method].describe())
    print(np.mean(ss[method].confuse.to_list(), axis=0))
    print()

for col in ['recall', 'precision', 'score', 'accuracy']:
    print(col)
    print(stats.ttest_rel(ss['net'][col], ss['svm'][col]))
    print()

for value in ['recall', 'precision', 'score', 'accuracy']:
    print('-' * 80)
    print(value)
    formula = f'{value} ~ subject + method + folder'
    model = ols(formula, data=summary).fit()
    anova = anova_lm(model)
    display(anova)
    print()

# %%
