# %%
import itertools
import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import threading
import matplotlib.pyplot as plt

from collections import defaultdict

from sklearn import metrics
from scipy import stats

import statsmodels.api as sm
# from statsmodels.formula.api import ols

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
# %%


def select_labels(y):
    # return [e for e in range(len(y))]

    n1 = len(np.where(y == 1)[0])
    n2 = len(np.where(y == 2)[0])
    ratio = n1 / n2 * 2

    # Selection
    selects = []
    for j, _y in enumerate(y):
        if _y == 1:
            [selects.append(j + e) for e in [-2, -1, 0, 1, -2]]
            continue

        if _y == 2:
            if random.random() < ratio:
                selects.append(j)
            continue

    # Return
    return selects


# def compute_scores(y_true, y_pred):
#     tp = 0
#     num_1 = len(np.where(y_true == 1)[0])
#     num_pos = 0
#     for j, p in enumerate(y_pred):
#         if p == 1:
#             num_pos += 1
#         else:
#             continue

#         try:
#             if 1 in y_true[j-1:j+2]:
#                 tp += 1
#         except:
#             pass

#     try:
#         recall = tp / num_1
#     except:
#         recall = 0

#     try:
#         precision = tp / num_pos
#     except:
#         precision = 0

#     return dict(_recall=recall, _precision=precision)


class MyFrame():
    def __init__(self):
        self.frame = pd.DataFrame()

    def append(self, obj):
        self.frame = self.frame.append(obj)

    @property
    def count(self):
        return len(self.frame)


class CropFrame(MyFrame):
    def __init__(self):
        super(CropFrame, self).__init__()

    def parse(self, res, name, method):
        t = time.time()
        print(f'Start {name}.')
        crop_pred_y = res['crop_pred_y']
        test_y = res['test_y']

        selects = select_labels(test_y)
        test_y = test_y[selects]
        crop_pred_y = crop_pred_y[selects]

        report = metrics.classification_report(y_true=test_y,
                                               y_pred=crop_pred_y,
                                               output_dict=True)
        passed = time.time() - t
        print(f'Done {name}, {passed:2.4f}.')

        report['1']['method'] = method
        report['1']['subject'] = name[:7]
        report['1']['cv'] = name[7:-9]
        report['1']['f1score'] = report['1']['f1-score']

        # _scores = compute_scores(y_pred=crop_pred_y, y_true=test_y)
        # report['1']['_recall'] = _scores['_recall']
        # report['1']['_precision'] = _scores['_precision']

        return pd.Series(report['1'], name=name[:9])

    def add(self, results_folder, method='-'):
        self.folder = results_folder
        for name in os.listdir(results_folder):
            print(name)
            res = pickle.load(open(os.path.join(results_folder, name), 'rb'))
            obj = self.parse(res, name, method)
            self.append(obj)


class SlideFrame(MyFrame):
    def __init__(self):
        super(SlideFrame, self).__init__()

    def parse(self, res, name='--'):
        t = time.time()
        print(f'Start {name}.')
        slide_pred_y = res['slide_pred_y']
        test_y = res['test_y']

        selects = select_labels(test_y)
        test_y = test_y[selects]
        slide_pred_y = slide_pred_y[selects]

        results = defaultdict(list)
        for j in range(141):
            report = metrics.classification_report(y_true=test_y,
                                                   y_pred=slide_pred_y[:, j],
                                                   output_dict=True)
            _report = report['1']
            for key in _report:
                if key == 'support':
                    continue
                results[key].append(_report[key])

        for key in results:
            results[key] = np.array(results[key])[np.newaxis, :]

        passed = time.time() - t
        print(f'Done {name}, {passed:2.4f}.')

        return pd.Series(results, name=name[:9])

    def add(self, results_folder):
        self.folder = results_folder
        for name in os.listdir(results_folder):
            print(name)
            res = pickle.load(open(os.path.join(results_folder, name), 'rb'))
            self.times = res['times']
            obj = self.parse(res, name)
            self.append(obj)


# %%
# Load crop results
# It is fast
crop_frame = CropFrame()
crop_frame.add('Results_crop', method='denoise')
crop_frame.add('Results_raw_crop', method='raw')
print('All done.')

# %%
# Load slide results
# It is slow
slide_frame = SlideFrame()
slide_frame.add('Results_slide')

raw_slide_frame = SlideFrame()
raw_slide_frame.add('Results_raw_slide')


# %%

print(crop_frame.folder)
frame = crop_frame.frame
frame.to_html('results_denoise.html')
display(frame)
for method in ['denoise', 'raw']:
    print(method)
    display(frame.loc[frame['method'] == method].describe())

for value in ['f1score', 'recall', 'precision']:
    print('-' * 80)
    print(value)
    formula = f'{value} ~ subject + method + cv'
    model = ols(formula, data=frame).fit()
    anova = anova_lm(model)
    anova.to_html(f'anova_{value}.html')
    display(anova)

f1 = frame.loc[frame['method'] == 'denoise']
f2 = frame.loc[frame['method'] == 'raw']
for value in ['f1score', 'recall', 'precision']:
    print(value)
    print(stats.ttest_rel(f2[value], f1[value]))

# %%


def plot(frame, times, ax=None, title='Sliding Scores'):
    plt.style.use('ggplot')
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    for key in frame:
        data = np.concatenate(frame[key])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        ax.plot(times, mean, label=key)
        ax.fill_between(times, mean+std, mean-std, alpha=0.2)

    # ax.set_xlim((times[0], times[-1]))
    ax.set_title(title)
    ax.legend()
    # ax.set_yticks([])
    ax.set_ylim([0.0, 1.0])
    ax.get_figure().tight_layout()
    # return fig


fig, axes = plt.subplots(2, 1, figsize=(8, 8))
plot(slide_frame.frame, slide_frame.times, axes[0], title='Denoise')
plot(raw_slide_frame.frame, raw_slide_frame.times, axes[1], title='Raw')

# %%
newframe = frame.groupby(['method', 'name']).mean()
newframe.pop('support')
newframe.pop('f1score')
display(newframe)

grouped = newframe.groupby(level='method')
grouped.boxplot(subplots=False, rot=45)

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for j, value in enumerate(['recall', 'precision', 'f1-score']):
    ax = axes[j]
    ax.boxplot([newframe.loc['denoise'][value],
                newframe.loc['raw'][value]],
               labels=['Raw', 'Denoise'],
               widths=0.4)
    ax.set_title(value)
fig.tight_layout()
print('')

# %%
