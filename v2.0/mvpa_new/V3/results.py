# %%
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

# %%


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

    def parse(self, res, name='--'):
        t = time.time()
        print(f'Start {name}.')
        crop_pred_y = res['crop_pred_y']
        test_y = res['test_y']
        report = metrics.classification_report(y_true=test_y,
                                               y_pred=crop_pred_y,
                                               output_dict=True)
        passed = time.time() - t
        print(f'Done {name}, {passed:2.4f}.')

        return pd.Series(report['1'], name=name[:9])

    def add(self, results_folder):
        self.folder = results_folder
        for name in os.listdir(results_folder):
            print(name)
            res = pickle.load(open(os.path.join(results_folder, name), 'rb'))
            obj = self.parse(res, name)
            self.append(obj)


class SlideFrame(MyFrame):
    def __init__(self):
        super(SlideFrame, self).__init__()

    def parse(self, res, name='--'):
        t = time.time()
        print(f'Start {name}.')
        slide_pred_y = res['slide_pred_y']
        test_y = res['test_y']

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

crop_frame1 = CropFrame()
crop_frame1.add('Results_crop')

crop_frame2 = CropFrame()
crop_frame2.add('Results_raw_crop')

# %%
slide_frame = SlideFrame()
t = time.time()
slide_frame.add('Results_raw_slide')
print('------------- {}'.format(time.time() - t))


# %%

print(crop_frame1.folder)
display(crop_frame1.frame.describe())

print(crop_frame2.folder)
display(crop_frame2.frame.describe())

# %%
frame = slide_frame.frame
times = slide_frame.times
plt.style.use('ggplot')
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for key in frame:
    data = np.concatenate(frame[key])
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    ax.plot(times, mean, label=key)
    ax.fill_between(times, mean+std, mean-std, alpha=0.2)

# ax.set_xlim((times[0], times[-1]))
ax.set_title(f'Sliding Scores on {slide_frame.count}')
ax.legend()
fig.tight_layout()

# %%
