# %%
import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading

from collections import defaultdict

from sklearn import metrics

# %%
results_folder = os.path.join('Results')


frame = pd.DataFrame()

for name in os.listdir(results_folder):
    print(name)

    res = pickle.load(open(os.path.join(results_folder, name), 'rb'))
    slide_pred_y = res['slide_pred_y']
    test_y = res['test_y']
    times = res['times']

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

    se = pd.Series(results, name=name[:9])
    frame = frame.append(se)

# %%
frame

# %%
plt.style.use('ggplot')
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for key in frame:
    data = np.concatenate(frame[key])
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    ax.plot(times, mean, label=key)
    ax.fill_between(times, mean+std, mean-std, alpha=0.2)

# ax.set_xlim((times[0], times[-1]))
ax.set_title('Sliding Scores')
ax.legend()
fig.tight_layout()

# %%
