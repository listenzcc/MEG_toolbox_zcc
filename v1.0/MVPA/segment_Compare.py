# %%
import os
import json
import numpy as np
import scipy.stats
import pandas as pd

# %%
crops_table = dict(a=(0.2, 0.4),
                   b=(0.4, 0.6),
                   c=(0.6, 0.8),
                   d=(0.2, 0.8),
                   e=(0.0, 1.0))

# %%
with open(os.path.join('.', 'results.json'), 'r') as f:
    results = json.load(f)

with open(os.path.join('.', 'results_baseline.json'), 'r') as f:
    results_baseline = json.load(f)

# %%
frame = pd.DataFrame(columns=['ID',
                              'Range',
                              'Baseline',
                              'Improved',
                              'Stat'])

for crop_name in crops_table:
    if not crop_name == 'e':
        continue
    print(crop_name)
    crop_range = crops_table[crop_name]
    candidates = ['1.0-recall',
                  '1.0-f1-score',
                  'accuracy']

    score_name = 'balanced-accuracy'

    scores = (np.array(results[crop_name]['1.0-recall']) +
              np.array(results[crop_name]['1.0-precision'])) / 2
    results[crop_name][score_name] = scores

    scores = (np.array(results_baseline[crop_name]['1.0-recall']) +
              np.array(results_baseline[crop_name]['1.0-precision'])) / 2
    results_baseline[crop_name][score_name] = scores

    for score_name in results[crop_name]:
        scores = results[crop_name][score_name]
        scores_baseline = results_baseline[crop_name][score_name]

        s, p = scipy.stats.ttest_rel(scores, scores_baseline)
        stat = [s, p, p < 0.05]
        dif = np.mean(scores) - np.mean(scores_baseline)
        std = np.std(scores)

        print(score_name, crop_range, np.mean(scores))

        frame = frame.append(dict(ID=score_name,
                                  Range=crop_range,
                                  Baseline=np.mean(scores_baseline),
                                  Improved=np.mean(scores),
                                  Stat=stat,
                                  ), ignore_index=True)

# frame.to_html('compare.html')
frame

# %%
