# %%
import os
import pickle
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import multiprocessing


# %%
folder = 'Results_generalize'
files = sorted(os.listdir(folder))

tmp_folder = 'tmp'
if not os.path.exists(tmp_folder):
    os.mkdir(tmp_folder)

# %%


for file in files:
    print(file)
    res = pickle.load(open(os.path.join(folder, file), 'rb'))

    def get_score(res, name):
        generalize_pred_y = res['generalize_pred_y']
        test_y = res['test_y']
        times = res['times']
        n = len(times)

        def score(y_pred, y_true=test_y):
            return metrics.classification_report(y_true=y_true,
                                                 y_pred=y_pred,
                                                 output_dict=True)['1']

        _scores = {
            'recall': np.zeros((n, n)),
            'precision': np.zeros((n, n)),
            'f1-score': np.zeros((n, n))
        }

        for j in range(n):
            for k in range(n):
                sc = score(generalize_pred_y[:, j, k])
                for key in _scores:
                    _scores[key][j, k] = sc[key]

        pickle.dump(_scores,
                    open(os.path.join(tmp_folder, f'{name}_scores.pkl'), 'wb'))

    p = multiprocessing.Process(target=get_score, args=(res, file))
    p.start()


# %%
scores = [pickle.load(open(os.path.join(tmp_folder, file), 'rb'))
          for file in os.listdir(tmp_folder)]

display(len(scores))

mean_scores = {
    'recall': np.mean([e['recall'] for e in scores], axis=0),
    'precision': np.mean([e['precision'] for e in scores], axis=0),
    'f1-score':  np.mean([e['f1-score'] for e in scores], axis=0),
}

t0, t1 = -0.2, 1.2
plt.style.use('default')
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for j, key in enumerate(scores[0]):
    ax = axes[j]
    im = ax.imshow(mean_scores[key], extent=[t0, t1, t0, t1],
                   origin='bottom', vmin=0, vmax=1)
    # ax.set_xticks(times)
    ax.set_title(key)
fig.tight_layout()
# fig.colorbar(ax)

# %%
