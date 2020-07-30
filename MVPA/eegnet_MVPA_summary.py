# %%
import pickle
from sklearn import metrics
import numpy as np

# %%
with open(os.path.join('results_eegnet', 'MEG_S03_eegnet.pkl'), 'rb') as f:
    content_eegnet = pickle.load(f)

with open(os.path.join('results', 'MEG_S03_segment.pkl'), 'rb') as f:
    content_my = pickle.load(f)


# %%
report = metrics.classification_report(y_true=content_eegnet['y_all'],
                                       y_pred=np.round(content_eegnet['y_pred']))

print(report)

# %%
_report = metrics.classification_report(y_true=content_my['y_all'],
                                        y_pred=np.round(content_my['d_y_pred']))

print(_report)

# %%
