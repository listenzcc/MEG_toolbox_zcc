# %%
import os
import pickle
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

# %%
folder = 'EEGnet_xdawn_1'


def pickle_read(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# %%
for file in os.listdir(folder):
    predicts = pickle_read(os.path.join(folder, file))
    # display(predicts)
    for pred in predicts:
        y_true = pred['y_true']
        _y_true = y_true.copy()
        _y_true[y_true != 1] = 0
        events = pred['events']
        y_pred12 = pred['y_pred12']
        y_pred14 = pred['y_pred14']
        y_pred124 = pred['y_pred124']

        y_pred_mix = (y_pred12 + y_pred14 + y_pred124) / 3

        y_pred = y_pred_mix.copy()
        for j, es in enumerate(events):
            finds = []
            for k, e in enumerate(events[max(j-20, 0):min(j+20, len(events))]):
                d = np.abs(es[0] - e[0])
                if d < 1200:
                    finds.append(y_pred_mix[j-20+k])
            if not y_pred_mix[j] == np.max(finds):
                y_pred[j] = 0

        order = np.argsort(y_true)
        order = range(len(order))
        # order = np.where(y_test == 1)[0]
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot(y_pred12[order], label='12')
        ax.plot(1+y_pred14[order], label='14')
        ax.plot(2+y_pred124[order], label='124')
        ax.plot(3+y_pred_mix[order], label='mix')
        ax.plot(-y_true[order], label='y')
        ax.plot(-y_pred[order], label='pred')
        ax.legend()

        print('pred')
        print(sklearn.metrics.classification_report(y_pred=y_pred > 0.0,
                                                    y_true=_y_true))


# %%
