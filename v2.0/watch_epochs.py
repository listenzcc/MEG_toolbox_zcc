# %%
import mne
import sklearn
import matplotlib.pyplot as plt
from sklearn import manifold
from tools.data_manager import DataManager

# %%
name = 'MEG_S02'

dm = DataManager(name)
dm.load_epochs(recompute=False)

# %%
dm.epochs_list[0]
# %%
dm.epochs_list[2]['4'].average().plot_joint()
# %%
dm.epochs_list
# %%
epochs_1, epochs_2 = dm.leave_one_session_out(includes=[1, 3, 5],
                                              excludes=[2, 4, 6])

# %%
for e in ['1', '2', '3', '4']:
    epochs_1[e].average().plot_joint(title=e)
print()

# %%
xdawn = mne.preprocessing.Xdawn(n_components=6)
xdawn.fit(epochs_1)
epochs = xdawn.apply(epochs_1)

# %%
epochs
# %%
for e in ['1', '2', '3', '4']:
    epochs['2'][e].average().plot_joint(title=e)
print()

# %%
a = xdawn.transform(epochs_1)
# %%
a.shape
# %%
tsne = manifold.TSNE(n_components=2, n_jobs=48)
vectorizer = mne.decoding.Vectorizer()
b = vectorizer.fit_transform(a)
b.shape

# %%
c = tsne.fit_transform(b)
c.shape

# %%
plt.style.use('ggplot')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
y = epochs_1.events[:, -1]
for j in [1, 2, 3, 4]:
    ax.scatter(c[y == j, 0], c[y == j, 1], alpha=0.3, label=j)
ax.legend()
# %%
