# %%
import mne
import sklearn
import matplotlib.pyplot as plt
from sklearn import manifold
from tools.data_manager import DataManager
from tools.computing_components import MyXdawn
from tools.epochs_tools import get_MVPA_data

# %%
name = 'MEG_S02'

dm = DataManager(name)
dm.load_epochs(recompute=False)

# %%
epochs_1, epochs_2 = dm.leave_one_session_out(includes=[1, 3, 5],
                                              excludes=[2, 4, 6])

# %%
X, y = get_MVPA_data(epochs_2)
X.shape, y.shape

# %%
my_xdawn = MyXdawn(epochs_1)
my_xdawn.fit()
epochs_1_xdawn = my_xdawn.apply()
epochs_2_xdawn = my_xdawn.apply(epochs_2)
data_1 = my_xdawn.transform()
data_2 = my_xdawn.transform(epochs_2)

# %%
for e in ['1', '2', '3', '4']:
    epochs_1_xdawn['1'][e].average().plot_joint(title=e)
    epochs_2_xdawn['1'][e].average().plot_joint(title=e)

epochs_1_xdawn['1']['3'].average().plot_joint(title='3 with filter 1')
epochs_1_xdawn['3']['3'].average().plot_joint(title='3 with filter 3')

print(data_1.shape, data_2.shape)

# %%
csp = mne.decoding.CSP(n_components=4)
csp.fit(epochs_1_xdawn['1'].get_data(), epochs_1_xdawn['1'].events[:, -1])

# %%
vectorizer = mne.decoding.Vectorizer()
tsne = manifold.TSNE(n_components=2, n_jobs=48)
# %%
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
help(mne.decoding.csp)
# %%
