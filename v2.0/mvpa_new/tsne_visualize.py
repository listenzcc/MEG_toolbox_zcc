# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# %%
tsne_folder = 'tsne_xdawn_x2'
tsne_files = sorted([os.path.join(tsne_folder, e)
                     for e in os.listdir(tsne_folder)])
display(tsne_files)

# %%
path = tsne_files[17]
# path = tsne_files[39]
# path = tsne_files[73]
data = pickle.load(open(path, 'rb'))
display(path, data)

train_y = data['train_y']
test_y = data['test_y']
train_x2 = data['x2'][:len(train_y)]
test_x2 = data['x2'][len(train_y):]
display(train_x2.shape, train_y.shape, test_x2.shape, test_y.shape)

# %%
plt.style.use('ggplot')
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for event, c in zip([1, 2, 4],
                    ['#aa3322', '#666666', '#90b060']):
    ax = axes[0]
    ax.scatter(train_x2[train_y == event, 0],
               train_x2[train_y == event, 1],
               label=event, c=c, alpha=0.2)

    ax = axes[1]
    ax.scatter(train_x2[train_y == event, 0],
               train_x2[train_y == event, 1],
               label=event, c=c, alpha=0.2)
    ax.scatter(test_x2[test_y == event, 0],
               test_x2[test_y == event, 1],
               label=f'+{event}',
               marker='+')

    ax = axes[2]
    ax.scatter(test_x2[test_y == event, 0],
               test_x2[test_y == event, 1], label=f'+{event}')

for ax in axes:
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])

axes[0].set_title('Train')
axes[2].set_title('Test')
axes[1].set_title('Joint')

fig.tight_layout()

figname = '{}.png'.format(os.path.basename(path))
fig.savefig(figname)
# %%
