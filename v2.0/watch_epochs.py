# %%
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
