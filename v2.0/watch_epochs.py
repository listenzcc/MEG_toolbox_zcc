# %%
from tools.data_manager import DataManager

# %%
name = 'MEG_S02'

dm = DataManager(name)
dm.load_epochs(recompute=False)

# %%
