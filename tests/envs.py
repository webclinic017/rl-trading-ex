# Convert to unittest

#%%
import sys

sys.path.append("..")

#%%
import os
import pandas as pd
from src.rl_trading.envs import PortfolioAllocationEnv, StockTradingEnv
import yaml

#%%
module_dir = os.path.dirname(os.path.dirname(__file__))
parameters = yaml.safe_load(open(os.path.join(module_dir, "parameters.yml")))

#%%
data_path = parameters["config"]["data_path"]
train_path = module_dir + data_path + "/train.csv"

#%%
df = pd.read_csv(train_path)

# %%
env_stock = StockTradingEnv(df=df)
