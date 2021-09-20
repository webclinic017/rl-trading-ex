import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd


class PortfolioAllocationEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(  # TODO: Check defaults
        self,
        df,
        max_trade=0.1,
        initial_amount=10000,
        transaction_cost=0.02,
        reward_scaling=1,
        start_ind=None,
        episode_len=None,
        seed=None,
    ):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.max_trade = max_trade
        self.initial_amount = initial_amount
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.start_ind = start_ind
        self.episode_len = episode_len
        self.seed = seed

        self.terminal = False
        self.action_space = spaces.Box(
            low=-self.max_trade, high=self.max_trade, shape=1
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=len(df.columns) - 1 + 2
        )  # All values passed from df less date and add cash, stock position
        # TODO: Fix

        self.state = self._initiate_state()

    def _seed(self, seed=None):  # Sets seed if supplied
        if seed is None:
            self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def _initiate_state(self, seed=None):  # Initiates state according to passed params
        self._seed(seed)

        self.cash = self.initial_amount  # TODO: Include portfolio value
        self.position = 0
        self.portfolio_value = 0
        self.fee = 0
        self.reward = 0
        self.trades = 0

        self.asset_memory = [self.initial_amount]
        self.date_memory = [self.date]
        self.actions_memory = []
        self.rewards_memory = []

        if self.start_ind is not None:
            self.ind = self.start_ind
        else:
            self.ind = 0

        if self.episode_len is not None:
            self.max_ind = self.ind + self.episode_len
        else:
            self.max_ind = len(self.df) - 1

        state = self._get_state()

        return state

    def _get_state(self):
        cash = self.cash
        position = self.position
        df_current_row = self.df.iloc[self.ind]
        df_current_row = list(df_current_row.drop(cols=["Date", "S&P Index"]))

        state = [cash, position] + df_current_row

        return state

    def _trade_stock(self, trade_size):
        current_price = self.df["ETF Price"][self.ind]
        cash_change = -(trade_size * current_price)

        self.fee = cash_change * self.transaction_cost

        if self.cash + cash_change >= 0:  # First check if there is enough cash
            self.cash = self.cash + cash_change
            self.position = self.position + trade_size
        else:
            pass

        return

    def step(self, action):

        self._trade_stock(action)

        current_price = self.df["ETF Price"][self.ind]
        next_price = self.df["ETF Price"][self.ind + 1]

        reward = (
            next_price * self.position - current_price * self.position
        ) * self.reward_scaling
        self.reward = reward

        self.ind = self.ind + 1
        self.date = self.df["Date"][self.ind]

        if self.ind < self.max_ind:
            done = False
        else:
            done = True

        self.asset_memory.append(self.reward + self.cash)
        self.date_memory.append(self.date)
        self.actions_memory.append(self.action)
        self.rewards_memory.append(self.reward)

        observation = self._get_state()
        info = {}

        return observation, reward, done, info

    def reset(self):
        state = self._initiate_state()

        return state

    def render(self, mode="human"):
        raise NotImplementedError

        return

    def close(self):
        raise NotImplementedError

        return


class StockTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(  # TODO: Check defaults
        self,
        df,
        max_trade=5,
        initial_amount=10000,
        transaction_cost=0.02,
        reward_scaling=1,
        start_ind=None,
        episode_len=None,
        seed=None,
    ):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.max_trade = max_trade
        self.initial_amount = initial_amount
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.start_ind = start_ind
        self.episode_len = episode_len
        self.seed = seed

        self.terminal = False
        self.action_space = spaces.Box(
            low=-self.max_trade, high=self.max_trade, shape=1
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=len(df.columns) - 1 + 2
        )  # All values passed from df less date and add cash, stock position
        # TODO: Fix

        self.state = self._initiate_state()

    def _seed(self, seed=None):  # Sets seed if supplied
        if seed is None:
            self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def _initiate_state(self, seed=None):  # Initiates state according to passed params
        self._seed(seed)

        self.cash = self.initial_amount  # TODO: Include portfolio value
        self.position = 0
        self.portfolio_value = 0
        self.fee = 0
        self.reward = 0
        self.trades = 0

        self.asset_memory = [self.initial_amount]
        self.date_memory = [self.date]
        self.actions_memory = []
        self.rewards_memory = []

        if self.start_ind is not None:
            self.ind = self.start_ind
        else:
            self.ind = 0

        if self.episode_len is not None:
            self.max_ind = self.ind + self.episode_len
        else:
            self.max_ind = len(self.df) - 1

        state = self._get_state()

        return state

    def _get_state(self):
        cash = self.cash
        position = self.position
        df_current_row = self.df.iloc[self.ind]
        df_current_row = list(df_current_row.drop(cols=["Date", "S&P Index"]))

        state = [cash, position] + df_current_row

        return state

    def _trade_stock(self, trade_size):
        current_price = self.df["ETF Price"][self.ind]
        cash_change = -(trade_size * current_price)

        self.fee = cash_change * self.transaction_cost

        if self.cash + cash_change >= 0:  # First check if there is enough cash
            self.cash = self.cash + cash_change
            self.position = self.position + trade_size
        else:
            pass

        return

    def step(self, action):

        self._trade_stock(action)

        current_price = self.df["ETF Price"][self.ind]
        next_price = self.df["ETF Price"][self.ind + 1]

        reward = (
            next_price * self.position - current_price * self.position
        ) * self.reward_scaling
        self.reward = reward

        self.ind = self.ind + 1
        self.date = self.df["Date"][self.ind]

        if self.ind < self.max_ind:
            done = False
        else:
            done = True

        self.asset_memory.append(self.reward + self.cash)
        self.date_memory.append(self.date)
        self.actions_memory.append(self.action)
        self.rewards_memory.append(self.reward)

        observation = self._get_state()
        info = {}

        return observation, reward, done, info

    def reset(self):
        state = self._initiate_state()

        return state

    def render(self, mode="human"):
        raise NotImplementedError

        return

    def close(self):
        raise NotImplementedError

        return
