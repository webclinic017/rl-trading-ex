from copy import copy
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class StockTradingEnv(gym.Env):
    def __init__(self, df, config):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.max_trade = config["max_trade"]
        self.max_short = config["max_short"]
        self.initial_amount = config["initial_amount"]
        self.transaction_cost = config["transaction_cost"]
        self.reward_scaling = config["reward_scaling"]
        self.episode_len = config["episode_len"]
        self.seed = config["seed"]

        self.ind_low = 0
        self.ind_high = len(self.df) - self.episode_len - 1

        self.action_space = spaces.Discrete(1)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, len(df.columns) - 2)
        )

        self.terminal = False
        self.state = self._initiate_state()

    def _seed(self):
        self.np_random, seed = seeding.np_random(self.seed)

        return [seed]

    def _initiate_state(self):
        self._seed()
        self.ind_start = np.random.randint(low=self.ind_low, high=self.ind_high)
        self.ind = copy(self.ind_start)

        self.cash = self.initial_amount
        self.position = 0
        self.portfolio_value = self.cash + self.position
        self.fees = 0
        self.trades = 0
        self.date = self.df.loc[self.ind, "Date"]
        self.reward = 0
        self.total_reward = 0

        self.asset_memory = [self.portfolio_value]
        self.date_memory = [self.date]
        self.actions_memory = []
        self.rewards_memory = []

        state = self._get_state()

        return state

    def _get_state(self):
        cash = self.cash
        position = self.position
        df_current_row = self.df.iloc[self.ind]
        df_current_row = df_current_row.drop(labels=["Date", "S&P Index"])
        df_current_row = list(df_current_row)

        state = [cash, position] + df_current_row

        return state

    def _trade_stock(self, trade_size):
        current_price = self.df.loc[self.ind, "ETF Price"]
        cash_change = -(trade_size * current_price)
        self.fee = cash_change * self.transaction_cost

        if self.cash + cash_change - self.fee >= 0:
            self.cash = self.cash + cash_change - self.fee
            self.position = self.position + trade_size
            self.portfolio_value = self.cash + self.position * current_price

        return

    def step(self, action):
        self._trade_stock(action)

        current_price = self.df.loc[self.ind, "ETF Price"]
        next_price = self.df.loc[self.ind + 1, "ETF Price"]

        reward = (
            next_price * self.position - current_price * self.position
        ) * self.reward_scaling
        self.total_reward = self.total_reward + reward
        self.reward = reward

        self.ind = self.ind + 1
        self.date = self.df.loc[self.ind, "Date"]

        if self.ind < self.ind_start + self.episode_len:
            done = False
        else:
            done = True

        self.asset_memory.append(self.position * next_price + self.cash)
        self.date_memory.append(self.date)
        self.actions_memory.append(action)
        self.rewards_memory.append(self.total_reward)

        observation = self._get_state()
        info = {}

        return observation, reward, done, info

    def reset(self):
        state = self._initiate_state()

        return state
