import matplotlib.pyplot as plt
import numpy as np


def plot_env(env):
    trades = np.array(env.actions_memory)
    trades = list(trades - ((np.max(trades) - 1) / 2))

    figure, axes = plt.subplots(nrows=2, ncols=2)
    axes[0, 0].plot(env.df["ETF Price"])
    axes[0, 0].set_title("Stock price")
    axes[0, 0].xaxis.set_visible(False)
    axes[0, 1].plot(env.asset_memory)
    axes[0, 1].set_title("Portfolio value")
    axes[0, 1].xaxis.set_visible(False)
    axes[1, 0].plot(trades)
    axes[1, 0].set_title("Trades")
    axes[1, 1].plot(env.rewards_memory)
    axes[1, 1].set_title("Rewards")

    return
