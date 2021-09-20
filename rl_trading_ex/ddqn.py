from common import ReplayBuffer, update_target

# import collections
# import math
# import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from ray import tune

# TODO: Type hints


class QNet(nn.Module):
    def __init__(self, state_space, num_hidden, max_trade):
        super(QNet, self).__init__()
        self.max_trade = max_trade
        self.fc_1 = nn.Linear(state_space, num_hidden)
        self.fc_2 = nn.Linear(
            num_hidden, max_trade * 2 + 1
        )  # Change position by max +/-N shares at every step incl. 0

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        q = self.fc_2(x)

        return q

    def get_action(self, state, epsilon):
        out = self.forward(state)
        coin = random.random()

        if coin < epsilon:
            action = random.randint(-self.max_trade, self.max_trade) / self.max_trade
        else:
            action = (out.argmax().item() - self.max_trade) / self.max_trade

        return action


class DoubleDQN:
    def __init__(
        self,
        state_space: int,
        max_trade: int,
        num_hidden: int,
        buffer_size: int,
        lr: float,
        batch_size: int,
        epsilon: float,
        min_epsilon: float,
        decay_epsilon: float,
        tau: float,
        iters: int,
        train_cycle: int,
        update_freq: int,
    ):
        self.state_space = state_space
        self.max_trade = max_trade
        self.num_hidden = num_hidden
        self.buffer_size = buffer_size
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_epsilon = decay_epsilon
        self.tau = tau
        self.iters = iters
        self.train_cycle = train_cycle
        self.update_freq = update_freq

        self.memory = ReplayBuffer(buffer_size=self.buffer_size)

        self.q = QNet(
            state_space=self.state_space,
            num_hidden=self.num_hidden,
            max_trade=self.max_trade,
        )
        self.q_target = QNet(
            state_space=self.state_space,
            num_hidden=self.num_hidden,
            max_trade=self.max_trade,
        )
        self.q_target.load_state_dict(self.q.state_dict())

        self.q_optimizer = optim.Adam(self.q.parameters(), lr=self.lr)

    def update(self):
        states, actions, rewards, next_states, done_masks = self.memory.sample(
            batch_size=self.batch_size
        )

        q_out = self.q(states)
        q_a = q_out.gather(
            1,
            (actions * self.max_trade + self.max_trade)  # TODO: Check fn
            # .type(torch.int64)
            .unsqueeze(1),
        )
        max_q_prime = self.q_target(next_states).max(1)[0].unsqueeze(1)

        r = rewards.unsqueeze(dim=1)
        target = (
            r
            + self.gamma * max_q_prime * done_masks
            # - self.params["k"] * np.var(r)  # TODO: Include variance parameter
        )

        self.q_optimizer.zero_grad()

        q_loss = F.smooth_l1_loss(
            q_a, target
        )  # less sensitive to outliers than MSE loss
        q_loss.backward()

        self.q_optimizer.step()

        return q_loss

    def train(self, env):
        epsilon = self.epsilon

        for iter in self.iters:
            state = torch.tensor(env.reset()).float()
            done = False

            self.epsilon = max(self.min_epsilon, epsilon * self.decay_epsilon)

            while not done:
                action = self.q.get_action(state, epsilon)
                # action = np.clip(action, 0, 1) # TODO: Should not be necessary

                next_state, reward, done, _ = env.step(
                    np.array([action])
                )  # TODO: Env requires np array input for now
                # next_s, r, done, _ = env.step(a)

                self.memory.add((state, action, reward, next_state, done))

                state = torch.tensor(next_state).float()

            for _ in range(self.train_cycle):
                loss = self.update()

            wandb.log({"loss": loss})
            tune.report(loss=loss)

            if iter % self.update_freq == 0:
                update_target(self.q, self.q_target, self.tau)

    def test(self, env):
        state = torch.tensor(env.reset()).float()
        done = False

        total_reward = 0

        while not done:
            action = self.q.sample_action(state, self.epsilon)
            next_state, reward, done, _ = env.step(np.array([action]))
            # next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = torch.tensor(next_state).float()

        stats = {
            "sharpe": env.sharpe_memory[-1],
            "reward": total_reward,
        }  # TODO: Check if adding more metrics

        return stats
