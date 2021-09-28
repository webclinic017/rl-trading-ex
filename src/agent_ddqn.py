from .common import ReplayBuffer, update_target

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, state_space, num_hidden, max_trade):
        super(QNet, self).__init__()
        self.max_trade = max_trade
        self.num_hidden = num_hidden
        self.fc_1 = nn.Linear(state_space, self.num_hidden)
        self.fc_2 = nn.Linear(self.num_hidden, max_trade * 2 + 1)

    def forward(self, x):
        x1 = F.relu(self.fc_1(x))
        q = self.fc_2(x1)

        return q

    def get_action(self, state, epsilon):
        out = self.forward(state)
        coin = random.random()

        if coin < epsilon:
            action = random.randint(-self.max_trade, self.max_trade)
        else:
            action = out.argmax().item() - self.max_trade - 1

        return action


class DDQN:
    def __init__(self, config):

        self.state_space = config["state_space"]
        self.max_trade = config["max_trade"]
        self.num_hidden = config["num_hidden"]
        self.lr = config["lr"]
        self.gamma = config["gamma"]
        self.var_k = config["var_k"]
        self.tau = config["tau"]
        self.batch_size = config["batch_size"]
        self.eps = config["eps"]
        self.eps_min = config["eps_min"]
        self.eps_decay = config["eps_decay"]
        self.buffer_size = config["buffer_size"]
        self.batch_size = config["batch_size"]
        self.update_freq = config["update_freq"]
        self.train_cycle = config["train_cycle"]
        self.iters = config["iters"]

        self.memory = ReplayBuffer(
            buffer_size=self.buffer_size, batch_size=self.batch_size
        )

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
        for _ in range(self.train_cycle):
            states, actions, rewards, next_states, done_masks = self.memory.sample()

            q_out = self.q(states)
            q_a = q_out.gather(1, actions.unsqueeze(1)).float()

            max_q_prime = self.q_target(next_states).max(1)[0].unsqueeze(1)
            r = rewards.unsqueeze(dim=1)
            target = (r + self.gamma * max_q_prime * done_masks).float()

            self.q_optimizer.zero_grad()

            q_loss = F.smooth_l1_loss(q_a, target)
            q_loss.backward()

            self.q_optimizer.step()

        return q_loss

    def train(self, env):
        epsilon = self.eps

        for iter in range(0, self.iters):
            curr_iter = iter + 1
            state = torch.tensor(env.reset()).float()
            done = False

            epsilon = max(self.eps_min, epsilon * self.eps_decay)

            while not done:
                if self.memory.size() <= self.buffer_size:
                    action = np.random.randint(low=0, high=self.max_trade * 2 + 1)
                else:
                    action = self.q.get_action(state, epsilon)

                next_state, reward, done, _ = env.step(action)
                next_state = next_state
                self.memory.add((state, action, reward, next_state, done))

                state = torch.tensor(next_state).float()

            q_loss = self.update()

            if curr_iter % self.update_freq == 0:
                update_target(self.q, self.q_target, self.tau)

            loss = {"q_loss": q_loss}

            return loss

    def test(self, env):
        state = torch.tensor(env.reset()).float()
        done = False

        while not done:
            action = self.q.get_action(state, 0)
            next_state, _, done, _ = env.step(action)
            state = torch.tensor(next_state).float()

        results = {"total reward": env.total_reward}

        return results
