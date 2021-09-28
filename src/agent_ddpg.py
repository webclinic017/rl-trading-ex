from .common import ReplayBuffer, noise_normal, update_target
import numpy as np

# import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from ray import tune
# import wandb


class PolicyNet(nn.Module):
    def __init__(self, state_space, num_hidden_p):
        super(PolicyNet, self).__init__()
        self.fc_1 = nn.Linear(state_space, num_hidden_p)
        self.fc_2 = nn.Linear(num_hidden_p, 1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        p = torch.tanh(self.fc_2(x))

        return p


class QNet(nn.Module):
    def __init__(self, state_space, num_hidden_q):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_space, round(num_hidden_q / 2))
        self.fc_a = nn.Linear(1, round(num_hidden_q / 2))
        self.fc_q = nn.Linear(num_hidden_q, 1)

    def forward(self, s, a):
        h1 = F.relu(self.fc_s(s))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = self.fc_q(cat)

        return q


class DDPG:
    def __init__(self, config):
        self.state_space = config["state_space"]
        self.num_hidden_p = config["num_hidden_p"]
        self.num_hidden_q = config["num_hidden_q"]
        self.lr_p = config["lr_p"]
        self.lr_q = config["lr_q"]
        self.gamma = config["gamma"]
        self.var_k = config["var_k"]
        self.tau = config["tau"]
        self.act_noise = config["act_noise"]
        self.buffer_size = config["buffer_size"]
        self.batch_size = config["batch_size"]
        self.update_freq = config["update_freq"]
        self.train_cycle = config["train_cycle"]
        self.iters = config["iters"]

        self.memory = ReplayBuffer(
            buffer_size=self.buffer_size, batch_size=self.batch_size
        )

        # print(self.state_space, self.num_hidden_p)
        self.p = PolicyNet(state_space=self.state_space, num_hidden_p=self.num_hidden_p)
        self.p_target = PolicyNet(
            state_space=self.state_space, num_hidden_p=self.num_hidden_p
        )
        self.p_target.load_state_dict(self.p.state_dict())

        # print(self.state_space, self.num_hidden_q)
        self.q = QNet(state_space=self.state_space, num_hidden_q=self.num_hidden_q)
        self.q_target = QNet(self.state_space, self.num_hidden_q)
        self.q_target.load_state_dict(self.q.state_dict())

        self.p_optimizer = optim.Adam(params=self.p.parameters(), lr=self.lr_p)
        self.q_optimizer = optim.Adam(params=self.q.parameters(), lr=self.lr_q)

    def update(self):
        for _ in range(self.train_cycle):
            states, actions, rewards, next_states, done_masks = self.memory.sample()

            r = rewards.unsqueeze(dim=1)  # TODO: Check torch operations
            target = (
                r
                + self.gamma
                * self.q_target(next_states, self.p_target(next_states))
                * done_masks
                # - self.var_k * np.var(r.numpy()) # TODO: Check variance parameter
            )

            self.p_optimizer.zero_grad()

            p_loss = self.q(states, self.p(states)).mean()
            p_loss.backward()

            self.p_optimizer.step()

            self.q_optimizer.zero_grad()

            q_loss = F.smooth_l1_loss(self.q(states, actions), target.detach())
            q_loss.backward()

            self.q_optimizer.step()

        return p_loss, q_loss

    def train(self, env):
        for iter in range(0, self.iters):
            curr_iter = iter + 1
            state = torch.tensor(env.reset()).float()
            done = False

            while not done:
                if self.memory.size() <= self.buffer_size:
                    action = np.random.uniform(low=-1, high=1)
                else:
                    noise = noise_normal(self.act_noise)
                    action = self.p(state).detach().numpy() + noise
                    action = (
                        np.clip(action, 0, 1) * self.max_trade * 2 - self.max_trade
                    )  # Convert [0,1] to [-max_trade, max_trade]

                next_state, reward, done, _ = env.step(action)
                self.memory.add((state, action, reward, next_state, done))

                state = torch.tensor(next_state).float()

        p_loss, q_loss = self.update()
        # TODO: WANDB LOG HERE/PASS TRAINING FN

        if curr_iter % self.update_freq == 0:
            update_target(self.p, self.p_target, self.tau)
            update_target(self.q, self.q_target, self.tau)

        loss = {"p_loss": p_loss, "q_loss": q_loss}

        return loss

    def test(self, env):
        state = torch.tensor(env.reset()).float()
        done = False

        while not done:
            action = self.p(state).detach().numpy()
            next_state, _, done, _ = env.step(action)
            state = torch.tensor(next_state).float()

        results = {"sharpe": env.sharpe_memory[-1]}

        return results
