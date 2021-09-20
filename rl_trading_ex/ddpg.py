from common import ReplayBuffer, noise_OU
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ray import tune
import wandb


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
        self.fc_s = nn.Linear(state_space, num_hidden_q / 2)
        self.fc_a = nn.Linear(1, num_hidden_q / 2)
        self.fc_q = nn.Linear(num_hidden_q, num_hidden_q / 2)
        self.fc_out = nn.Linear(num_hidden_q / 2, 1)

    def forward(self, s, a):
        h1 = F.relu(self.fc_s(s))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)

        return q


class DDPG:
    # TODO: Review all params and only use necessary
    def __init__(
        self,
        state_space: int,
        max_trade: int,
        num_hidden_p: int,
        num_hidden_q: int,
        lr_p: float,
        lr_q: float,
        buffer_size: int,
        gamma: float,
        k: float,
        tau: float,
        var_k: float,
        explore_pc: float,
        iters: float,
        cycle: int,
        epsilon: float,
        min_epsilon: float,
        decay_epsilon: float,
        update_freq: int,
    ):

        self.state_space = state_space
        self.max_trade = max_trade
        self.num_hidden_p = num_hidden_p
        self.num_hidden_q = num_hidden_q
        self.lr_p = lr_p
        self.lr_q = lr_q
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.k = k
        self.tau = tau
        self.var_k = var_k
        self.explore_pc = explore_pc
        self.iters = iters
        self.cycle = cycle
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_epsilon = decay_epsilon
        self.update_freq = update_freq

        self.expl_iters = round(self.iters * self.explore_pc)

        self.memory = ReplayBuffer(self.buffer_size)

        self.p = PolicyNet(self.state_space, self.num_hidden_p)
        self.p_target = PolicyNet(self.state_space, self.num_hidden_p)
        self.p_target.load_state_dict(self.p.state_dict())

        self.q = QNet(self.state_space, self.num_hidden_q)
        self.q_target = QNet(self.state_space, self.num_hidden_q)
        self.q_target.load_state_dict(self.q.state_dict())

        self.p_optimizer = optim.Adam(self.p.parameters(), lr=self.lr_p)
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=self.lr_q)

    def update(self):
        for _ in range(self.cycle):
            states, actions, rewards, next_states, done_masks = self.memory.sample(
                batch_size=self.batch_size
            )

            r = rewards.unsqueeze(dim=1)  # TODO: Check torch operations
            target = (
                r
                + self.gamma
                * self.q_target(next_states, self.p_target(next_states))
                * done_masks
                - self.var_k * np.var(r.numpy())  # TODO: Review variance term
            )

            self.p_optimizer.zero_grad()

            # TODO: Check loss functions used
            p_loss = self.q(states, self.p(states)).mean()
            p_loss.backward()

            self.p_optimizer.step()

            self.q_optimizer.zero_grad()

            q_loss = F.smooth_l1_loss(self.q(states, actions), target.detach())
            q_loss.backward()

            self.q_optimizer.step()

        return p_loss, q_loss

    def train(self, env):
        epsilon = self.epsilon

        for iter in self.iters:
            state = torch.tensor(env.reset()).float()
            done = False

            self.epsilon = max(self.min_epsilon, epsilon * self.decay_epsilon)

            while not done:
                if iter <= self.expl_iters:
                    action = np.random.rand(low=0, high=1)
                else:
                    # noise = noise_OU(x, mu, theta, sigma, epoch - expl_iters) # TODO: Implement noise after fn checked
                    noise = 0  # Placeholder
                    action = self.p(state).detach().numpy() + noise
                    action = np.clip(a, -1, 1) * self.max_trade

                next_s, reward, done, _ = env.step(action)

                # if env.day % env.reset_days != 0:
                #     self.memory.add((s, a, r, next_s, done))
                self.memory.add((s, a, r, next_s, done))
                s = torch.tensor(next_s).float()

        self.epoch += 1

        for _ in self.cycle:
            p_loss, q_loss = self.torch_step()  # log loss here

        target_update(self.p, self.p_target, self.tau)
        target_update(self.q, self.q_target, self.tau)

        loss = {"p_loss_ddpg": p_loss, "q_loss_ddpg": q_loss}

        return loss

    def test(self, test_env):
        s = torch.tensor(test_env.reset()).float()
        done = False
        epoch = 0

        total_reward = 0

        while not done:
            a = self.p(s).detach().numpy()
            next_s, r, done, _ = test_env.step(a)
            s = torch.tensor(next_s).float()
            total_reward += r

        stats = {"sharpe": test_env.sharpe[-1], "reward": total_reward}

        return test_env, test_env.sharpe, test_env.total_reward
