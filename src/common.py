import collections
import numpy as np
import random
import torch


def update_target(net, net_target, tau):
    for param, param_target in zip(net.parameters(), net_target.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def noise_normal(sigma):
    noise = np.random.normal(0, sigma)
    return noise


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=self.buffer_size)

    def size(self):
        return len(self.buffer)

    def add(self, observation):
        self.buffer.append(observation)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, done_masks = [], [], [], [], []

        for observation in batch:
            s, a, r, n_s, done = observation
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(n_s)
            if done:
                done_mask = 0
            else:
                done_mask = 1
            done_masks.append([done_mask])

        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.tensor(next_states, dtype=torch.float)
        done_masks = torch.tensor(done_masks)

        return states, actions, rewards, next_states, done_masks
