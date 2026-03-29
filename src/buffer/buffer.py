import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.state_buffer = np.empty((capacity, 128), dtype=np.float32)
        self.action_buffer = np.empty(capacity, dtype=np.int32)
        self.reward_buffer = np.empty(capacity, dtype=np.float32)
        self.state_next_buffer = np.empty((capacity, 128), dtype=np.float32)
        self.done_buffer = np.empty(capacity, dtype=np.float32)

        self.capacity = capacity
        self.pointer = 0

    def state_dict(self):
        return (
            self.capacity,
            self.pointer,
            self.state_buffer,
            self.action_buffer,
            self.reward_buffer,
            self.state_next_buffer,
            self.done_buffer,
        )

    @classmethod
    def load_state(cls, state):
        tracker = cls(capacity=state[0])
        tracker.pointer = state[1]
        tracker.state_buffer = state[2]
        tracker.action_buffer = state[3]
        tracker.reward_buffer = state[4]
        tracker.state_next_buffer = state[5]
        tracker.done_buffer = state[6]

        return tracker

    def push(self, s, a, r, s2, d):
        idx = self.pointer % self.capacity
        self.state_buffer[idx] = s
        self.action_buffer[idx] = a
        self.reward_buffer[idx] = r
        self.state_next_buffer[idx] = s2
        self.done_buffer[idx] = d

        self.pointer += 1

    def sample(self, batch_size):
        idx = np.random.choice(len(self), batch_size)

        s, a, r, s2, d = map(
            lambda x: x[idx],
            [self.state_buffer, self.action_buffer, self.reward_buffer,
             self.state_next_buffer, self.done_buffer]
        )

        return (
            torch.from_numpy(s).to(torch.float32).to(device),
            torch.from_numpy(a).to(torch.int64).to(device),
            torch.from_numpy(r).to(torch.float32).to(device),
            torch.from_numpy(s2).to(torch.float32).to(device),
            torch.from_numpy(d).to(torch.float32).to(device),
        )

    def __len__(self):
        return min(self.capacity, self.pointer)
