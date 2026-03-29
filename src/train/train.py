import gymnasium as gym
from pathlib import Path
import torch
import torch.nn as nn
import random
from torch.optim import AdamW
from src.models import *
from src.buffer import *
import ale_py
from collections import deque
import numpy as np
import time
from src.measure_time import *

ROOT = Path(__file__).resolve().parents[2]
saved_path = ROOT / 'checkpoint' / 'checkpoint.tar'
gym.register_envs(ale_py)
env = gym.make('ALE/Breakout-v5', obs_type='ram')


replay_buffer_capacity = 100_000
gamma = 0.99
batch_size = 64
epsilon_max, epsilon_min = 1.0, 0.05


if __name__ == "__main__":
    print(f'Device: {device}')
    q = QNetwork().to(device)
    q_target = QNetwork().to(device)
    optimizer = AdamW(q.parameters(), lr=3e-4)
    buffer = ReplayBuffer(replay_buffer_capacity)

    try:
        if device == 'cpu':
            checkpoint = torch.load(saved_path, weights_only=False, map_location='cpu')
        else:
            checkpoint = torch.load(saved_path, weights_only=False)

        q.load_state_dict(checkpoint['model'])
        q_target.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epsilon = checkpoint['epsilon']
        best_reward = checkpoint['best_reward']
        episode = checkpoint['episode']
        step = checkpoint['step']
        print('Checkpoint Loaded')
    except Exception as e:
        print(e)
        print('Checkpoint Not Loaded')

        epsilon = 1.0
        best_reward = 0
        episode, step = 0, 0

    target_update = 1000

    print_episode = 100
    reward_history = deque(maxlen=print_episode)
    loss_history = deque(maxlen=print_episode)
    loss_episode_history = deque()

    now = time.time()

    while True:
        episode += 1
        epsilon = max(epsilon - (epsilon_max - epsilon_min) / replay_buffer_capacity, epsilon_min)
        s, _ = env.reset(seed=42)
        done = False
        episode_reward = 0

        # Start One Episode
        while not done:
            if random.random() < epsilon:
                with timed(timer, "select_random_action"):
                    a = env.action_space.sample()
            else:
                with timed(timer, "select_action_from_nn"):
                    with torch.no_grad():
                        qs = q(torch.tensor(s, dtype=torch.float32, device=device))
                        a = qs.argmax().item()

            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            buffer.push(s, a, r, s2, done)

            s = s2
            episode_reward += r
            step += 1

            if len(buffer) > batch_size:
                with timed(timer, "sample"):
                    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                with timed(timer, "tran_step"):
                    q_values = q(states).gather(1, actions.unsqueeze(1)).squeeze()  # (B, )

                    with torch.no_grad():
                        next_q = q_target(next_states).max(1)[0]
                        target = rewards + gamma * next_q * (1 - dones)

                    loss = nn.functional.huber_loss(q_values, target)
                    loss_episode_history.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if step % target_update == 0:
                q_target.load_state_dict(q.state_dict())
                loss_episode_mean = np.mean(loss_episode_history)
                loss_history.append(loss_episode_mean)
                loss_episode_history.clear()

        reward_history.append(int(episode_reward))
        mean_reward = np.mean(reward_history)

        if mean_reward > best_reward:
            best_reward = episode_reward
            print(f"New Best Score: {best_reward:.2f}")

        if mean_reward >= 40:
            break

        if episode % print_episode == 0:
            runtime = int(time.time() - now)
            now = time.time()
            m, s = divmod(runtime, 60)

            print(
                f"Ep: {episode} |",
                f"Reward: {mean_reward:.2f} |",
                f"eps: {epsilon:.2f} |",
                f"loss: {np.mean(loss_history):.4f} |"
                f"Run Time: {m}m {s}s"
            )

            with timed(timer, "save"):
                torch.save({
                    'model': q.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epsilon': epsilon,
                    'best_reward': best_reward,
                    'episode': episode,
                    'step': step
                }, saved_path)

            timer.report()
            breakpoint()
