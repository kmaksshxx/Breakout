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


ROOT = Path(__file__).resolve().parents[2]
saved_path = ROOT / 'checkpoint' / 'checkpoint.tar'

if __name__ == "__main__":
    gym.register_envs(ale_py)
    env = gym.make('ALE/Breakout-ram-v5', mode=0, difficulty=0)

    print(f'Device: {device}')
    q = QNetwork().to(device)
    q_target = QNetwork().to(device)
    optimizer = AdamW(q.parameters())
    buffer = ReplayBuffer()

    epsilon = 1.0
    best_reward = 0

    try:
        if device == 'cpu':
            checkpoint = torch.load(saved_path, weights_only=False, map_location='cpu')
        else:
            checkpoint = torch.load(saved_path, weights_only=False)
        q.load_state_dict(checkpoint['model'])
        q_target.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        buffer.buffer = checkpoint['buffer']
        epsilon = checkpoint['epsilon']
        best_reward = checkpoint['best_reward']
        print('Checkpoint Loaded')
    except Exception as e:
        print(e)
        print('Checkpoint Not Loaded')

    gamma = 0.99
    batch_size = 64

    epsilon_max = 1.0
    epsilon_min = 0.05
    random_episodes = 1000

    target_update = 10000

    print_episode = 100

    step = 0

    episode = 0

    reward_history = deque(maxlen=print_episode)

    while True:
        episode += 1
        s, _ = env.reset(seed=42)
        done = False
        episode_reward = 0

        # Start One Episode
        while not done:
            if random.random() < epsilon:
                a = env.action_space.sample()
            else:
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
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                # q(states) : (B, ACTION_SIZE)
                # actions : (B, )

                q_values = q(states).gather(1, actions.unsqueeze(1)).squeeze()  # (B, )

                with torch.no_grad():
                    next_q = q_target(next_states).max(1)[0]
                    target = rewards + gamma * next_q * (1 - dones)

                loss = nn.functional.huber_loss(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step % target_update == 0:
                q_target.load_state_dict(q.state_dict())

        reward_history.append(int(episode_reward))
        mean_reward = np.mean(reward_history)

        epsilon = max(epsilon - (epsilon_max - epsilon_min) / random_episodes, epsilon_min)

        if mean_reward >= best_reward:
            best_reward = episode_reward

        if mean_reward >= 40:
            break

        if episode % print_episode == 0:
            print(f"Episode: {episode} | reward: {mean_reward:.2f} | Best Score: {best_reward:.2f}")
            torch.save({
                'model': q.state_dict(),
                'optimizer': optimizer.state_dict(),
                'buffer': buffer.buffer,
                'epsilon': epsilon,
                'best_reward': best_reward
            }, saved_path)
