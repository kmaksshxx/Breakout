import gymnasium as gym
from pathlib import Path
import torch
import torch.nn as nn
import random
from torch.optim import AdamW
from src.models import *
from src.buffer import *
import ale_py


ROOT = Path(__file__).resolve().parents[2]
saved_path = ROOT / 'checkpoint' / 'checkpoint.tar'

if __name__ == "__main__":
    gym.register_envs(ale_py)
    env = gym.make('ALE/Breakout-ram-v5', mode=0, difficulty=0)

    q = QNetwork().to(device)
    q_target = QNetwork().to(device)
    optimizer = AdamW(q.parameters())

    try:
        checkpoint = torch.load(saved_path, weights_only=True)
        q.load_state_dict(checkpoint['model'])
        q_target.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Checkpoint Loaded')
    except Exception as e:
        print('Checkpoint Not Loaded')

    buffer = ReplayBuffer()

    gamma = 0.99
    batch_size = 64

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    target_update = 200

    step = 0
    best_reward = 0

    for episode in range(100_000):
        s, _ = env.reset(seed=42)
        done = False
        total_reward = 0

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
            total_reward += r
            step += 1

            if len(buffer) > batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                # q(states) : (B, ACTION_SIZE)
                # actions : (B, )

                q_values = q(states).gather(1, actions.unsqueeze(1)).squeeze()  # (B, )

                with torch.no_grad():
                    next_q = q_target(next_states).max(1)[0]
                    target = rewards + gamma * next_q * (1 - dones)

                loss = nn.functional.mse_loss(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step % target_update == 0:
                q_target.load_state_dict(q.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if total_reward >= best_reward:
            print('new best score')
            best_reward = total_reward

            torch.save({
                'model': q.state_dict(),
                'optimizer': optimizer.state_dict()
            }, saved_path)

        if total_reward >= 40:
            break

        if episode % 100 == 0 and episode > 0:
            print(f"Episode: {episode} | score: {int(total_reward)} | Best Score: {int(best_reward)}")
