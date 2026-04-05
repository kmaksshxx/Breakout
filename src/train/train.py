import gymnasium as gym
import torch
import torch.nn.functional as F
import random
from torch.optim import AdamW
from src.models import *
from src.buffer import *
import ale_py
from collections import deque
import numpy as np
from src.measure_time import *
import hydra
from omegaconf import DictConfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

saved_path = ROOT / "checkpoint" / "checkpoint.tar"
gym.register_envs(ale_py)
env = gym.make('ALE/Breakout-v5', obs_type='ram')
epsilon_max, epsilon_min = 1.0, 0.05
frame_stack = deque(maxlen=4)
seed = 42


def env_reset(seed):
    s, _ = env.reset(seed=seed)
    for _ in range(4):
        frame_stack.append(s)
    return np.concatenate(frame_stack)


def env_step(a: int, life: int):
    """
    :return: state, reward, done, lives
    """
    s, r, _, _, info = env.step(a)
    frame_stack.append(s)
    new_life = int(info['lives'])
    return np.concatenate(frame_stack), r, new_life < life, new_life


def load_checkpoint():
    if device == 'cpu':
        checkpoint = torch.load(saved_path, weights_only=False, map_location='cpu')
    else:
        checkpoint = torch.load(saved_path, weights_only=False)

    return checkpoint


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f'Device: {device}')
    q = QNetwork().to(device)
    q_target = QNetwork().to(device)
    optimizer = AdamW(q.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.capacity)

    try:
        checkpoint = load_checkpoint()
        q.load_state_dict(checkpoint['model'])
        q_target.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        best_reward = checkpoint['best_reward']
        episode = checkpoint['episode']
        step = checkpoint['step']
        print('Checkpoint Loaded')

    except Exception as e:
        print(e)
        print('Checkpoint Not Loaded')

        best_reward = 0
        episode, step = 0, 0

    epsilon = cfg.epsilon
    target_update = cfg.target_update

    reward_history = deque(maxlen=cfg.print_episode)
    loss_history = deque(maxlen=cfg.print_episode)
    loss_episode_history = deque()
    timer.reset(f"episode: {episode + cfg.print_episode}")

    while True:
        episode += 1
        s = env_reset(seed)
        episode_reward = 0
        life = 5

        # Start One Episode
        while True:
            epsilon = max(epsilon - (epsilon_max - epsilon_min) / cfg.random_steps, epsilon_min)

            if random.random() < epsilon:
                with timed(timer, "select_random_action"):
                    a = env.action_space.sample()
            else:
                with timed(timer, "select_action_from_nn"):
                    with torch.no_grad():
                        qs = q(torch.tensor(s, dtype=torch.float32, device=device))
                        a = qs.argmax().item()

            s2, r, done, life = env_step(a, life)

            buffer.push(s, a, r, s2, done)

            s = s2
            episode_reward += r
            step += 1

            # train step
            if len(buffer) > cfg.batch_size:
                with timed(timer, "sample"):
                    states, actions, rewards, next_states, dones = buffer.sample(cfg.batch_size)

                with timed(timer, "tran_step"):
                    q_values = q(states).gather(1, actions.unsqueeze(1)).squeeze()  # (B, )

                    with torch.no_grad():
                        next_q = q_target(next_states).max(1)[0]
                        target = rewards + cfg.gamma * next_q * (1 - dones)

                    loss = F.huber_loss(q_values, target)
                    loss_episode_history.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if step % target_update == 0:
                q_target.load_state_dict(q.state_dict())
                loss_episode_mean = np.mean(loss_episode_history)
                loss_history.append(loss_episode_mean)
                loss_episode_history.clear()

            if life == 0:
                break

        reward_history.append(int(episode_reward))
        mean_reward = np.mean(reward_history)

        if mean_reward > best_reward:
            best_reward = episode_reward
            print(f"New Best Score: {best_reward:.2f}")

        if mean_reward >= cfg.goal:
            break

        if episode % cfg.print_episode == 0:
            with timed(timer, "save"):
                torch.save({
                    'model': q.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_reward': best_reward,
                    'episode': episode,
                    'step': step
                }, saved_path)

            print(
                f"\nEp: {episode} |",
                f"Reward: {mean_reward:.2f} |",
                f"eps: {epsilon:.2f} |",
                f"loss: {np.mean(loss_history):.4f}\n"
            )

            if cfg.print_timer:
                timer.report()
                timer.reset(f"episode: {episode + cfg.print_episode}")


if __name__ == "__main__":
    main()
