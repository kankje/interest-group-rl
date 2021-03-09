from itertools import count
import gym
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from config import config, device, eps
from memory import Memory, Transition
from model import load_model, save_model
from worker import Worker
import plot


def normalize(tensor):
    return (tensor - tensor.mean()) / (tensor.std() + eps)


def discount_rewards(rewards, masks):
    rewards = torch.Tensor(rewards).to(device)

    discounted_rewards = torch.zeros_like(rewards).to(device)
    running_reward = 0

    for t in reversed(range(len(rewards))):
        running_reward = rewards[t] + config.gamma * running_reward * masks[t]
        discounted_rewards[t] = running_reward

    return normalize(discounted_rewards)


def calculate_loss(memory):
    action_log_probs, values, rewards, masks = Transition(*zip(*memory.transitions))

    discounted_rewards = discount_rewards(rewards, masks)
    action_log_probs = torch.stack(action_log_probs).to(device)
    values = torch.stack(values).to(device)

    actor_loss = (-action_log_probs * (discounted_rewards - values.detach())).mean()
    critic_loss = F.smooth_l1_loss(values, discounted_rewards, reduction='none').mean()

    return (
        actor_loss + critic_loss,
        actor_loss.detach().cpu().numpy(),
        critic_loss.detach().cpu().numpy()
    )


def train_model(model_optimizer, loss):
    model_optimizer.zero_grad()
    loss.backward()
    model_optimizer.step()


def run_training():
    env = gym.make(config.env)

    torch.manual_seed(config.seed)

    model = load_model(env)
    model_optimizer = optim.Adam(model.parameters(), lr=config.lr)
    memory = Memory()

    workers = [Worker(config.seed + i) for i in range(config.worker_count)]

    for training_count in count():
        worker_observations = np.zeros((config.worker_count, len(env.observation_space.high)))
        worker_action_log_probs = torch.zeros((config.worker_count, config.max_timestep))
        worker_values = torch.zeros((config.worker_count, config.max_timestep))
        worker_rewards = np.zeros((config.worker_count, config.max_timestep))
        worker_dones = np.zeros((config.worker_count, config.max_timestep), dtype=np.int8)

        for w, worker in enumerate(workers):
            worker.child.send(('reset', None))
            worker_observations[w] = worker.child.recv()

        for t in range(config.max_timestep):
            for w, worker in enumerate(workers):
                action, worker_action_log_probs[w, t], worker_values[w, t] = model.select_action(worker_observations[w])
                worker.child.send(('step', action))

            for w, worker in enumerate(workers):
                worker_observations[w], worker_rewards[w, t], worker_dones[w, t], info = worker.child.recv()

        episode_total_rewards = []
        episode_durations = []

        for w, worker in enumerate(workers):
            episode_start_index = 0

            for t in range(len(worker_rewards[w])):
                memory.push(
                    worker_action_log_probs[w][t],
                    worker_values[w][t],
                    worker_rewards[w][t],
                    0 if worker_dones[w][t] else 1
                )

                if worker_dones[w][t]:
                    episode_total_rewards.append(np.sum(worker_rewards[w, episode_start_index:t]))
                    episode_durations.append(t)
                    episode_start_index = t + 1

        loss, graphable_actor_loss, graphable_critic_loss = calculate_loss(memory)
        train_model(model_optimizer, loss)
        save_model(model, training_count)
        memory.clear()

        plot.add_point(
            np.mean(episode_total_rewards),
            np.mean(episode_durations),
            graphable_actor_loss,
            graphable_critic_loss
        )
        plot.render()


def run_eval():
    env = gym.make(config.env)

    model = load_model(env, is_eval=True)
    observation = env.reset()

    while True:
        env.render()
        action, action_log_prob, value = model.select_action(observation)
        observation, reward, done, info = env.step(action)

        if done:
            observation = env.reset()


if __name__ == '__main__':
    if config.eval:
        run_eval()
    else:
        run_training()
