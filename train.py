from itertools import count
import gym
import torch
from torch import optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader
import numpy as np
from config import config, device, eps
from memory import Memory, Transition
from model import load_model, save_model
from worker import Worker
import plot


def normalize(tensor):
    return (tensor - tensor.mean()) / (tensor.std() + eps)


def calculate_advantages(rewards, values, masks):
    rewards = torch.Tensor(rewards).to(device)
    advantages = torch.zeros_like(rewards)

    previous_value = 0
    running_advantage = 0

    for t in reversed(range(len(rewards))):
        running_tderror = rewards[t] + config.gamma * previous_value * masks[t] - values[t]
        running_advantage = running_tderror + (config.gamma * config.lambda_) * running_advantage * masks[t]

        previous_value = values[t]
        advantages[t] = running_advantage

    return advantages


def calculate_loss(actions, advantages, old_policies, old_values, policies, values):
    policies_distribution = Categorical(policies)
    old_policies_distribution = Categorical(old_policies)

    sampled_return = old_values + advantages
    sampled_normalized_advantage = normalize(advantages)

    # Policy

    log_policies = policies_distribution.log_prob(actions)
    old_log_policies = old_policies_distribution.log_prob(actions)

    ratio = torch.exp(log_policies - old_log_policies)

    clipped_ratio = ratio.clamp(min=1.0 - config.epsilon, max=1.0 + config.epsilon)
    actor_loss = torch.min(ratio * sampled_normalized_advantage, clipped_ratio * sampled_normalized_advantage).mean()

    # Entropy Bonus

    entropy_loss = policies_distribution.entropy().mean()

    # Value

    clipped_value = old_values + (values - old_values).clamp(min=-config.epsilon, max=config.epsilon)
    critic_loss = 0.5 * torch.max((values - sampled_return) ** 2, (clipped_value - sampled_return) ** 2).mean()
    loss = -(actor_loss - 0.5 * critic_loss + 0.01 * entropy_loss)

    return loss, actor_loss.item(), critic_loss.item(), entropy_loss.item()


def train_model(model, model_optimizer, memory):
    observations, actions, rewards, masks = Transition(*zip(*memory.transitions))

    observations = torch.Tensor(observations).to(device)

    old_policies, old_values = model(observations)
    old_policies = old_policies.detach()
    old_values = old_values.detach()
    advantages = calculate_advantages(rewards, old_values, masks)

    data_loader = DataLoader(
        dataset=list(zip(observations, actions, advantages, old_policies, old_values)),
        batch_size=config.batch_size,
        shuffle=True
    )

    graphable_actor_losses = []
    graphable_critic_losses = []
    graphable_entropy_losses = []

    for _ in range(config.epochs):
        for _, sample in enumerate(data_loader):
            observations_sample, actions_sample, advantages_sample, old_policies_sample, old_values_sample = sample

            policies, values = model(observations_sample)
            loss, graphable_actor_loss, graphable_critic_loss, graphable_entropy_loss = calculate_loss(
                actions_sample.to(device),
                advantages_sample.to(device),
                old_policies_sample.to(device),
                old_values_sample.to(device),
                policies,
                values
            )

            graphable_actor_losses.append(graphable_actor_loss)
            graphable_critic_losses.append(graphable_critic_loss)
            graphable_entropy_losses.append(graphable_entropy_loss)

            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

    return np.mean(graphable_actor_losses), np.mean(graphable_critic_losses), np.mean(graphable_entropy_losses)


def run_training():
    env = gym.make(config.env)

    torch.manual_seed(config.seed)

    model = load_model(env)
    model_optimizer = optim.Adam(model.parameters(), lr=config.lr)
    memory = Memory()

    workers = [Worker(config.seed + i) for i in range(config.worker_count)]

    for training_count in count():
        worker_observations = np.zeros((config.worker_count, config.max_timestep, len(env.observation_space.high)))
        worker_actions = np.zeros((config.worker_count, config.max_timestep))
        worker_rewards = np.zeros((config.worker_count, config.max_timestep))
        worker_dones = np.zeros((config.worker_count, config.max_timestep), dtype=np.int8)

        for w, worker in enumerate(workers):
            worker.child.send(('reset', None))
            worker_observations[w, 0] = worker.child.recv()

        for t in range(config.max_timestep):
            for w, worker in enumerate(workers):
                worker_actions[w, t] = model.select_action(worker_observations[w, t])
                worker.child.send(('step', int(worker_actions[w, t])))

            for w, worker in enumerate(workers):
                observation, worker_rewards[w, t], worker_dones[w, t], info = worker.child.recv()

                if t + 1 < config.max_timestep:
                    worker_observations[w, t + 1] = observation

        episode_total_rewards = []
        episode_durations = []

        for w, worker in enumerate(workers):
            episode_start_index = 0

            for t in range(len(worker_rewards[w])):
                memory.push(
                    worker_observations[w][t],
                    worker_actions[w][t],
                    worker_rewards[w][t],
                    0 if worker_dones[w][t] else 1
                )

                if worker_dones[w][t]:
                    episode_total_rewards.append(np.sum(worker_rewards[w, episode_start_index:t]))
                    episode_durations.append(t)
                    episode_start_index = t + 1

        graphable_actor_loss, graphable_critic_loss, graphable_entropy_loss = train_model(
            model,
            model_optimizer,
            memory
        )

        save_model(model, training_count)
        memory.clear()

        plot.add_point(
            np.mean(episode_total_rewards),
            np.mean(episode_durations),
            graphable_actor_loss,
            graphable_critic_loss,
            graphable_entropy_loss
        )
        plot.render()


def run_eval():
    env = gym.make(config.env)

    model = load_model(env, is_eval=True)
    observation = env.reset()

    while True:
        env.render()
        action = model.select_action(observation)
        observation, reward, done, info = env.step(action)

        if done:
            observation = env.reset()


if __name__ == '__main__':
    if config.eval:
        run_eval()
    else:
        run_training()
