import math
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
import plot
import tictactoe


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

    save_model(model, 0)

    env = gym.make(config.env)
    env.seed(config.seed)

    for training_count in count():
        observations = np.zeros((config.max_timestep, 9))
        actions = np.zeros((config.max_timestep,))
        rewards = np.zeros((config.max_timestep,))
        dones = np.zeros((config.max_timestep,), dtype=np.int8)

        observations[0] = env.reset()
        opponent_model = load_model(
            env,
            is_eval=True,
            training_count=max(0, math.floor(max(0, training_count - 1) / 50) * 50)
        )

        is_model_x = True

        for t in range(config.max_timestep):
            # We make a move

            actions[t] = model.select_action(observations[t], env.get_legal_actions())
            observation, rewards[t], dones[t], _ = env.step(int(actions[t]))

            if not dones[t]:
                # If game not done, opponent makes move
                action = opponent_model.select_action(observation, env.get_legal_actions())
                observation, reward, dones[t], _ = env.step(int(action))

                if dones[t]:
                    rewards[t] = -reward

            if dones[t]:
                observation = env.reset()
                is_model_x = not is_model_x

                if not is_model_x:
                    # If we're playing as O, opponent makes first move
                    action = opponent_model.select_action(observation, env.get_legal_actions())
                    observation, _, _, _ = env.step(int(action))

            if t + 1 < config.max_timestep:
                observations[t + 1] = observation

        episode_total_rewards = []
        episode_durations = []

        episode_start_index = 0

        for t in range(len(rewards)):
            memory.push(
                observations[t],
                actions[t],
                rewards[t],
                0 if dones[t] else 1
            )

            if dones[t]:
                episode_total_rewards.append(np.sum(rewards[episode_start_index:t + 1]))
                episode_durations.append(t - episode_start_index + 1)
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
    is_model_x = True

    def get_user_action():
        x, y = [int(value.strip()) for value in input('\nYour move: ').split(',')]
        return (x - 1) + (y - 1) * 3

    while True:
        action = model.select_action(observation, env.get_legal_actions())
        observation, reward, done, _ = env.step(action)

        print('\nAfter opponent move')
        env.render(mode='ansi')

        if reward == 100:
            print('\n--- YOU LOSE ---\n')

        if not done:
            observation, reward, done, _ = env.step(get_user_action())

            if reward == 100:
                print('\n--- YOU WIN ---\n')

        if done:
            if reward == 0:
                print('\n--- TIE ---\n')

            observation = env.reset()
            is_model_x = not is_model_x

            if not is_model_x:
                observation, _, _, _ = env.step(get_user_action())


if __name__ == '__main__':
    if config.eval:
        run_eval()
    else:
        run_training()
