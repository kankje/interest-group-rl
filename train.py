import gym
import torch
from torch import optim
from config import config, device, eps
from itertools import count
from memory import Memory, Transition
from model import load_model, save_model


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
    action_log_probs, rewards, masks = Transition(*zip(*memory.transitions))

    discounted_rewards = discount_rewards(rewards, masks)
    action_log_probs = torch.stack(action_log_probs).to(device)

    loss = (-action_log_probs * discounted_rewards).mean()

    return loss, loss.detach().cpu().numpy()


def train_model(model_optimizer, loss):
    model_optimizer.zero_grad()
    loss.backward()
    model_optimizer.step()


def run_training():
    env = gym.make(config.env)

    env.seed(config.seed)
    torch.manual_seed(config.seed)

    model = load_model(env)
    model_optimizer = optim.Adam(model.parameters(), lr=config.lr)
    memory = Memory()

    for episode_number in count():
        observation = env.reset()

        for _ in range(config.max_timestep):
            action, action_log_prob = model.select_action(observation)
            observation, reward, done, info = env.step(action)
            memory.push(action_log_prob, reward, 0 if done else 1)

            if done:
                break

        loss, graphable_loss = calculate_loss(memory)
        train_model(model_optimizer, loss)
        save_model(model, episode_number)
        memory.clear()


def run_eval():
    env = gym.make(config.env)

    model = load_model(env, is_eval=True)
    observation = env.reset()

    while True:
        env.render()
        action, action_log_prob = model.select_action(observation)
        observation, reward, done, info = env.step(action)

        if done:
            observation = env.reset()


if __name__ == '__main__':
    if config.eval:
        run_eval()
    else:
        run_training()
