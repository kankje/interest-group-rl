import gym
import torch
from torch import optim
from config import config
from itertools import count
from memory import Memory
from model import load_model, save_model


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
