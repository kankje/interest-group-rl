from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from config import config, device
import numpy as np


class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs, is_eval=False):
        super().__init__()

        self.is_eval = is_eval

        self.shared_fc1 = nn.Linear(num_inputs, 128)
        self.actor_fc1 = nn.Linear(128, num_outputs)
        self.critic_fc1 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.shared_fc1(x))
        policy = F.softmax(self.actor_fc1(x), dim=-1)
        value = self.critic_fc1(x)

        return policy, value

    def select_action(self, observation, legal_actions):
        observation = torch.from_numpy(observation).float().to(device)
        policies, value = self(observation)
        masked_policies = policies * torch.from_numpy(legal_actions).to(device)

        if torch.isclose(torch.sum(masked_policies), torch.tensor(0.)):
            return np.random.choice(np.where(legal_actions == 1)[0].astype(int))

        categorical_distribution = Categorical(masked_policies)

        if self.is_eval:
            selected_action = torch.argmax(categorical_distribution.probs)
        else:
            selected_action = categorical_distribution.sample()

        return selected_action.item()


def load_model(env, is_eval=False, training_count=None):
    model = Model(num_inputs=9, num_outputs=env.action_space.n, is_eval=is_eval)
    model.to(device)

    filename = config.filename if training_count is None else config.history_filename.format(training_count)
    if path.exists(filename):
        model.load_state_dict(torch.load(filename))

    if is_eval:
        model.eval()

    return model


def save_model(model, training_count):
    torch.save(model.state_dict(), config.filename)

    if training_count > 0 and training_count % 50 == 0:
        torch.save(model.state_dict(), config.history_filename.format(training_count))
