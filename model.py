from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from config import config, device


class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs, is_eval=False):
        super().__init__()

        self.is_eval = is_eval

        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy = F.softmax(self.fc2(x), dim=-1)

        return policy

    def select_action(self, observation):
        observation = torch.from_numpy(observation).float().to(device)
        action_probs = self(observation)
        categorical_distribution = Categorical(action_probs)

        if self.is_eval:
            selected_action = torch.argmax(categorical_distribution.probs)
        else:
            selected_action = categorical_distribution.sample()

        return (
            selected_action.item(),
            categorical_distribution.log_prob(selected_action),
        )


def load_model(env, is_eval=False):
    model = Model(num_inputs=len(env.observation_space.high), num_outputs=env.action_space.n, is_eval=is_eval)
    model.to(device)

    if path.exists(config.filename):
        model.load_state_dict(torch.load(config.filename))

    if is_eval:
        model.eval()

    return model


def save_model(model, training_count):
    torch.save(model.state_dict(), config.filename)

    if training_count > 0 and training_count % 500 == 0:
        torch.save(model.state_dict(), config.history_filename.format(training_count))
