from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config, device


class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy = F.softmax(self.fc2(x), dim=-1)

        return policy


def load_model(env, is_eval=False):
    model = Model(num_inputs=len(env.observation_space.high), num_outputs=env.action_space.n)
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
