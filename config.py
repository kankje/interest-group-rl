import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--eval', action='store_true')
parser.add_argument('--env', default='CartPole-v1')
parser.add_argument('--seed', type=int, default=543)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max-timestep', type=int, default=1000)
parser.add_argument('--filename', default='models/cartpole.pth')
parser.add_argument('--history-filename', default='models/cartpole.{}.pth')

config = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
