import argparse
import torch
import torch.multiprocessing as multiprocessing
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--eval', action='store_true')
parser.add_argument('--env', default='LunarLander-v2')
parser.add_argument('--seed', type=int, default=543)
parser.add_argument('--worker-count', type=int, default=multiprocessing.cpu_count())
parser.add_argument('--max-timestep', type=int, default=1000)
parser.add_argument('--filename', default='models/lunar.ppo.pth')
parser.add_argument('--history-filename', default='models/lunar.ppo.{}.pth')
parser.add_argument('--graph-filename', default='graphs/lunar.ppo.png')

parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lambda', dest='lambda_', type=float, default=0.96)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--epsilon', type=float, default=0.2)

config = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eps = np.finfo(np.float32).eps.item()
