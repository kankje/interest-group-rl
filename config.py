import argparse
import torch
import torch.multiprocessing as multiprocessing
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--eval', action='store_true')
parser.add_argument('--env', default='TicTacToe-v0')
parser.add_argument('--seed', type=int, default=543)
parser.add_argument('--worker-count', type=int, default=multiprocessing.cpu_count())
parser.add_argument('--max-timestep', type=int, default=500)  # At least 100 games
parser.add_argument('--filename', default='models/tictactoe.ppo.pth')
parser.add_argument('--history-filename', default='models/tictactoe.ppo.{}.pth')
parser.add_argument('--graph-filename', default='graphs/tictactoe.ppo.png')

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lambda', dest='lambda_', type=float, default=0.96)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--epsilon', type=float, default=0.2)

config = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eps = np.finfo(np.float32).eps.item()
