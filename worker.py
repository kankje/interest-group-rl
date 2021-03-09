import gym
import torch.multiprocessing as multiprocessing
from config import config


def worker_process(remote, seed):
    env = gym.make(config.env)
    env.seed(seed)

    while True:
        cmd, data = remote.recv()

        if cmd == 'reset':
            remote.send(env.reset())
        elif cmd == 'step':
            observation, reward, done, info = env.step(data)
            if done:
                observation = env.reset()
            remote.send((observation, reward, done, info))
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError


class Worker:
    def __init__(self, seed):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, seed))
        self.process.start()
