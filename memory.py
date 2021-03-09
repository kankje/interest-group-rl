from collections import namedtuple

Transition = namedtuple('Transition', ('observation', 'action', 'reward', 'mask'))


class Memory:
    def __init__(self):
        self.transitions = []

    def push(self, *args):
        self.transitions.append(Transition(*args))

    def clear(self):
        self.transitions = []

    def __len__(self):
        return len(self.transitions)
