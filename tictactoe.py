import gym
import gym.spaces
import numpy as np


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['ansi']}

    def __init__(self):
        #  0  0  0
        #  0  1  0
        #  0  0  0
        self.action_space = gym.spaces.Discrete(9)

        # 1 = ❌, -1 = ⭕️
        #  0  0 -1
        #  0  1  0
        #  0  0  0
        self.observation_space = gym.spaces.Box(
            np.array([
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
            ]),
            np.array([
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ])
        )

        self.reset()

    # Returns (observation, reward, is_done, info)
    def _step(self, action):
        x, y = np.unravel_index(action, self.observation_space.high.shape, order='F')

        if not self.is_legal_action(x, y):
            raise ValueError('Invalid action {},{} on {}'.format(x, y, self.state))

        self.take_action(x, y)

        if self.is_tie():
            return self.get_observation(), 0, True, {}

        winner = self.get_winner()
        if winner is None:
            self.player_in_turn = 1 if self.player_in_turn == -1 else -1
            return self.get_observation(), 10, False, {}

        reward = 100 if winner == self.player_in_turn else -100
        return self.get_observation(), reward, True, {}

    def _reset(self):
        self.state = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        # 1 = ❌, -1 = ⭕️
        self.player_in_turn = 1

        return self.get_observation()

    def _render(self, mode='human', close=False):
        if mode == 'rgb_array' or mode == 'human':
            return

        def map_state(n):
            if n == 0:
                return '-'
            elif n == 1:
                return 'x'
            elif n == -1:
                return 'o'

        def map_row(row):
            new_row = list(map(map_state, row))
            return f'{new_row[0]} {new_row[1]} {new_row[2]}'

        res = map(map_row, self.state)
        list_res = [i for i in list(res)]

        print(f'{list_res[0]}\n{list_res[1]}\n{list_res[2]}')

    def _seed(self, seed=None):
        return []

    def is_legal_action(self, x, y):
        return self.state[y][x] == 0

    def get_legal_actions(self):
        return 1 - np.abs(np.array(self.state).flatten())

    def take_action(self, x, y):
        self.state[y][x] = self.player_in_turn

    def get_observation(self):
        return np.array(self.state).flatten() * self.player_in_turn

    def is_tie(self):
        return np.count_nonzero(self.state) == 9 and self.get_winner() is None

    def get_winner(self):
        for row in range(3):
            player = self.state[row][0]
            if player != 0 and player == self.state[row][1] == self.state[row][2]:
                return player

        for column in range(3):
            player = self.state[0][column]
            if player != 0 and player == self.state[1][column] == self.state[2][column]:
                return player

        player = self.state[0][0]
        if player != 0 and player == self.state[1][1] == self.state[2][2]:
            return player

        player = self.state[0][2]
        if player != 0 and player == self.state[1][1] == self.state[2][0]:
            return player

        return None


gym.envs.register(
    id='TicTacToe-v0',
    entry_point='tictactoe:TicTacToeEnv',
)
