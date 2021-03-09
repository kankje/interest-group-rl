import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import index_of
from config import config

fig, axs = plt.subplots(nrows=2, ncols=2, num=1)
axs[0][0].set_title('Reward')
axs[0][1].set_title('Duration')
axs[1][0].set_title('Actor loss')
axs[1][1].set_title('Critic loss')

lines = []
rolling_lines = []
for ax in axs.flat:
    lines.append(ax.plot([])[0])
    rolling_lines.append(ax.plot([])[0])

plt.tight_layout()

rolling_period = 50
data = [[] for line in lines]
rolling_data = [[] for line in rolling_lines]


def add_point(*args):
    for index, value in enumerate(args):
        data[index].append(value)
        rolling_data[index].append(np.mean(data[index][-rolling_period:]))


def render():
    for index, ax in enumerate(axs.flat):
        x, y = index_of(data[index])
        lines[index].set_data(x, y)

        x, y = index_of(rolling_data[index])
        rolling_lines[index].set_data(x, y)

        ax.relim()
        ax.autoscale_view()

    plt.draw()
    plt.pause(0.001)
    plt.savefig(config.graph_filename)
