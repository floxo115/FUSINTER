from typing import Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

# The dataset from the paper
paper_raw_dataset = np.array([
    *[1, 0] * 1,
    *[2, 1] * 1,
    *[3, 0] * 2,
    *[4, 0] * 2,
    *[5, 0] * 3,
    *[6, 0] * 3,
    *[7, 0] * 3,
    *[8, 0] * 2,
    *[9, 0] * 3,
    *[10, 0] * 3,
    *[11, 0] * 3,
    *[12, 0] * 2,
    *[13, 0] * 2,
    *[13, 1] * 1,
    *[14, 0] * 3,
    *[15, 1] * 3,
    *[16, 0] * 3,
    *[17, 1] * 3,
    *[18, 0] * 2,
    *[18, 1] * 1,
    *[19, 1] * 2,
    *[20, 0] * 2,
    *[22, 0] * 3,
    *[23, 1] * 3,
    *[27, 1] * 2,
    *[28, 1] * 3,
    *[29, 1] * 3,
    *[30, 1] * 3,
    *[31, 1] * 3,
    *[33, 1] * 1,
    *[34, 1] * 3,
    *[35, 1] * 3,
    *[36, 1] * 3,
    *[37, 1] * 2,
    *[37, 0] * 1,
    *[38, 0] * 2,
    *[38, 1] * 1,
    *[39, 1] * 2,
    *[40, 0] * 2,
]).reshape(-1, 2)

paper_dataset_x = paper_raw_dataset[:, 0]
paper_dataset_y = paper_raw_dataset[:, 1]


def get_plot_for_paper_data(data_x: np.ndarray, data_y: np.ndarray, title="") -> Tuple[plt.Figure, plt.Axes]:
    """
    creates a plot as given in the paper for data given in the paper
    :param data_x: array of integer values
    :param data_y:  array of integer labels
    :param title: optional title of the plot
    :return: Matplotlib Figure and Axes for plotting
    """
    fig, ax = plt.subplots(1, 1)
    data_len = len(data_x)

    already_in_position = [1]*data_len
    y_offset = 1
    for cur_label in np.unique(data_y):
        positions = []
        for cur_value in data_x[data_y == cur_label]:
            positions.append((cur_value, y_offset * already_in_position[cur_value]))
            already_in_position[cur_value] += 1

        ax.scatter(*zip(*positions), label=f"class {cur_label}")

    ax.grid()
    ax.set_ylim(0.5,max(already_in_position))
    ax.legend()
    ax.set_title(title)
    return fig, ax