from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt, colors as mcolors

colors = list(mcolors.TABLEAU_COLORS.values())

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

    already_in_position = [1] * data_len
    y_offset = 1
    for cur_label, color in zip(np.unique(data_y), colors):
        positions = []
        for cur_value in data_x[data_y == cur_label]:
            positions.append((cur_value, y_offset * already_in_position[cur_value]))
            already_in_position[cur_value] += 1

        ax.scatter(*zip(*positions), s=100, color=color, label=f"class {cur_label}")

    ax.grid()
    ax.set_ylim(0.5, max(already_in_position))
    ax.legend()
    ax.set_title(title)

    fig.set_figwidth(15)

    return fig, ax


def add_split_lines_to_plot(ax: plt.Axes, split_lines: np.ndarray, labels: np.ndarray, x_offset=0) -> plt.Axes:
    """
    Adds lines indicating the interval splits to plot like in the paper
    :param ax: matplotlib Axes
    :param split_lines:  numpy array from the fusinter splitting algorithm
    :param labels:  numpy array of labels from the fusinter splitting algorithm
    :param x_offset: scalar offset for the splitting lines. Can be set for better visibility.
    :return: The matplotlib Axes object given to the function
    """
    label_types = np.unique(labels)
    split_lines = np.hstack((split_lines, split_lines[-1] + 1))
    label_types = np.roll(label_types, -1)
    y_min, y_max = ax.get_ylim()
    for label_type, color in zip(label_types, colors):
        label_idx = labels == label_type
        cur_splits = split_lines[label_idx]

        if label_type == -1:
            label = "split class mixed"
        else:
            label = f"split class {label_type}"

        ax.vlines(x=cur_splits + x_offset, linewidth=3, ymin=y_min, ymax=y_max, color=color,
                  label=label)

        ax.legend()
    return ax
