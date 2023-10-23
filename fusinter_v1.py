from typing import Tuple, List

import numpy as np


class FUSINTERDiscretizer:
    """
    very primitive implementation of the FUSINTER discretization algorithm
    """

    def __init__(self, data_x: np.ndarray, data_y: np.ndarray):
        """
        :param data_x: a 1D array of values
        :param data_y: a 1D array of labels
        """
        if not isinstance(data_x, np.ndarray):
            raise ValueError("data_x should be a numpy array")
        if data_x.ndim != 1:
            raise ValueError(f"the number of dims of data_x should be 1 {data_y.shape}")

        if not isinstance(data_y, np.ndarray):
            raise ValueError("data_y should be a numpy array")
        if data_y.ndim != 1:
            raise ValueError(f"the number of dims of data_y should be 1 {data_y.shape}")
        if not np.issubdtype(data_y.dtype, np.integer):
            raise ValueError(f"the dtype of data_y should be int32, but is {data_y.dtype}")

        if len(data_x) != len(data_y):
            raise ValueError("the given arrays should be of the same size, but are not")

        self.data_x = data_x.copy()
        self.data_y = data_y.copy()

        # sort the given dataset
        sort_idx = np.argsort(self.data_x)
        self.data_y = self.data_y[sort_idx]
        self.data_x = self.data_x[sort_idx]

    def apply(self) -> np.ndarray:
        """
        :return: a numpy array of continuous interval split points for the discretization of the classes dataset
        """

        self.get_initial_intervals()
        return np.array([])

    def get_initial_intervals(self):
        """
        :return: a numpy array of initial split points as described in step 2 of the algorithm
        """

        def get_label_of_next_value(index: int) -> Tuple[int, int]:
            """
            returns the label of the next value from a array of value
            :param index: starting index
            :return: Tuple of label and end index
            """
            value = self.data_x[index]
            label = self.data_y[index]
            index += 1
            while index < len(self.data_x):
                if value != self.data_x[index]:
                    break

                if label != self.data_y[index]:
                    label = -1

                index += 1

            return label, index

        splits = []
        labels = []
        index = 0
        # we process the first value to get its label
        label, index = get_label_of_next_value(index)
        labels.append(label)

        while index < len(self.data_x):
            # for each value we compare the label to the one before and add splits accordingly
            label, index = get_label_of_next_value(index)
            # if the label to the one before differs OR if the label is -1 we add a split
            if label != labels[-1] or labels[-1] == -1:
                splits.append(self.data_x[index - 1])
                labels.append(label)

        return np.array(splits, dtype=np.float64), np.array(labels, dtype=np.int32)

    def create_table(self, init_splits: np.ndarray):
        """
        creates a table from the initial splits of the data
        :return: np.matrix with k columns and n rows from k splits and n classes
        """

        raise NotImplementedError()
