from typing import Tuple

import numpy as np


class Splitter:
    """
    Class to compute the initial intervals of the FUSINTER algorithm
    """

    def __init__(self, data_x: np.ndarray, data_y: np.ndarray):
        """
        :param data_x: data values (assumed to be ordered)
        :param data_y: data labels (assumed to be in the range 0...K for K unique labels)
        """

        self.data_x = data_x
        self.data_y = data_y

    def _get_label_of_next_value(self, index: int) -> Tuple[int, int]:
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

    def apply(self):
        """
        :return: a numpy array of initial split points as described in step 2 of the algorithm
        """
        splits = []
        labels = []
        index = 0
        # we process the first value to get its label
        label, index = self._get_label_of_next_value(index)
        labels.append(label)

        while index < len(self.data_x):
            # for each value we compare the label to the one before and add splits accordingly
            label, index = self._get_label_of_next_value(index)
            # if the label to the one before differs OR if the label is -1 we add a split
            if label != labels[-1] or labels[-1] == -1:
                splits.append(self.data_x[index - 1])
                labels.append(label)

        return np.array(splits, dtype=np.float64), np.array(labels, dtype=np.int32)
