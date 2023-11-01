import numpy as np

from .splitter import Splitter


class FUSINTERDiscretizer:
    """
    very primitive implementation of the FUSINTER discretization algorithm
    """

    def __init__(self, data_x: np.ndarray, data_y: np.ndarray, splitter=Splitter):
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

        label_masks = []
        for label in np.unique(self.data_y):
            label_masks.append(self.data_y == label)

        for i, label_mask in enumerate(label_masks):
            self.data_y[label_mask] = i

        # sort the given dataset
        sort_idx = np.argsort(self.data_x)
        self.data_y = self.data_y[sort_idx]
        self.data_x = self.data_x[sort_idx]

        self.splitter = Splitter(self.data_x, self.data_y)

    # TODO Test and Document
    def apply(self) -> np.ndarray:
        """
        :return: a numpy array of continuous interval split points for the discretization of the classes dataset
        """

        splits, _ = self.get_initial_intervals()
        table = self.create_table(splits)

        while len(splits) > 1:
            merged_tables = []
            for i, _ in enumerate(splits):
                merged_tables.append(self.compress_table(table, i))

            split_values = np.zeros(len(merged_tables), dtype=np.float64)
            for i, merged_table in enumerate(merged_tables):
                split_values[i] = shannon_entropy(table) - shannon_entropy(merged_table)

            max_ind = np.argmax(split_values)
            if split_values[max_ind] <= 0:
                break

            splits = np.delete(splits, max_ind)
            table = merged_tables[max_ind]

        return splits

    def get_initial_intervals(self):
        return self.splitter.apply()

    def create_table(self, init_splits: np.ndarray) -> np.ndarray:
        """
        creates a table from the initial splits of the data
        :return: np.matrix with k columns and n rows from k splits and n classes
        """

        n_labels = len(np.unique(self.data_y))
        n_splits = len(init_splits) + 1

        table = np.zeros((n_labels, n_splits), dtype=int)

        n_labels_in_interval = np.zeros(n_labels, dtype=int)
        i = 0

        for split_idx, split_val in enumerate(init_splits):
            while self.data_x[i] < split_val:
                n_labels_in_interval[self.data_y[i]] += 1
                i += 1

            table[:, split_idx] = n_labels_in_interval
            n_labels_in_interval[:] = 0

        while i < len(self.data_x):
            n_labels_in_interval[self.data_y[i]] += 1
            i += 1

        table[:, n_splits - 1] = n_labels_in_interval
        n_labels_in_interval[:] = 0
        return table

    def compress_table(self, input_table: np.ndarray, i: int) -> np.ndarray:
        """
        returns a table like the one returned by create_table but with the i-th and (i+1)-th columns merged
        :param input_table: a table like the one returned by create_table
        :param i: the index for the merge
        :return: a table like the input but with 2 consecutive columns merged
        """
        n, m = input_table.shape

        if i < 0 or (m - 1) < i:
            raise ValueError(f"the parameter i has to have values between 0 and (len of columns -1), but is {i}")

        new_table = np.zeros((n, m - 1), dtype=int)

        # We iterate over the new and the old table to compute the values for the new table
        init_table_index, new_table_index = 0, 0
        while init_table_index < m:
            if init_table_index == i:
                # if we are in the first column of the merged columns we add their values together and insert the result
                # into the new table
                new_table[:, new_table_index] = np.sum(input_table[:, init_table_index:(init_table_index + 2)], axis=1)
                new_table_index += 1
            elif init_table_index != i + 1:
                # if we are in no column to be merged we simply copy the values to the new table
                new_table[:, new_table_index] = input_table[:, init_table_index]
                new_table_index += 1

            # the case of being in the second merge column has no actions except not incrementing the new table index
            # in any case the init table index has to be incremented
            init_table_index += 1

        return new_table


def shannon_entropy(input_table: np.ndarray, alpha=0.975, lam=1) -> float:
    """
    Implementation of the Shannon Entropy formula on page 12 of the paper
    :param input_table: a numpy array in the form of a table generated by FUSINTER
    :param alpha: a scalar weight parameter (see the paper)
    :param lam: a scalar weight parameter (see the paper)
    :return: a scalar value for estimation splits
    """
    m, k = input_table.shape
    n = np.sum(input_table)
    result = 0
    for j in range(k):
        n_j = np.sum(input_table[:, j])
        col_fac = alpha * n_j / n
        col_sum = 0
        for i in range(m):
            p = (input_table[i, j] + lam) / (n_j + m * lam)
            col_sum += -(p * np.log2(p))
        result += col_fac * col_sum + (1 - alpha) * m * lam / n_j

    return result
