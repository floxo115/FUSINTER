import numpy as np

from .splitter import Splitter
from .table_manager import TableManager
from .merge_value_computer import MergeValueComputer


# TODO Not needed anymore, but I keep them for now for reference
# def shannon_entropy(input_table: np.ndarray, alpha, lam) -> float:
#     """
#     Implementation of the Shannon Entropy formula on page 12 of the paper
#     :param input_table: a numpy array in the form of a table generated by FUSINTER
#     :param alpha: a scalar weight parameter (see the paper)
#     :param lam: a scalar weight parameter (see the paper)
#     :return: a scalar value for estimation splits
#     """
#     m, k = input_table.shape
#     n = np.sum(input_table)
#     result = 0
#     for j in range(k):
#         n_j = np.sum(input_table[:, j])
#         col_fac = alpha * n_j / n
#         col_sum = 0
#         for i in range(m):
#             p = (input_table[i, j] + lam) / (n_j + m * lam)
#             col_sum += -(p * np.log2(p))
#         result += col_fac * col_sum + (1 - alpha) * m * lam / n_j
#
#     return result
#
#
# def quadratic_entropy(input_table: np.ndarray, alpha, lam) -> float:
#     """
#     Implementation of the Quadratic Entropy formula on page 12 of the paper
#     :param input_table: a numpy array in the form of a table generated by FUSINTER
#     :param alpha: a scalar weight parameter (see the paper)
#     :param lam: a scalar weight parameter (see the paper)
#     :return: a scalar value for estimation splits
#     """
#     m, k = input_table.shape
#     n = np.sum(input_table)
#     result = 0
#     for j in range(k):
#         n_j = np.sum(input_table[:, j])
#         col_fac = alpha * n_j / n
#         col_sum = 0
#         for i in range(m):
#             p = (input_table[i, j] + lam) / (n_j + m * lam)
#             col_sum += p * (1 - p)
#         result += col_fac * col_sum + (1 - alpha) * m * lam / n_j
#
#     return result


class FUSINTERDiscretizer:
    """
    very primitive implementation of the FUSINTER discretization algorithm
    """

    def __init__(self,
                 data_x: np.ndarray,
                 data_y: np.ndarray,
                 splitter=Splitter,
                 table_manager=TableManager,
                 # entropy_func=quadratic_entropy,
                 ):
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

        self.splitter = splitter(self.data_x, self.data_y)
        self.table_manager = table_manager(self.data_x, self.data_y)
        self.merge_value_computer = MergeValueComputer
        #self.entropy_func = entropy_func

    def apply(self, alpha=0.975, lam=1.) -> np.ndarray:
        """
        :alpha: alpha float parameter for the entropy merge criterion
        :lam: lambda float parameter for the entropy merge criterion
        :return: a numpy array of continuous interval split points for the discretization of the classes dataset
        """

        splits, _ = self.get_initial_intervals()
        table = self.create_table(splits)

        mvc = MergeValueComputer(table, alpha, lam)
        while len(splits) >= 1:
            split_values = np.round(mvc.get_all_deltas(), 15)

            max_ind = np.argmax(split_values).item()
            if split_values[max_ind] <= 0:
                break

            # TODO find best way to do the update
            splits = np.delete(splits, max_ind)
            # table[:, max_ind] = table[:, max_ind + 1]
            # table = np.delete(table, max_ind + 1, axis=1)
            table = self.table_manager.compress_table(table, max_ind)

            mvc.update(table, max_ind)

        return splits

    def get_initial_intervals(self):
        return self.splitter.apply()

    def create_table(self, init_splits: np.ndarray) -> np.ndarray:
        """
        creates a table from the initial splits of the data
        :return: np.matrix with k columns and n rows from k splits and n classes
        """
        return self.table_manager.create_table(init_splits)

    def compress_table(self, input_table: np.ndarray, i: int) -> np.ndarray:
        """
        returns a table like the one returned by create_table but with the i-th and (i+1)-th columns merged
        :param input_table: a table like the one returned by create_table
        :param i: the index for the merge
        :return: a table like the input but with 2 consecutive columns merged
        """
        return self.table_manager.compress_table(input_table, i)
