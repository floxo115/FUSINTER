import numpy as np

from .merge_value_computer import MergeValueComputer
from .splitter import Splitter
from .table_manager import TableManager


class FUSINTERDiscretizer:

    def __init__(self, alpha=0.95, lam=1):
        self.alpha = alpha
        self.lam = lam
        self.data_x = np.array([])
        self.data_x = np.array([])
        self.computed_splits = np.array([])

    def _init_dataset(self,
                      data_x: np.ndarray,
                      data_y: np.ndarray,
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

        self.splitter = Splitter(self.data_x, self.data_y)
        self.table_manager = TableManager(self.data_x, self.data_y)
        self.merge_value_computer = MergeValueComputer

    def transform(self, data):
        return np.digitize(data, self.computed_splits)

    def fit(self, data_x: np.ndarray, data_y: np.ndarray) -> np.ndarray:
        """
        :alpha: alpha float parameter for the entropy merge criterion
        :lam: lambda float parameter for the entropy merge criterion
        :return: a numpy array of continuous interval split points for the discretization of the classes dataset
        """

        self._init_dataset(data_x, data_y)
        splits, _ = self.get_initial_intervals()
        table = self.create_table(splits)

        mvc = MergeValueComputer(table, self.alpha, self.lam)
        while len(splits) >= 1:
            split_values = np.round(mvc.get_all_deltas(), 5)

            max_ind = np.argmax(split_values).item()
            if split_values[max_ind] <= 0:
                break

            splits = np.delete(splits, max_ind)
            table = self.table_manager.compress_table(table, max_ind)

            mvc.update(table, max_ind)

        self.computed_splits = splits
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
