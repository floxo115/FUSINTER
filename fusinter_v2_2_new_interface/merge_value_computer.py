import numpy as np
from numba import jit

@jit(nopython=True)
def shannon_entropy(input_column: np.ndarray, alpha, lam, m, n) -> float:
    """
    Implementation of the Shannon Entropy for one column formula on page 12 of the paper
    :param input_column: a numpy array in the form of a column generated by FUSINTER
    :param alpha: a scalar weight parameter (see the paper)
    :param lam: a scalar weight parameter (see the paper)
    :m: the number of classes
    :n: sum of all components of the table where column is from
    :return: a scalar value for estimation splits
    """
    col_sum = 0
    n_j = np.sum(input_column)
    col_fac = alpha * n_j / n
    for i in range(m):
        p = (input_column[i] + lam) / (n_j + m * lam)
        col_sum += -(p * np.log2(p))
    result = col_fac * col_sum + (1 - alpha) * m * lam / n_j

    return result


class MergeValueComputer:
    """
    Computes phi values for split tables generated by the FUSINTER algorithm.
    It memorizes the entropies of the columns of a given split table and then
    computes the necessary values for removing split points with memoization
    techniques.
    """
    def __init__(self, input_table, alpha, lam, entropy_func=shannon_entropy):
        self.table = input_table
        self.alpha = alpha
        self.lam = lam

        # compute necessary values for updating the split values
        self.m = input_table.shape[0]
        self.k = input_table.shape[1]
        self.n = np.sum(input_table)
        self.entropy_func = lambda x: entropy_func(x, self.alpha, self.lam, self.m, self.n)

        # compute the entropy for all columns of init table
        self.cols_entropy = np.apply_along_axis(self.entropy_func, 0, self.table)

        # compute the init deltas for all possible splits of the init table
        self.deltas = np.zeros(self.k-1)
        for col_idx in range(self.k-1):
            self.deltas[col_idx] = self.compute_delta(col_idx)


    def compute_merge_entropy(self, col_idx):
        """
        compute the entropy of the column created by merging col_idx and col_idx + 1 of the actual table
        :param col_idx:  integer representing the columns to be merged
        :return: float representing the entropy of the merged columns
        """
        return self.entropy_func(np.sum(self.table[:, col_idx:col_idx + 2], axis=1))

    def get_table_entropy(self):
        """
        Computes the entropy value of the whole table
        :return: entropy of the table
        """
        return np.sum(self.cols_entropy)

    def compute_delta(self, col_index, left=False):
        """
        Computes the deltas of the table merged at given index, and left or neighbor
        :param col_index: Index of colum that should be merged with its successor
        :param left: If True, the delta for the left neighbor is computed, If False the delta for the right one
        :return: delta of the possible merge
        """
        if left is True:
            col_index = col_index - 1

        # the new delta is the sum of the entropies before merging two columns minus the entropy of the new column
        # generated by merging them
        delta = 0
        delta += self.cols_entropy[col_index]
        delta += self.cols_entropy[col_index + 1]
        delta -= self.compute_merge_entropy(col_index)
        return delta

    def get_all_deltas(self):
        """
        Computes deltas of all possible merges of original table.
        :return: numpy array of all possible deltas obtained by merging columns
        """
        return  self.deltas

    def update(self, table, max_ind):
        """
        Updated the MergeValueComputer without reinitializing it
        :param table:  table with merged columns
        :param max_ind:  split index with maximum value
        """

        # replace old table with merged version of itself
        self.table = table

        # compute the new entropy of the merged columns and remove the second entropy of the old ones
        self.cols_entropy[max_ind] = self.entropy_func(self.table[:, max_ind].ravel())
        self.cols_entropy = np.delete(self.cols_entropy, max_ind + 1)

        # compute the new delta values of the deltas neighboring the removed one, but don't do it, if the removed one
        # was the first or the last one. Then only remove the possible neighbors.
        if max_ind != 0:
            self.deltas[max_ind - 1] = self.compute_delta(max_ind, left=True)
        if max_ind != len(self.deltas) -1:
            self.deltas[max_ind + 1] = self.compute_delta(max_ind)

        self.deltas = np.delete(self.deltas, max_ind)





