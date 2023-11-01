import numpy as np
import pytest

import datasets
from .splitter import Splitter


@pytest.fixture()
def init_data():
    """
    :return: a list of initial values and labels for initializing FUSINTERDiscretizer
    """
    return [
        (
            datasets.paper_dataset_x,
            datasets.paper_dataset_y,
        ),
        (
            np.array([-10, -10, -10, -9, -9, -8, -8, -8, -8, 2, 2, 3, 3], dtype=np.float64),
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2], dtype=np.int32),
        )]


@pytest.fixture()
def init_split_fixture():
    """
    :return: a list of tuples containing ( expected splits, expected labels)
    """
    return [
        (
            np.array([2, 3, 13, 14, 15, 16, 17, 18, 19, 20, 23, 37, 38, 39, 40], dtype=np.float64),
            np.array([0, 1, 0, -1, 0, 1, 0, 1, -1, 1, 0, 1, -1, -1, 1, 0], dtype=np.int32),
        ),
        (
            np.array([-8., 2.], dtype=np.float64),
            np.array([0, -1, 2], dtype=np.int32),
        )
    ]


class TestSplitter:
    def test_get_initial_intervals(self, init_data, init_split_fixture):
        for (data_x, data_y), (exp_splits, exp_labels) in zip(init_data, init_split_fixture):
            splitter = Splitter(data_x, data_y)
            act_splits, act_labels = splitter.apply()
            assert np.all(act_splits == exp_splits)
            assert np.all(act_labels == exp_labels)
