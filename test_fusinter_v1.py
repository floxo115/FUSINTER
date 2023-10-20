import numpy as np
import pytest

import datasets
from fusinter_v1 import FUSINTERDiscretizer


@pytest.fixture
def init_split_fixture():
    """
    :return: a list of tuples containing (values, labels, expected splits, expected labels)
    """
    return [
        (
            datasets.paper_dataset_x,
            datasets.paper_dataset_y,
            np.array([2, 3, 13, 14, 15, 16, 17, 18, 19, 20, 23, 37, 38, 39, 40], dtype=np.float64),
            np.array([0, 1, 0, -1, 0, 1, 0, 1, -1, 1, 0, 1, -1, -1, 1, 0], dtype=np.int32)
        ),
        (
            np.array([-10, -10, -10, -9, -9, -8, -8, -8, -8, 3, 3, 2, 2], dtype=np.float64),
            np.array([1, 1, 1, 1, 1, 1, 2, 2, 1, 3, 3, 3, 3], dtype=np.int32),
            np.array([-8., 2.], dtype=np.float64),
            np.array([1, -1, 3], dtype=np.int32)
        )
    ]


class TestFusinterV1:
    def test_init_with_correct_input(self):
        data_x = np.random.rand(50)
        data_y = np.random.randint(0, 5, 50)
        fusinter = FUSINTERDiscretizer(data_x, data_y)

        sort_idx = np.argsort(data_x)
        assert np.all(fusinter.data_x == data_x[sort_idx])
        assert np.all(fusinter.data_y == data_y[sort_idx])

    def test_init_with_arrays_of_diff_len_should_throw(self):
        data_x = np.random.rand(59)
        data_y = np.random.randint(0, 5, 50)
        with pytest.raises(ValueError):
            fusinter = FUSINTERDiscretizer(data_x, data_y)

    def test_init_with_data_y_being_not_of_int_dtype(self):
        data_x = np.random.rand(59)
        data_y = np.random.rand(59)
        with pytest.raises(ValueError):
            fusinter = FUSINTERDiscretizer(data_x, data_y)

    def test_get_initial_intervals(self, init_split_fixture):
        for (data_x, data_y, exp_splits, exp_labels) in init_split_fixture:
            fusinter = FUSINTERDiscretizer(data_x, data_y)
            act_splits, act_labels = fusinter.get_initial_intervals()
            assert np.all(act_splits == exp_splits)
            assert np.all(act_labels == exp_labels)
