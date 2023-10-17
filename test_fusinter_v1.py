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
            (2, 3, 13, 14, 15, 16, 17, 18, 19, 20, 22, 27, 28, 29, 30),
            (0, 1, 0, -1, 0, 1, 0, 1, -1, 1, 0, 1, -1, -1, 1, 0)
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
            assert act_splits == exp_splits
            assert act_labels == exp_labels
