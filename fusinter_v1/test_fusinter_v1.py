import numpy as np
import pytest

import datasets
from .fusinter_v1 import FUSINTERDiscretizer, shannon_entropy


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
            np.array([-10, -10, -10, -9, -9, -8, -8, -8, -8, 3, 3, 2, 2], dtype=np.float64),
            np.array([1, 1, 1, 1, 1, 1, 2, 2, 1, 3, 3, 3, 3], dtype=np.int32),
        )]






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





def test_shannon_entropy():
    input_table = np.array([2,1,0,0,2,4,0,3,0], dtype=int).reshape(3,3)
    assert np.isclose(shannon_entropy(input_table, alpha=0.5, lam=0.2), 0.7954323401173174)