import numpy as np
import pytest

from .fusinter_v1 import FUSINTERDiscretizer, shannon_entropy


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


@pytest.mark.parametrize("input_table, alpha, lam, expected", [
    (
            np.array([2, 1, 0, 0, 2, 4, 0, 3, 0], dtype=int).reshape(3, 3),
            0.5, 0.2,
            0.7954323401173173,
    ),
    (
            np.array([8, 0, 2, 9, 1, 2], dtype=int).reshape(2, 3),
            0.6, 0.9,
            1.5388406493870612,
    )
])
def test_shannon_entropy(input_table, alpha, lam, expected):
    assert np.isclose(shannon_entropy(input_table, alpha=alpha, lam=lam), expected)
