import numpy as np
import pytest

from .merge_value_computer import MergeValueComputer
from fusinter_v1.fusinter_v1 import shannon_entropy


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
def test_merge_value_computer_get_table_entropy(input_table, alpha, lam, expected):
    mvc = MergeValueComputer(input_table, alpha, lam)
    table_entropy = mvc.get_table_entropy()

    assert np.isclose(table_entropy, expected)


@pytest.mark.parametrize("original_table, merged_table, alpha, lam, merge_index", [
    (
            np.array([2, 1, 0, 0, 2, 4, 0, 3, 0], dtype=int).reshape(3, 3),
            np.array([3, 0, 2, 4, 3, 0], dtype=int).reshape(3, 2),
            0.5, 0.2, 0,
    ),
    (
            np.array([2, 1, 0, 0, 2, 4, 0, 3, 0], dtype=int).reshape(3, 3),
            np.array([2, 1, 0, 6, 0, 3], dtype=int).reshape(3, 2),
            0.5, 0.2, 1,
    ),
])
def test_get_merged_table_entropy(original_table, merged_table,  alpha, lam, merge_index):
    expected = shannon_entropy(merged_table, alpha, lam)
    mvc = MergeValueComputer(original_table, alpha, lam)

    assert np.isclose(mvc.get_table_entropy() - mvc.compute_delta(merge_index), expected)
