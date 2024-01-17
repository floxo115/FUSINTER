import numpy as np
import pytest

from datasets.paper_dataset import paper_dataset_x, paper_dataset_y
from fusinter_v1 import FUSINTERDiscretizer as FUSINTERDiscretizer_v1
from .fusinter_v2 import FUSINTERDiscretizer


class TestFusinterV2:
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

    def test_if_apply_give_same_result_as_v1(self):
        for alpha in np.linspace(0.01, 1, 10):
            for lam in np.linspace(0.01, 2, 20):
                v1 = FUSINTERDiscretizer_v1(paper_dataset_x, paper_dataset_y)
                v1_splits = v1.apply(alpha, lam)
                v2 = FUSINTERDiscretizer(paper_dataset_x, paper_dataset_y)
                v2_splits = v2.apply(alpha, lam)


                if len(v1_splits) == 0:
                    assert len(v2_splits) == 0
                else:
                    assert np.all(v1_splits == v2_splits)
