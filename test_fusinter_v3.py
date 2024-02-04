import numpy as np

import pandas as pd

from datasets.paper_dataset import paper_dataset_x, paper_dataset_y
from fusinter_v2_2 import FUSINTERDiscretizer as FUSINTERDiscretizer_v1
from FUSINTER_v3_pybind import FUSINTERDiscretizer


class TestFusinterV2:
    def test_if_apply_give_same_result_as_v1(self):
        for alpha in np.linspace(0.01, 1, 10):
            for lam in np.linspace(0.01, 2, 20):
                v1 = FUSINTERDiscretizer_v1(paper_dataset_x, paper_dataset_y)
                v1_splits = v1.apply(alpha, lam)
                v2 = FUSINTERDiscretizer(alpha, lam)
                v2_splits = v2.fit(paper_dataset_x, paper_dataset_y)


                if len(v1_splits) == 0:
                    assert len(v2_splits) == 0
                else:
                    assert np.all(v1_splits == v2_splits)

                #print(np.round(v1_splits,5))
                #print(np.round(v2_splits,5))
                #print("----------------------------------------")

    def test_if_apply_give_same_result_as_v1_with_big_ds(self):
                
                cov_df = pd.read_csv("datasets/covtype.data", header=None)
                cov_y = cov_df.pop(cov_df.shape[1] - 1).to_numpy()
                cov_x = cov_df.to_numpy()[:, 0:1].ravel() # all noncatecorical features

                v1 = FUSINTERDiscretizer_v1(cov_x, cov_y)
                v1_splits = v1.apply(0.975, 1.)
                v2 = FUSINTERDiscretizer(0.975, 1.)
                v2_splits = v2.fit(cov_x, cov_y)

                print(np.round(v1_splits,5))
                print(np.round(v2_splits,5))
                print("----------------------------------------")


                if len(v1_splits) == 0:
                    assert len(v2_splits) == 0
                else:
                    assert np.all(v1_splits == v2_splits)

                print(np.round(v1_splits,5))
                print(np.round(v2_splits,5))
                print("----------------------------------------")
