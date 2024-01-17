from timeit import timeit

from fusinter_v2_2 import FUSINTERDiscretizer
from datasets.paper_dataset import *

import pandas as pd

cov_df = pd.read_csv("datasets/covtype.data", header=None)
cov_y = cov_df.pop(cov_df.shape[1] - 1).to_numpy()
cov_x0 = cov_df.pop(0).to_numpy()


def f(data_x, data_y):
    fusinter = FUSINTERDiscretizer(data_x, data_y)
    fusinter.apply()


if __name__ == "__main__":
    print("benchmark v1")
    print("paper dataset", end=": ")
    setup = "from benchmark_v2_2 import f, paper_dataset_x, paper_dataset_y, cov_x0, cov_y"
    print(round(timeit("f(paper_dataset_x, paper_dataset_y)", setup=setup, number=3) / 3, 4))
    print("cov feature 0", end=": ")
    print(round(timeit("f(cov_x0, cov_y)", setup=setup, number=1) / 1, 4))