from timeit import timeit

from fusinter_v2_2_new_interface import FUSINTERDiscretizer
from datasets.paper_dataset import *

paper_dataset_x = paper_dataset_x.reshape(-1, 1)

import pandas as pd

cov_df = pd.read_csv("datasets/covtype.data", header=None)
cov_y = cov_df.pop(cov_df.shape[1] - 1).to_numpy()
cov_x = cov_df.to_numpy()[:, 0:10] # all noncatecorical features

wave_df = pd.read_csv("datasets/waveform.data", header=None)
wave_y = wave_df.pop(wave_df.shape[1] - 1).to_numpy()
wave_x = wave_df.to_numpy() # all noncatecorical features


def f(data_x, data_y):
    for col_idx in range(data_x.shape[1]):
        x = data_x[:, col_idx].ravel()
        fusinter = FUSINTERDiscretizer(0.975, 1.)
        fusinter.fit(x, data_y)


if __name__ == "__main__":
    setup = """
from benchmark_v2_2 import f, \
paper_dataset_x, paper_dataset_y, \
cov_x, cov_y, \
wave_x, wave_y
"""
    n_runs = 3
    print("benchmark v2.2")

    print("paper dataset", end=": ")
    print(round(timeit("f(paper_dataset_x, paper_dataset_y)", setup=setup, number=n_runs) / n_runs, 4), flush=True)

    print("covertype features 0 to 9", end=": ")
    print(round(timeit("f(cov_x, cov_y)", setup=setup, number=n_runs) / n_runs, 4), flush=True)

    print("wave all features", end=": ")
    print(round(timeit("f(wave_x, wave_y)", setup=setup, number=n_runs) / n_runs, 4), flush=True)

