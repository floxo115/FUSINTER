import numpy as np
from timeit import timeit

from datasets import paper_dataset_y, paper_dataset_x
from fusinter_v1 import FUSINTERDiscretizer


def f(dataset_increase_factor = 10):
    data_x = np.copy(paper_dataset_x)
    data_y = np.copy(paper_dataset_y)
    data_x = data_x + 40 * np.arange(0,dataset_increase_factor).reshape(dataset_increase_factor,1)
    data_x = data_x.ravel()
    data_y = np.tile(data_y, dataset_increase_factor)
    fusinter = FUSINTERDiscretizer(data_x, data_y)
    fusinter.apply()


if __name__ == "__main__":
    print("benchmark v1")
    print(f"f took {timeit('f()', setup='from benchmark_v1 import f', number=10 ** 1) / 10 ** 1} seconds to run")
