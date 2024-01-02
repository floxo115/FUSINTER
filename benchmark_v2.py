import numpy as np
from datasets import paper_dataset_y, paper_dataset_x
from fusinter_v2 import FUSINTERDiscretizer
from timeit import timeit

def f(dataset_increase_factor = 10):
    data_x = np.copy(paper_dataset_x)
    data_y = np.copy(paper_dataset_y)
    data_x = data_x + 40 * np.arange(0,dataset_increase_factor).reshape(dataset_increase_factor,1)
    data_x = data_x.ravel()
    data_y = np.tile(data_y, dataset_increase_factor)
    fusinter = FUSINTERDiscretizer(data_x, data_y)
    fusinter.apply()

if __name__ == "__main__":
    print("benchmark v2")
    for i in range(1, 100):
        print(f"f took {round(timeit('f('+str(i)+')', setup='from benchmark_v2 import f', number=5) / 5,4)} seconds to run with multiplicator {i}")
