import datasets
from datasets import paper_dataset_y, paper_dataset_x
from fusinter_v2 import FUSINTERDiscretizer
from timeit import timeit

def f():
    fusinter = FUSINTERDiscretizer(paper_dataset_x, paper_dataset_y)
    fusinter.apply()

print(f"f took {timeit('f()', setup='from benchmark_v2 import f', number=10**3)/10**3} seconds to run")
