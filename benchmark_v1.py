from timeit import timeit

from fusinter_v1 import FUSINTERDiscretizer


def f(data_x, data_y):
    fusinter = FUSINTERDiscretizer(data_x, data_y)
    fusinter.apply()


if __name__ == "__main__":
    print("benchmark v1")
    print("paper dataset", end=": ")
    setup = "from benchmark_v1 import f\nfrom datasets import paper_dataset_y, paper_dataset_x"
    print(round(timeit("f(paper_dataset_x, paper_dataset_y)", setup=setup, number=3) / 3, 4))
