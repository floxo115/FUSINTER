import datasets
from matplotlib import pyplot as plt
from datasets import  paper_raw_dataset, paper_dataset_y, paper_dataset_x

print(paper_raw_dataset)

fig, ax = datasets.get_plot_for_paper_data(datasets.paper_dataset_x, datasets.paper_dataset_y, title="data from paper")

plt.show()

from fusinter_v1 import FUSINTERDiscretizer

fusinter = FUSINTERDiscretizer(paper_dataset_x, paper_dataset_y)

print(fusinter.get_initial_intervals())