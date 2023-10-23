import datasets
from matplotlib import pyplot as plt
from datasets import  paper_raw_dataset, paper_dataset_y, paper_dataset_x

print(paper_raw_dataset)

fig, ax = datasets.get_plot_for_paper_data(datasets.paper_dataset_x, datasets.paper_dataset_y, title="data from paper")


from fusinter_v1 import FUSINTERDiscretizer

fusinter = FUSINTERDiscretizer(paper_dataset_x, paper_dataset_y)

split_lines, split_labels = fusinter.get_initial_intervals()
ax = datasets.add_split_lines_to_plot(ax, split_lines, split_labels, x_offset=-0.5)
ax.legend()
fig.show()

print(fusinter.create_table(split_lines))
