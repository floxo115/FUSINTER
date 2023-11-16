import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import datasets
from datasets import paper_dataset_y, paper_dataset_x
from fusinter_v1 import FUSINTERDiscretizer

# Comparisons to the paper outputs
# 1. Plot to compare the initial splits of the FUSINTER implementation to the ones given in the paper on page 341

fig, ax = datasets.get_plot_for_paper_data(datasets.paper_dataset_x, datasets.paper_dataset_y, title="data from paper")
fusinter = FUSINTERDiscretizer(paper_dataset_x, paper_dataset_y)
splits, splits_labels = fusinter.get_initial_intervals()
ax = datasets.add_split_lines_to_plot(ax, splits, splits_labels[:-1], x_offset=-0.3)

fig.show()

# Plot final splits from fusinter for paper dataset
final_splits = fusinter.apply()
fig, ax = datasets.get_plot_for_paper_data(datasets.paper_dataset_x, datasets.paper_dataset_y, title="data from paper")
ax = datasets.add_split_lines_to_plot(ax, final_splits, np.zeros(len(final_splits)), x_offset=-0.3)
fig.show()

# 2. Plot the same for the Iris dataset (petal length feature)
from sklearn.datasets import load_iris

iris_ds = load_iris()
data_x = iris_ds["data"][:, 2]
data_y = iris_ds["target"]

fig, ax = plt.subplots(1, 1)
for i in np.unique(data_y):
    sns.kdeplot(data_x[data_y == i], ax=ax, label=f"class {i}")

fusinter = FUSINTERDiscretizer(data_x, data_y)
splits, splits_labels = fusinter.get_initial_intervals()
ax = datasets.add_split_lines_to_plot(ax, splits, splits_labels[:-1])
plt.title("Iris Dataset (Petal Length) Initial Splits")
plt.show()

fig, ax = plt.subplots(1, 1)
for i in np.unique(data_y):
    sns.kdeplot(data_x[data_y == i], ax=ax, label=f"class {i}")

fusinter = FUSINTERDiscretizer(data_x, data_y)
final_splits = fusinter.apply(alpha=0.93, lam=0.94)
print(final_splits)

ax = datasets.add_split_lines_to_plot(ax, final_splits, np.zeros(len(final_splits)))
fig.legend()
plt.title("Iris Dataset (Petal Length) Final Splits")
plt.show()
