import numpy as np

from datasets.paper_dataset import paper_dataset_y, paper_dataset_x

# # Comparisons to the paper outputs
# # 1. Plot to compare the initial splits of the FUSINTER implementation to the ones given in the paper on page 341
#
# fig, ax = datasets.get_plot_for_paper_data(datasets.paper_dataset_x, datasets.paper_dataset_y, title="data from paper")
# fusinter = FUSINTERDiscretizer(paper_dataset_x, paper_dataset_y)
# splits, splits_labels = fusinter.get_initial_intervals()
# ax = datasets.add_split_lines_to_plot(ax, splits, splits_labels[:-1], x_offset=-0.3)
#
# fig.show()
#
# # Plot final splits from fusinter for paper dataset
# final_splits = fusinter.apply()
# fig, ax = datasets.get_plot_for_paper_data(datasets.paper_dataset_x, datasets.paper_dataset_y, title="data from paper")
# ax = datasets.add_split_lines_to_plot(ax, final_splits, np.zeros(len(final_splits)), x_offset=-0.3)
# fig.show()
#
# # 2. Plot the same for the Iris dataset (petal length feature)
# from sklearn.datasets import load_iris
#
# iris_ds = load_iris()
# data_x = iris_ds["data"][:, 2]
# data_y = iris_ds["target"]
#
# fig, ax = plt.subplots(1, 1)
# for i in np.unique(data_y):
#     sns.kdeplot(data_x[data_y == i], ax=ax, label=f"class {i}")
#
# fusinter = FUSINTERDiscretizer(data_x, data_y)
# splits, splits_labels = fusinter.get_initial_intervals()
# ax = datasets.add_split_lines_to_plot(ax, splits, splits_labels[:-1])
# plt.title("Iris Dataset (Petal Length) Initial Splits")
# plt.show()
#
# fig, ax = plt.subplots(1, 1)
# for i in np.unique(data_y):
#     sns.kdeplot(data_x[data_y == i], ax=ax, label=f"class {i}")
#
# fusinter = FUSINTERDiscretizer(data_x, data_y)
# final_splits = fusinter.apply(alpha=0.93, lam=0.94)
# print(final_splits)
#
# ax = datasets.add_split_lines_to_plot(ax, final_splits, np.zeros(len(final_splits)))
# fig.legend()
# plt.title("Iris Dataset (Petal Length) Final Splits")
# plt.show()


# alpha = 0.6
# lam = 0.4
# fusinter = FUSINTERDiscretizer(paper_dataset_x, paper_dataset_y)
# final_splits_v1 = fusinter.apply(alpha=alpha, lam=lam)
# fusinter = FUSINTERDiscretizer_v2(paper_dataset_x, paper_dataset_y)
# final_splits_v2 = fusinter.apply(alpha=alpha, lam=lam)

# from fusinter_v2.splitter import Splitter
# from fusinter_v2.table_manager import TableManager
#
# splitter = Splitter(paper_dataset_x, paper_dataset_y)
# input_splits, _ = splitter.apply()
#
# tm = TableManager(paper_dataset_x, paper_dataset_y)
# table = tm.create_table(input_splits)
# print(table)


# from fusinter_v2_2_new_interface import FUSINTERDiscretizer
#
# discretizer = FUSINTERDiscretizer(0.95, 1)
#
# discretizer.fit(paper_dataset_x, paper_dataset_y)
# print(discretizer.transform(paper_dataset_x))
#
# from fusinter_v2_2 import FUSINTERDiscretizer
#
# discretizer = FUSINTERDiscretizer(paper_dataset_x, paper_dataset_y)
# print(np.digitize(paper_dataset_x, discretizer.apply(0.95, 1)))

from fusinter_v2_2_new_interface import FUSINTERDiscretizer
import pandas as pd

discretizer = FUSINTERDiscretizer(0.95, 1)

cov_df = pd.read_csv("datasets/covtype.data", header=None)
cov_y = cov_df.pop(cov_df.shape[1] - 1).to_numpy()
cov_x0 = cov_df.pop(0).to_numpy()

discretizer.fit(cov_x0, cov_y)
print(discretizer.transform(sorted(cov_x0)))
