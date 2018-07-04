import os
from matplotlib import pyplot as plt
import seaborn as sns

from config import Dirs
from datasets import load_test_data
from utils.persistence import load

sns.set_style("whitegrid")
sns.set_context("paper")


def plot_results():
    """
        Utility function to realize Figure 1 of the paper.
    """

    fontsize = 12
    ordered_columns = ["observed", "nn", "lr"]
    test_data, _ = load_test_data("ours")

    path = os.path.join(Dirs.PREDS_DIR, "all_predictions.pkl")
    predictions_table = load(path)

    data = predictions_table[ordered_columns]
    data["num_gaweeks"] = test_data.num_gaweeks

    data = data.groupby("num_gaweeks").sum()[ordered_columns]
    data.columns = ["OBSERVED", "PREDICTED PISA", "PREDICTED Logistic2"]
    data.index.name = "GESTATIONAL AGE"

    colors = ["#deebe1", "#9ecae1", "#3182bd"]
    ax = data.plot(kind="bar",
                   rot=0,
                   fontsize=fontsize,
                   use_index=False,
                   color=colors,
                   edgecolor=["k"] * 48)

    ax.set_ylabel("MORTALITY", fontsize=fontsize)
    ax.set_xticklabels(range(22, 40))

    ax.set_xlabel("GESTATIONAL AGE (Weeks)", fontsize=fontsize)
    ax.grid(False)

    plt.legend(frameon=True, fontsize=fontsize)

    path = os.path.join(Dirs.FIGURES_DIR, "figure.png")
    plt.savefig(path, dpi=600)


if __name__ == '__main__':
    plot_results()
