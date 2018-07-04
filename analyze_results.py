import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, brier_score_loss

from config import Settings
from datasets import load_test_data
from utils.persistence import (load_predictions, save_evaluation)


def full_scenario(data):
    """
        Filters data according to the first scenario
        (full test set). It is only a convenience
        function, does not actually do anything
        except returning the full dataset.

    Arguments:
        data {pandas DataFrame} -- DataFrame containing
                                   predictions for all the
                                   scored models.

    Returns:
        data -- DataFrame containing predictions for
                all the scored models.
    """

    return data


def elbwi_scenario(data):
    """
        Filters data according to the second
        scenario (ELBWI).

    Arguments:
        data {pandas DataFrame} -- DataFrame containing
                                   predictions for all the
                                   scored models.

    Returns:
        filtered_rows -- Filtered predictions DataFrame.
    """
    filtered_rows = data[(data.num_gaweeks < 26) &
                         (data.num_bwgt <= 999) &
                         (data.num_bwgt >= 400)]
    return filtered_rows


def vlbwi_scenario(data):
    """
        Filters data according to the third
        scenario (VLBWI).

    Arguments:
        data {pandas DataFrame} -- DataFrame containing
                                   predictions for all the
                                   scored models.

    Returns:
        filtered_rows -- Filtered predictions DataFrame.
    """

    filtered_rows = data[(data.num_bwgt <= 1500) &
                         (data.num_bwgt >= 1000)]
    return filtered_rows


def singletons_scenario(data):
    """
        Filters data according to the fourth
        scenario (SINGLETONS).

    Arguments:
        data {pandas DataFrame} -- DataFrame containing
                                   predictions for all the
                                   scored models.

    Returns:
        filtered_rows -- Filtered predictions DataFrame.
    """

    filtered_rows = data[(data.cat_mult == 0) &
                         (data.num_gaweeks <= 32) &
                         (data.num_gaweeks >= 23)]
    return filtered_rows


SCENARIOS = {
    "FULL": full_scenario,
    "ELBWI": elbwi_scenario,
    "VLBWI": vlbwi_scenario,
    "SINGLETONS": singletons_scenario
}


def compute_score(predictions_table, metric):
    """
        Computes a given score for each model
        examined in the study.

    Arguments:
        predictions_table {pandas DataFrame} -- DataFrame containing
                                                all the models predictions.
        metric {func}                        -- Scoring function. Can be
                                                either `roc_auc_score`,
                                                for ROC AUC, or
                                                `brier_score_loss`,
                                                for Brier loss.

    Returns:
        table -- A table of scores for each model considered.
    """

    all_models = Settings.BASELINES + ["lr", "nn"]

    table = pd.DataFrame(columns=all_models,
                         dtype=float)

    # Test data is needed because we must filter
    # rows based on values of particular features
    # (e.g. for VLBWI, we need test rows where
    # birth weight is above 1000 but below 1500).
    X_test, _ = load_test_data("ours")
    X_test = X_test.reset_index()

    for name, filter_scenario in SCENARIOS.items():
        scenario_scores = []
        rows = filter_scenario(X_test).index
        for model_name in all_models:
            y_true = predictions_table.loc[rows, "observed"]
            y_pred = predictions_table.loc[rows, model_name]
            score = metric(y_true, y_pred)
            scenario_scores.append(score)
        table.loc[name, :] = np.array(scenario_scores)

    return table


def compute_scores():
    """
        Computes both ROC AUC and Brier loss scores for
        each model considered in the study, and saves
        the corresponding results in pickle binary format.
    """

    prediction_table = load_predictions("all")

    # ROC AUC scores
    roc_aucs = compute_score(prediction_table, roc_auc_score)
    roc_aucs = roc_aucs.round(4)
    save_evaluation(roc_aucs, "roc_auc")

    # Brier loss scores
    brier_losses = compute_score(prediction_table, brier_score_loss)
    brier_losses = brier_losses.round(4)
    save_evaluation(brier_losses, "brier_loss")


if __name__ == '__main__':
    compute_scores()
