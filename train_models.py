import numpy as np
import pandas as pd

from searchgrid import make_grid_search
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.testing import ignore_warnings

from config import Settings
from datasets import load_train_data
from utils.pipelines import get_pipeline, get_baseline_pipeline
from utils.persistence import (save_results, save_model)
from utils.parsers import get_model_selection_parser


def calculate_scores(all_scores, mode):
    """Processes raw scores from the model selection procedure
       and puts them into a pandas DataFrame.

    Arguments:
        all_scores {dict} -- a dictionary containing train and test scores
                             for each models that was cross-validated.
        mode {str}        -- flag that helps determine if scores belong
                             to candidate models or baselines.

    Returns:
        table    -- a pandas DataFrame with the scores nicely formatted.
    """

    if mode == "candidate_models":
        all_models = Settings.CANDIDATE_MODELS
    elif mode == "baselines":
        all_models = Settings.TRAINABLE_BASELINES

    table = pd.DataFrame(index=all_models,
                         columns=["train_mean", "train_std",
                                  "test_mean", "test_std"],
                         dtype=float)

    for model_name in all_models:
        score = all_scores[model_name]
        best = np.argmin(score["rank_test_score"])
        table.loc[model_name, "train_mean"] = score["mean_train_score"][best]
        table.loc[model_name, "train_std"] = score["std_train_score"][best]
        table.loc[model_name, "test_mean"] = score["mean_test_score"][best]
        table.loc[model_name, "test_std"] = score["std_test_score"][best]
    table = table.round(4)

    return table


@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=UserWarning)
@ignore_warnings(category=ResourceWarning)
def run_model_selection(test):
    """
        Runs 5-CV grid search on each of the candidate models.

    Arguments:
        test {bool}    -- Whether to use simplified grids (
                          (for testing purposes only).
    """

    mode = "test" if test else "model_selection"
    X, y = load_train_data(name="ours")

    model_scores, model_params = {}, {}
    for model_name in Settings.CANDIDATE_MODELS:
        pipe = get_pipeline(model_name, mode=mode)
        cv = StratifiedKFold(
            n_splits=Settings.MODEL_SELECTION_SPLIT,
            random_state=Settings.RANDOM_SEED)

        grid = make_grid_search(pipe,
                                cv=cv,
                                verbose=1,
                                scoring="roc_auc",
                                n_jobs=Settings.N_CPUS,
                                return_train_score=True)

        grid.fit(X, y)

        model_scores[model_name] = grid.cv_results_
        model_params[model_name] = grid.best_params_

        if model_name in ["lr", "nn"]:
            best_model = grid.best_estimator_
            save_model(best_model, grid.cv_results_, model_name)

    model_scores = calculate_scores(model_scores, mode="candidate_models")
    save_results(model_scores, "models_scores")
    save_results(model_params, "models_best_parameters")


@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=UserWarning)
@ignore_warnings(category=ResourceWarning)
def run_baselines_model_selection(test):
    """
        Runs 5-CV grid search on each trainable baseline
        (bw, bwga and logreg).

    Arguments:
        test {bool}    -- Whether to use simplified grids (
                          (for testing purposes only).
    """

    mode = "test" if test else "model_selection"

    baseline_scores, baseline_params = {}, {}
    for baseline in Settings.TRAINABLE_BASELINES:
        X, y = load_train_data(name=baseline)

        pipe = get_baseline_pipeline(baseline, mode=mode)

        cv = StratifiedKFold(
            n_splits=Settings.MODEL_SELECTION_SPLIT,
            random_state=Settings.RANDOM_SEED)

        grid = make_grid_search(pipe,
                                cv=cv,
                                verbose=1,
                                scoring="roc_auc",
                                n_jobs=Settings.N_CPUS,
                                return_train_score=True)

        grid.fit(X, y)

        baseline_scores[baseline] = grid.cv_results_
        baseline_params[baseline] = grid.best_params_

        best_model = grid.best_estimator_
        save_model(best_model, grid.cv_results_, baseline)

    baseline_scores = calculate_scores(baseline_scores, mode="baselines")
    save_results(baseline_scores, "baselines_scores")
    save_results(baseline_params, "baselines_best_parameters")


if __name__ == '__main__':
    parser = get_model_selection_parser()
    args = parser.parse_args()
    run_model_selection(args.test)
    run_baselines_model_selection(args.test)
