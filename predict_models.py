import numpy as np
import pandas as pd
from scipy.special import expit

from config import Settings
from datasets import load_test_data
from utils.persistence import load_model, save_predictions


def _logistic(X, W, b):
    """Computes a logistic regression prediction.

    Arguments:
        X {numpy matrix} -- Matrix of size [num_samples, num_features].
        W {numpy array}  -- Array of coefficients of size [num_features].
        b {float}        -- Intercept (or bias).

    Returns:
        predictions -- A vector of [num_samples] containing
                       the probabilistic predictions.
    """

    z = np.dot(X, W) + b
    predictions = expit(z)

    return predictions


def _predict_model(model_name, X_test):
    """
        Loads a model and computes its predictions
        on a test set.

    Arguments:
        model_name {str}          -- An identifier for the model.
        X_test {pandas DataFrame} -- The test set.

    Returns:
        predictions -- A numpy array of predictions.
    """

    model = load_model(model_name)
    predictions = model.predict_proba(X_test)[:, 1]

    return predictions


def compute_predictions():
    all_models = Settings.BASELINES + ["lr", "nn"]

    predictions_table = pd.DataFrame(columns=all_models,
                                     dtype=float)

    for model_name in all_models:

        if model_name in Settings.CANDIDATE_MODELS:
            dataset_name = "ours"
        else:
            dataset_name = model_name

        X_test, y_test = load_test_data(name=dataset_name)

        if model_name == "mankt":
            intercept = -10.28
            coefs = np.array([0.46, -0.47, 0.45, -0.28])
            predictions = 1 - _logistic(X_test.values, coefs, intercept)
        elif model_name == "tyson":
            intercept = 3.39
            coefs = np.array([-0.005113, 1.574100, 0.959800,
                              0.472100, -0.445400, -0.258200,
                              -0.606200])
            predictions = _logistic(X_test.values, coefs, intercept)
        else:
            predictions = _predict_model(model_name, X_test)

        predictions_table[model_name] = predictions

    predictions_table["observed"] = y_test.values
    save_predictions(predictions_table, "all")

    return predictions_table


if __name__ == '__main__':
    compute_predictions()
