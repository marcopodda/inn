import numpy as np

from sklearn.linear_model import LogisticRegression as LRC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLP
from xgboost import XGBClassifier as GBC

from config import Settings


def _integer_grid(low, high, size):
    """Utility to create a grid of integers"""

    return np.linspace(low, high, size, dtype=np.int)


def _float_grid(low, high, size, log=False):
    """Utility to create a grid of floats"""
    if log:
        return np.logspace(low, high, size)
    else:
        return np.linspace(low, high, size)


def _tuple_grid(low, high, size):
    """Utility to create a grid of tuples"""

    return [tuple([n]) for n in _integer_grid(low, high, size)]


"""
   This dictionary contains the base settings
   for the candidate models that are used in
   the double cross-validation procedure. On
   each of these models, grid search will find
   its best hyperparameters.
"""


MODELS = {
    "lr": LRC(solver="saga",
              penalty="l2",
              warm_start=True,
              n_jobs=Settings.N_CPUS,
              random_state=Settings.RANDOM_SEED),
    "knn": KNC(metric="euclidean",
               n_jobs=Settings.N_CPUS,
               algorithm="ball_tree",
               weights="uniform",
               p=1),
    "rf": RFC(warm_start=True,
              n_jobs=Settings.N_CPUS,
              criterion="entropy",
              bootstrap=True,
              max_features="log2",
              random_state=Settings.RANDOM_SEED),
    "xgb": GBC(silent=True,
               subsample=0.15,
               n_jobs=1,
               objective="binary:logistic",
               base_score=0.5,
               random_state=Settings.RANDOM_SEED),
    "svm": SVC(kernel="rbf",
               class_weight="balanced",
               probability=True,
               cache_size=10000,
               random_state=Settings.RANDOM_SEED),
    "nn": MLP(warm_start=True,
              activation="tanh",
              random_state=Settings.RANDOM_SEED)
}


"""
   Grids of hyperparameters used in model selection.
   Sensible hyperparameter intervals were determined
   by multiple grid searches, and the width of such
   intervals was refined (shrunk) incrementally.
   These additional trials are not reported. For
   completeness, we report the very first grid
   that was searched:

   MODELSELECTION_INITIAL_GRIDS = {
        "lr": {
            "C": _float_grid(-5, 5, 250, log=True),
        },
        "knn": {
            "n_neighbors": _integer_grid(5, 2000, 12),
            "leaf_size": _integer_grid(5, 2000, 15)
        },
        "rf": {
            "n_estimators": _integer_grid(50, 3000, 12),
            "max_depth": _integer_grid(2, 15, 14)
        },
        "xgb": {
            "n_estimators": _integer_grid(50, 3000, 12),
            "max_depth": _integer_grid(2, 14, 5),
            "learning_rate": _float_grid(-4, -1, 4, log=True),
            "colsample_bytree": _float_grid(0.01, 0.99, 5)
        },
        "svm": {
            "C": _float_grid(-4, 4, 18, log=True),
            "gamma": _float_grid(-4, 4, 18, log=True)
        },
        "nn": {
            "hidden_layer_sizes": _tuple_grid(10, 250, 9),
            "alpha": _float_grid(-4, -1, 4, log=True),
            "learning_rate_init": _float_grid(-5, -2, 4, log=True),
            "max_iter": _integer_grid(10, 190, 10)
        }
    }

    Below is a table illustrating some details of the final grid that
    was used to do the final optimization.
    Table legend:
    - grid dim:      dimension of the grid (# of hyper-parameters optimized)
    - tested conf:   how many hyperparameter configurations were tested
    - fitted models: since the number of outer folds of the CV procedure
                     was 5, a total of 5 models were fitted on different
                     folds for each hyperparameter configuration.
                     Hence, fitted models = tested confs x 5

                             grid dim   tested confs   fitted models
    Logistic Regression:            1             60             300
    K-Nearest Neighbor:             2             36             180
    Random Forest:                  2             48             240
    Gradient Boosting:              4            216            1080
    Support Vector Machine:         2             54             270
    Neural Network:                 4            375            1875
"""

MODELSELECTION_GRIDS = {
    "lr": {
        "C": _float_grid(0.02, 0.03, 60),
    },
    "knn": {
        "n_neighbors": _integer_grid(350, 500, 6),
        "leaf_size": _integer_grid(500, 1000, 6)
    },
    "rf": {
        "n_estimators": _integer_grid(600, 1600, 6),
        "max_depth": _integer_grid(5, 14, 8)
    },
    "xgb": {
        "n_estimators": _integer_grid(1100, 1300, 3),
        "max_depth": _integer_grid(3, 5, 3),
        "learning_rate": _float_grid(0.005, 0.008, 4),
        "colsample_bytree": _float_grid(0.8, 0.9, 6)
    },
    "svm": {
        "C": _float_grid(900, 1400, 6),
        "gamma": _float_grid(0.0001, 0.003, 9)
    },
    "nn": {
        "hidden_layer_sizes": _tuple_grid(40, 50, 3),
        "alpha": _float_grid(0.001, 0.003, 5),
        "learning_rate_init": _float_grid(0.001, 0.003, 5),
        "max_iter": _integer_grid(90, 110, 5)
    }
}


"""
   Grids of hyperparameters used to tune baseline models.
"""

BASELINE_GRIDS = {
    "lr": {
        "C": _float_grid(-5, 5, 150, log=True),
    }
}


"""
   This grid is used for testing purposes only.
"""

TEST_GRIDS = {
    "lr": {
        "C": _float_grid(-2, -1, 2, log=True)
    },
    "knn": {
        "n_neighbors": [300],
    },
    "rf": {
        "n_estimators": _integer_grid(10, 11, 2),
    },
    "xgb": {
        "n_estimators": _integer_grid(10, 11, 2),
    },
    "svm": {
        "C": [0.1],
    },
    "nn": {
        "max_iter": _integer_grid(5, 6, 2)
    }
}
