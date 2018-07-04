from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from searchgrid import set_grid

from .grids import (MODELS, MODELSELECTION_GRIDS,
                    TEST_GRIDS, BASELINE_GRIDS)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
        Implements the scikit-learn Trasformers API to
        construct a Transformer which basically selects
        a subset of columns in the input DataFrame,
        according to some given prefix.
    """

    def __init__(self, prefix):
        self.prefix = prefix

    def fit(self, X, y=None, **params):
        return self

    def transform(self, X, **params):
        cols = [c for c in X.columns if c.startswith(self.prefix)]
        return X[cols]


def get_pipeline(model_name, mode):
    """
        Constructs a Pipeline object that contains
        the logical steps of preprocessing and training.
        The hyperparameters specified in the object are
        then optimized using grid search.

    Arguments:
        model_name {str} -- Name of the model.
        mode {str}       -- Construction mode of the grid.
                            There are three possible choices:
                            - `model_selection`: uses a grid
                              specific for choosing the best
                              model.
                            - `test`: uses a simplified grid
                              for testing purposes only.

    Returns:
        pipe -- The constructed Pipeline object.
    """

    # select the appropriate grid based on the mode
    if mode == "model_selection":
        grid = MODELSELECTION_GRIDS[model_name]
    elif mode == "test":
        grid = TEST_GRIDS[model_name]

    model_grid = set_grid(MODELS[model_name], **grid)
    pipe = Pipeline([
        ('preprocessing', FeatureUnion([
            ('num', Pipeline([
                ('cs', ColumnSelector("num")),
                ('ss', StandardScaler())])),
            ('cat', Pipeline([
                ('cs', ColumnSelector("cat"))])),
            ("oh", Pipeline([
                ("cs", ColumnSelector("oh")),
                ("oe", OneHotEncoder(sparse=False))]))])),
        ('clf', model_grid)])

    return pipe


def get_baseline_pipeline(model_name, mode):
    """
        Similar to get_pipeline, but for baseline models.
    """

    if mode == "model_selection":
        grid = BASELINE_GRIDS["lr"]
    elif mode == "test":
        grid = TEST_GRIDS["lr"]

    model_grid = set_grid(MODELS["lr"], **grid)

    if model_name in ["bw", "bwga"]:
        pipe = Pipeline([
            ('ss', StandardScaler()),
            ('clf', model_grid)])

    elif model_name == "logreg":
        pipe = Pipeline([
            ('preprocessing', FeatureUnion([
                ('num', Pipeline([
                    ('cs', ColumnSelector("num")),
                    ('ss', StandardScaler())])),
                ('cat', Pipeline([
                    ('cs', ColumnSelector("cat"))]))])),
            ('clf', model_grid)])

    return pipe
