import os
import pandas as pd

from config import Dirs, Settings
from utils.parsers import get_dataset_parser
from utils.persistence import save

"""
   This dictionary contains names of the features
   for each dataset that will be used.
   ORDER OF FEATURES MUST NOT BE CHANGED.
"""

CONFIG = {
    "bw": {
        "index": ["id", "hospno", "byear"],
        "num": ["bwgt"],
        "cat": [],
        "oh": []
    },
    "bwga": {
        "index": ["id", "hospno", "byear"],
        "num": ["bwgt", "gaweeks"],
        "cat": [],
        "oh": []
    },
    "mankt": {
        "index": ["id", "hospno", "byear"],
        "num": ["gaweeks", "bwgt"],
        "cat": ["sex", "ga23"],
        "oh": []
    },
    "tyson": {
        "index": ["id", "hospno", "byear"],
        "num": ["bwgt"],
        "cat": ["ga22", "ga23", "ga24",
                "sex", "mult", "aster"],
        "oh": []
    },
    "logreg": {
        "index": ["id", "hospno", "byear"],
        "num": ["gaweeks", "gasq", "ap1"],
        "cat": ["sex", "locate", "mult",
                "sga10", "vagdel"],
        "oh": []
    },
    "ours": {
        "index": ["id", "hospno", "byear"],
        "num": ["bwgt", "gaweeks", "ap1", "ap5"],
        "cat": ["sex", "vagdel", "chorio", "pcare",
                "aster", "mhypertens", "mult"],
        "oh": ["race"]
    }
}


def _add_column_prefix(data, prefix):
    """Utility that appends a prefix to columns of a pandas DataFrame.

    Arguments:
        data {DataFrame} -- the DataFrame to process.
        prefix {str}     -- the predix to add to the DataFrame's columns.

    Returns:
        new_data -- pandas DataFrame with the columns modified accordingly.
    """

    columns = data.columns.tolist()
    new_names = ["{}_{}".format(prefix, c) for c in columns]
    mapping = dict(zip(columns, new_names))
    new_data = data.rename(columns=mapping)
    return new_data


def _load_raw(years):
    """Loads the dataset in .csv format.

    Arguments:
        years {str} -- years where the data was collected.
                      Possible choices are:
                      2008-2014 are years used for training
                      2015-2016 are years used for testing

    Returns:
        raw_data   -- The data as a pandas DataFrame
    """
    filename = "INN{}.csv".format(years)
    path = os.path.join(Dirs.RAW_DIR, filename)
    raw_data = pd.read_csv(path)
    return raw_data


def _create_dataset(name, years):
    """Preprocesses the dataset by appending to each
       column's name its type:
       - `num` for numeric data
       - `cat` for binary (0/1) data
       - `oh` for categorical data
       This information is useful when generating the
       modeling pipeline, as different data types
       are handled with different preprocessing
       techniques.
       Finally, saves the dataset in binary format.

    Arguments:
        name {[type]} -- [description]
        years {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    config = CONFIG[name]
    data = _load_raw(years)
    data = data.set_index(config["index"])

    if name == "mankt":
        data.gaweeks = data.gadays / 7 + data.gaweeks
        data.bwgt = 1 / (data.bwgt**2)

    num = data[config["num"]]
    cat = data[config["cat"]]
    oh = data[config["oh"]]
    target = data[["died"]]

    num = _add_column_prefix(num, "num")
    cat = _add_column_prefix(cat, "cat")
    oh = _add_column_prefix(oh, "oh")

    new_data = pd.concat([num, cat, oh, target], axis=1)
    path = os.path.join(Dirs.DATA_DIR, "{}_{}.pkl".format(name, years))

    if not os.path.exists(path):
        save(new_data, path)

    return new_data


def _load_dataset(name, years):
    """Loads a dataset in binary format.

    Arguments:
        name {str}   -- name of the dataset.
                        Possible choices are:
                        - `ours`: the dataset that was
                          used in our experiments
                        - `bw`: baseline with only
                          birth weight feature
                        - `bwga`: baseline with birth
                          weight and gestational age
                          as features
                        - `logreg`: baseline with features
                          similar to the VON-RA model
                        - `mankt`: baseline with features
                          from the paper by Manktelow et al.
                        - `tyson`: baseline with features
                          from the paper by Tyson et al.
        years {str}  -- years when data was collected
                        (see _load_raw docstring for details)

    Returns:
        features -- a pandas DataFrame containing the features.
        labels   -- a pandas Series containing the labels
    """

    if name in Settings.CANDIDATE_MODELS:
        name = "ours"

    path = os.path.join(Dirs.DATA_DIR, "{}_{}.pkl".format(name, years))
    try:
        data = pd.read_pickle(path)
    except FileNotFoundError:
        data = _create_dataset(name, years)

    feature_cols = [c for c in data.columns if c != Settings.TARGET]
    features = data[feature_cols]
    labels = data[Settings.TARGET]
    return features, labels


def load_train_data(name):
    """Helper function to load a training set.

    Keyword Arguments:
        name {name} -- name of the dataset

    Returns:
        features -- a pandas DataFrame containing the features.
        labels   -- a pandas Series containing the labels
    """
    features, labels = _load_dataset(name, years=Settings.TRAIN_YEARS)
    return features, labels


def load_test_data(name):
    """Helper function to load a test set.

    Keyword Arguments:
        name {name} -- name of the dataset

    Returns:
        features -- a pandas DataFrame containing the features.
        labels   -- a pandas Series containing the labels
    """
    features, labels = _load_dataset(name, years=Settings.TEST_YEARS)
    return features, labels


if __name__ == '__main__':
    parser = get_dataset_parser()
    args = parser.parse_args()

    if args.name == "all":
        # create train and test data for our models
        _create_dataset("ours", Settings.TRAIN_YEARS)
        _create_dataset("ours", Settings.TEST_YEARS)

        # create train and test data for baselines
        _create_dataset("bw", Settings.TRAIN_YEARS)
        _create_dataset("bw", Settings.TEST_YEARS)

        _create_dataset("bwga", Settings.TRAIN_YEARS)
        _create_dataset("bwga", Settings.TEST_YEARS)

        _create_dataset("logreg", Settings.TRAIN_YEARS)
        _create_dataset("logreg", Settings.TEST_YEARS)

        # models mankt and tyson don't need train sets
        _create_dataset("mankt", Settings.TEST_YEARS)
        _create_dataset("tyson", Settings.TEST_YEARS)
    else:
        _create_dataset(args.name, args.years)
