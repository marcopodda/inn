import os


class Dirs:
    """Object that stores directory path used throughout the experiments."""

    ROOT_DIR = os.path.abspath(".")
    DATA_DIR = os.path.join(ROOT_DIR, ".data")
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")
    FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
    PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
    MODELS_DIR = os.path.join(RESULTS_DIR, ".models")
    MODELSELECTION_DIR = os.path.join(RESULTS_DIR, "model_selection")
    EVALUATION_DIR = os.path.join(RESULTS_DIR, "evaluation")


class Settings:
    """Object that stores settings used throughout the experiments."""

    N_CPUS = 16
    TRAIN_YEARS = "2008-2014"
    TEST_YEARS = "2015-2016"
    TARGET = "died"
    RANDOM_SEED = 1234
    CANDIDATE_MODELS = ["lr", "knn", "rf", "xgb", "svm", "nn"]
    BASELINES = ["bw", "bwga", "mankt", "tyson", "logreg"]
    TRAINABLE_BASELINES = ["bw", "bwga", "logreg"]
    MODEL_SELECTION_SPLIT = 5
