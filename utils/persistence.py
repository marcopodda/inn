import os
import pickle as pkl

from sklearn.externals import joblib

from config import Dirs


def load(path):
    """
        Loads an object saved in pickle binary format from a given path.

    Arguments:
        path {str} -- Path where the object is saved.

    Returns:
        obj -- Loaded object.
    """
    obj = pkl.load(open(path, "rb"))
    return obj


def save(obj, path):
    """
        Saves a python object in pickle binary format to a given path.

    Arguments:
        obj {type} -- Object to save.
        path {str} -- Path where the object will be saved.
    """
    pkl.dump(obj, open(path, "wb"))


def load_model(model_name):
    """
        Loads a model given its name.

    Arguments:
        model_name {str} -- An identifying string for the model, used
                            to construct the path where the model is stored.
                            Possible choices are:
                            - `nn`: the neural network classifier.
                            - `lr`: the logistic regression classifier.

    Returns:
        model -- A scikit-learn classifier.
    """

    path = os.path.join(Dirs.MODELS_DIR, model_name, "model.pkl")
    model = joblib.load(path)
    return model


def save_model(model, cv_results, model_name):
    """
        Saves a scikit-learn model to file using the joblib library.
        In addition, saves its cross-validation scores, its parameters,
        weights and biases for reproducibility.
        According to scikit-learn documentation:

        "While models saved using one version of scikit-learn might load
        in other versions, this is entirely unsupported and inadvisable.
        It should also be kept in mind that operations performed on such
        data could give different and unexpected results.

        In order to rebuild a similar model with future versions of
        scikit-learn, additional metadata should be saved along the
        pickled model:

        - The training data, e.g. a reference to a immutable snapshot
        - The python source code used to generate the model
        - The versions of scikit-learn and its dependencies
        - The cross validation score obtained on the training data

        This should make it possible to check that the cross-validation
        score is in the same range as before."

    Arguments:
        model {Classifier} -- scikit-learn classifier to save
        model_name {str}   -- An identifying string for the model, used
                              to construct the path where the
                              model will be saved.
    """

    model_dir = os.path.join(Dirs.MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, path)

    path = os.path.join(model_dir, "cv.pkl")
    save(cv_results, path)

    clf = model.named_steps["clf"]
    path = os.path.join(model_dir, "params.pkl")
    save(clf.get_params(), path)

    name = "weights" if model_name == "nn" else "coefs"
    param = clf.coefs_ if model_name == "nn" else clf.coef_
    path = os.path.join(model_dir, "{}.pkl".format(name))
    save(param, path)

    name = "biases" if model_name == "nn" else "intercept"
    param = clf.intercepts_ if model_name == "nn" else clf.intercept_
    path = os.path.join(model_dir, "{}.pkl".format(name))
    save(param, path)


def load_predictions(name):
    """Loads a numpy array containing predictions of a model.

    Arguments:
        name {str}                -- Prefix of the file to load.

    Returns:
        predictions -- A numpy array containing the predictions.
    """

    filename = "{}.pkl".format(name)
    path = os.path.join(Dirs.PREDICTIONS_DIR, filename)
    predictions = load(path)
    return predictions


def save_predictions(predictions, name):
    """Saves a numpy array containing predictions of a model.

    Arguments:
        predictions {numpy array} -- A numpy array containing the predictions.
        name {str}                -- Prefix of the file to save.
    """

    filename = "{}.pkl".format(name)
    path = os.path.join(Dirs.PREDICTIONS_DIR, filename)
    save(predictions, path)


def save_results(results, name):
    """Saves an object containing model selection results.

    Arguments:
        results {numpy array} -- A numpy array containing the results.
        name {str}            -- Prefix of the file to save.
    """

    filename = "{}.pkl".format(name)
    path = os.path.join(Dirs.MODELSELECTION_DIR, filename)
    save(results, path)


def save_evaluation(evaluation, name):
    """Saves a numpy array containing evaluation of a model.

    Arguments:
        evaluation {numpy array} -- A numpy array containing the evaluation.
        name {str}               -- Prefix of the file to save.
    """

    filename = "{}.pkl".format(name)
    path = os.path.join(Dirs.EVALUATION_DIR, filename)
    save(evaluation, path)
