import argparse


def get_dataset_parser():
    """
        Constructs a Parser for the datasets.py file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=str,
                        choices=["2008-2014", "2015-2016"],
                        default="2008-2014",
                        help="""Which years to consider(2008-2014
                                for training, 2015-2016 for test).""")
    parser.add_argument("--name", type=str,
                        choices=["bw", "bwga", "mankt",
                                 "tyson", "logreg",
                                 "ours", "all"],
                        help="""Dataset name""",
                        default="all")
    return parser


def get_model_selection_parser():
    """
        Constructs an Argument Parser for the model_selection.py file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--test",
                        action="store_true",
                        default=False,
                        help="Use test grid.")
    return parser
