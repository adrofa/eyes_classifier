import sys
import logging
import pickle
import json


def get_logger(logger_name):
    """Initializes logger and prints 1st log-message into stdout."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger_ = logging.getLogger(logger_name)
    logger_.addHandler(handler)
    logger_.setLevel(logging.INFO)
    logger_.info("Module started.")
    return logger_


def pkl_dump(obj, file):
    """Dump object with pickle.
    Args:
        obj: object to dump.
        file(str): the destination file.
    """
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def pkl_load(file):
    """Load object from pickle.
    Args:
        file(str): the pickle-file containing the object.
    """
    with open(file, "rb") as f:
        obj = pickle.load(f)
    return obj


def jsn_dump(dct, file):
    """Dump dict as json."""
    with open(file, "w") as f:
        json.dump(dct, f, sort_keys=True, indent=4)