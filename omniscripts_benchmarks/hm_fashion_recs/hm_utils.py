import os
from pathlib import Path

import numpy as np

from omniscripts.pandas_backend import Backend, pd


DEBUG = bool(os.getenv("DEBUG", False))


# Use experimental code that modify benchmark
def check_experimental():
    EXPERIMENTAL = bool(os.getenv("EXPERIMENTAL", False))
    return EXPERIMENTAL and Backend.get_modin_cfg() is not None


if check_experimental():
    grp_kwargs = {"exp_implementation": True}
else:
    grp_kwargs = {}


# TODO: modin bug, that's why we use iloc[]
# https://github.com/modin-project/modin/issues/5461
def modin_fix(df):
    return df.iloc[: len(df)]


def load_data(preprocessed_data_path):
    transactions = pd.read_pickle(preprocessed_data_path / "transactions_train.pkl")
    users = pd.read_pickle(preprocessed_data_path / "users.pkl")
    items = pd.read_pickle(preprocessed_data_path / "items.pkl")

    return transactions, users, items


def get_workdir_paths(raw_data_path, workdir="./hm_tmpdir"):
    """Get paths in the workdir, which is shared across several scripts, and create necessary
    folders."""
    workdir = Path(workdir)

    paths = dict(
        raw_data_path=Path(raw_data_path),
        workdir=workdir,
        preprocessed_data=workdir / "preprocessed",
        artifacts=workdir / "artifacts",
        lfm_features=workdir / "lfm",
        user_features=workdir / "user_features",
    )
    for p in paths.values():
        p.mkdir(exist_ok=True, parents=True)

    return paths


# https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=12):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=12):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
