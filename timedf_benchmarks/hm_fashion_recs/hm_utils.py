from pathlib import Path
from contextlib import contextmanager

import numpy as np

from timedf.backend import Backend, pd


def check_experimental(modin_exp):
    return modin_exp and Backend.get_modin_cfg() is not None


# Use experimental groupby from
# https://modin.readthedocs.io/en/latest/flow/modin/experimental/reshuffling_groupby.html
@contextmanager
def maybe_modin_exp(modin_exp):
    if check_experimental(modin_exp):
        import modin

        if hasattr(modin.config, "ExperimentalGroupbyImpl"):
            print("Activating exp function")
            modin.config.ExperimentalGroupbyImpl.put(True)
        else:
            print("Using modin without support of experimental groupby")

        try:
            yield None
        finally:
            if hasattr(modin.config, "ExperimentalGroupbyImpl"):
                modin.config.ExperimentalGroupbyImpl.put(False)
    else:
        yield None


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
