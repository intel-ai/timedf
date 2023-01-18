from __future__ import annotations
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from lightfm import LightFM

from utils.pandas_backend import pd


LIGHTFM_PARAMS = {
    "learning_schedule": "adadelta",
    "loss": "bpr",
    "learning_rate": 0.005,
    "random_state": 42,
}
EPOCHS = 100


class CFG:
    """Configuration for preprocessing"""

    dim = 16


def train_lfm(*, lfm_features_path: Path, week: int, dim: int = CFG.dim):
    dataset = "100"

    path_prefix = lfm_features_path / f"lfm_i_i_dataset{dataset}_week{week}_dim{dim}"
    transactions = pd.read_pickle(f"input/{dataset}/transactions_train.pkl")
    users = pd.read_pickle(f"input/{dataset}/users.pkl")
    items = pd.read_pickle(f"input/{dataset}/items.pkl")
    n_user = len(users)
    n_item = len(items)
    a = transactions.query("@week <= week")[["user", "item"]].drop_duplicates(ignore_index=True)
    a_train = sparse.lil_matrix((n_user, n_item))
    a_train[a["user"], a["item"]] = 1

    lightfm_params = LIGHTFM_PARAMS.copy()
    lightfm_params["no_components"] = dim

    model = LightFM(**lightfm_params)
    model.fit(a_train, epochs=EPOCHS, num_threads=os.cpu_count(), verbose=True)
    save_path = f"{path_prefix}_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)


def _load_resources(lfm_features_path, week: int, dim: int = CFG.dim):
    path_prefix = lfm_features_path / f"lfm_i_i_week{week}_dim{dim}"
    model_path = f"{path_prefix}_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    user_features = None
    item_features = None
    return model, user_features, item_features


def calc_embeddings(
    lfm_features_path, week: int, dim: int = CFG.dim
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model, user_features, item_features = _load_resources(lfm_features_path, week, dim)

    biases, embeddings = model.get_user_representations(user_features)
    n_user = len(biases)
    a = np.hstack([embeddings, biases.reshape(n_user, 1)])
    user_embeddings = pd.DataFrame(a, columns=[f"user_rep_{i}" for i in range(dim + 1)])
    user_embeddings = pd.concat([pd.DataFrame({"user": range(n_user)}), user_embeddings], axis=1)

    biases, embeddings = model.get_item_representations(item_features)
    n_item = len(biases)
    a = np.hstack([embeddings, biases.reshape(n_item, 1)])
    item_embeddings = pd.DataFrame(a, columns=[f"item_rep_{i}" for i in range(dim + 1)])
    item_embeddings = pd.concat([pd.DataFrame({"item": range(n_item)}), item_embeddings], axis=1)
    return user_embeddings, item_embeddings
