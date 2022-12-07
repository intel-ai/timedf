import os
import pickle
from pathlib import Path

import pandas as pd
from lightfm import LightFM
from scipy import sparse

from hm_fashion_recs.vars import lfm_features_path


LIGHTFM_PARAMS = {
    "learning_schedule": "adadelta",
    "loss": "bpr",
    "learning_rate": 0.005,
    "random_state": 42,
}
EPOCHS = 100


def train_lfm(*, lfm_features_path: Path = lfm_features_path, week: int, dim: int = 16):
    dataset = "100"

    path_prefix = lfm_features_path / f"lfm_i_i_dataset{dataset}_week{week}_dim{dim}"
    print(path_prefix)
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
