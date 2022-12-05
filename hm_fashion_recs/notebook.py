####################
# Notebook content #

from contextlib import contextmanager
import gc
from pathlib import Path
import pickle

import catboost
import matplotlib.pyplot as plt
import numpy as np

# import modin.pandas as pd
import pandas as pd
from metric import mapk

from vars import preprocessed_data_path, train_weeks, lfm_features_path, working_dir


# train params
# FIXME: choose parameter
# n_iterations = 10_000
N_ITERATIONS = 50


def load_data(preprocessed_data_path):
    transactions = pd.read_pickle(preprocessed_data_path / "transactions_train.pkl")
    users = pd.read_pickle(preprocessed_data_path / "users.pkl")
    items = pd.read_pickle(preprocessed_data_path / "items.pkl")

    return transactions, users, items


def concat_train(datasets, begin, num):
    train = pd.concat([datasets[idx] for idx in range(begin, begin + num)])
    return train


def make_dataset(candidates, begin_shift=1, end_shift=1):
    # Since the learning period of the pretrained model is different at the time of evaluation and at the time of submission, leave candidates

    for i, candidates_subset in enumerate(candidates):
        dataset = attach_features(
            transactions, users, items, candidates_subset, begin_shift + i, train_weeks + end_shift
        )

        dataset["query_group"] = (
            datasets[idx]["week"].astype(str) + "_" + datasets[idx]["user"].astype(str)
        )
        dataset = dataset.sort_values(by="query_group").reset_index(drop=True)

    valid = datasets[0]
    train = concat_train(datasets, end_shift, train_weeks)

    return train, valid


def get_query_group(df):
    def run_length_encoding(sequence):
        comp_seq_index, = np.concatenate(([True], sequence[1:] != sequence[:-1], [True])).nonzero()
        return sequence[comp_seq_index[:-1]], np.ediff1d(comp_seq_index)

    users = df["user"].values
    _, group = run_length_encoding(users)
    return list(group)


def train_model(*, train, valid=None, feature_columns, cat_features, best_iteration=None):
    assert (valid is None) ^ (
        best_iteration is None
    ), "We either have val set or already know best iteration"
    train_dataset = catboost.Pool(
        data=train[feature_columns],
        label=train["y"],
        group_id=train["query_group"],
        cat_features=cat_features,
    )

    valid_dataset = (
        None
        if valid is None
        else catboost.Pool(
            data=valid[feature_columns],
            label=valid["y"],
            group_id=valid["query_group"],
            cat_features=cat_features,
        )
    )

    params = {
        "loss_function": "YetiRank",
        # If we already know best iteration, then just use it
        "use_best_model": best_iteration is None,
        "one_hot_max_size": 300,
        "iterations": best_iteration or N_ITERATIONS,
    }
    model = catboost.CatBoost(params)
    model.fit(train_dataset, eval_set=valid_dataset)

    if valid is not None:
        plt.plot(model.get_evals_result()["validation"]["PFound"])

    feature_importance = model.get_feature_importance(train_dataset)
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(8, 16))
    plt.yticks(range(len(feature_columns)), np.array(feature_columns)[sorted_idx])
    plt.barh(range(len(feature_columns)), feature_importance[sorted_idx])
    return model


def predict_new_week(transactions, users, items, feature_columns):
    """This function predicts in chunks to avid OOM problem"""
    all_users = users["user"].values
    preds = []
    n_split_prediction = 10
    n_chunk = (len(all_users) + n_split_prediction - 1) // n_split_prediction
    for i in range(0, len(all_users), n_chunk):
        print(f"chunk: {i}")
        target_users = all_users[i : i + n_chunk]

        candidates = create_candidates(transactions, target_users, 0)
        candidates = attach_features(transactions, users, items, candidates, 0, train_weeks)

        candidates["pred"] = model.predict(candidates[feature_columns])
        pred = candidates.groupby(["user", "item"])["pred"].max().reset_index()
        pred = (
            pred.sort_values(by=["user", "pred"], ascending=False)
            .reset_index(drop=True)
            .groupby("user")["item"]
            .apply(lambda x: list(x)[:12])
            .reset_index()
        )
        preds.append(pred)

    pred = pd.concat(preds).reset_index(drop=True)
    assert len(pred) == len(all_users)
    assert np.array_equal(pred["user"].values, all_users)
    return pred


def prepare_submission(pred, preprocessed_path: Path, tmp_path: Path):
    mp_user = pd.read_pickle(preprocessed_path / "mp_customer_id.pkl")
    mp_item = pd.read_pickle(preprocessed_path / "mp_article_id.pkl")

    a_user = mp_user["val"].values
    a_item = mp_item["val"].values

    pred["customer_id"] = pred["user"].apply(lambda x: a_user[x])
    pred["prediction"] = pred["item"].apply(lambda x: list(map(lambda y: a_item[y], x)))

    pred["prediction"] = pred["prediction"].apply(lambda x: " ".join(map(str, x)))

    submission = pred[["customer_id", "prediction"]]
    submission.to_csv(tmp_path / "submission.csv", index=False)


transactions, users, items = load_data(preprocessed_data_path=preprocessed_data_path)
age_shifts = get_age_shifts(transactions)

candidates, candidates_valid_all = make_candidates(transactions=transactions)

train, valid = make_dataset(candidates=candidates, begin_shift=1, end_shift=1)
dataset_valid_all = attach_features(
    transactions, users, items, candidates_valid_all, 1, train_weeks + 1
)


#################
# Train a model #
#################

feature_columns = [c for c in valid.columns if c not in ["y", "strategy", "query_group", "week"]]
print(feature_columns)

cat_feature_values = [c for c in feature_columns if c.endswith("idx")]
cat_features = [feature_columns.index(c) for c in cat_feature_values]
print(cat_feature_values, cat_features)


model = train_model(
    train=train, valid=valid, feature_columns=feature_columns, cat_features=cat_features
)
best_iteration = model.get_best_iteration()

del train, valid

gc.collect()
with open("output/model_for_validation.pkl", "wb") as f:
    pickle.dump(model, f)


######################
# Predict on val set #
######################

pred = dataset_valid_all[["user", "item"]].reset_index(drop=True)
pred["pred"] = model.predict(dataset_valid_all[feature_columns])

pred = pred.groupby(["user", "item"])["pred"].max().reset_index()
pred = (
    pred.sort_values(by=["user", "pred"], ascending=False)
    .reset_index(drop=True)
    .groupby("user")["item"]
    .apply(lambda x: list(x)[:12])
    .reset_index()
)

gt = (
    transactions.query("week == 0")
    .groupby("user")["item"]
    .apply(list)
    .reset_index()
    .rename(columns={"item": "gt"})
)
merged = gt.merge(pred, on="user", how="left")
merged["item"] = merged["item"].fillna("").apply(list)

merged.to_pickle(f"output/merged_{dataset}.pkl")
dataset_valid_all.to_pickle(f"output/valid_all_{dataset}.pkl")

print("mAP@12:", mapk(merged["gt"], merged["item"]))

################
# Submission #
################

# For submission model training
train, valid = make_dataset(candidates=candidates, begin_shift=1, end_shift=0)

model = train_model(
    train=train,
    feature_columns=feature_columns,
    cat_features=cat_features,
    best_iteration=best_iteration,
)

del train, valid
gc.collect()
with open("output/model_for_submission.pkl", "wb") as f:
    pickle.dump(model, f)

del candidates, candidates_valid_all
gc.collect()


pred = predict_new_week(
    transactions=transactions, users=users, items=items, feature_columns=feature_columns
)
prepare_submission(pred, preprocessed_path=preprocessed_data_path, tmp_path=working_dir)
