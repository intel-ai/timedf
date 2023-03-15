"""This script reproduces notebook from the original solution."""

import gc
import logging
from pathlib import Path

import catboost
import matplotlib.pyplot as plt
import numpy as np
from omniscripts.benchmark import BaseBenchmark, BenchmarkResults

from omniscripts.pandas_backend import pd, modin_cfg

from .hm_utils import mapk, load_data, get_workdir_paths
from .fe import get_age_shifts, attach_features
from .candidates import create_candidates, make_weekly_candidates
from .preprocess import run_complete_preprocessing
from .tm import tm


logger = logging.getLogger(__name__)


class CFG:
    train_weeks = 6
    n_iterations = 10_000

    use_lfm = True


LIMITED_DATA = True
if LIMITED_DATA:
    CFG.train_weeks = 1
    # CFG.n_iterations = 50


def concat_train(datasets, begin, num):
    train = pd.concat([datasets[idx] for idx in range(begin, begin + num)])
    return train


def make_dataset(
    candidates,
    transactions,
    users,
    items,
    begin_shift=1,
    end_shift=1,
    *,
    age_shifts,
    user_features_path,
):
    # Since the learning period of the pretrained model is different at the time of evaluation and at the time of submission, leave candidates

    datasets = []
    for i, candidates_subset in enumerate(candidates):
        with tm.timeit(f'attach_features_week={i}'):
            dataset = attach_features(
                transactions,
                users,
                items,
                candidates_subset,
                begin_shift + i,
                CFG.train_weeks + end_shift,
                age_shifts=age_shifts,
                user_features_path=user_features_path,
            )

            dataset["query_group"] = dataset["week"].astype(str) + "_" + dataset["user"].astype(str)
            dataset = dataset.sort_values(by="query_group").reset_index(drop=True)
            datasets.append(dataset)

    valid = datasets[0]
    with tm.timeit('concat'):
        train = concat_train(datasets, end_shift, CFG.train_weeks)

    return train, valid


def get_feature_cols(dataset):
    return [c for c in dataset.columns if c not in ["y", "strategy", "query_group", "week"]]


def train_model(*, train, valid=None, best_iteration=None):
    assert (valid is None) ^ (
        best_iteration is None
    ), "We either have val set or already know best iteration"

    feature_columns = get_feature_cols(train)

    cat_feature_values = [c for c in feature_columns if c.endswith("idx")]
    cat_features = [feature_columns.index(c) for c in cat_feature_values]

    data = train[feature_columns]
    y = train["y"]
    group_id = train["query_group"]

    # We make this conversion because Catboost checks type of input dataframe agains pandas
    if modin_cfg is not None:
        data = data._to_pandas()
        y = y._to_pandas()
        group_id = group_id._to_pandas()


    train_dataset = catboost.Pool(
        data=data,
        label=y,
        group_id=group_id,
        cat_features=cat_features,
    )

    if valid is not None:
        data = valid[feature_columns]
        y = valid["y"]
        group_id = valid["query_group"]

        # We make this conversion because Catboost checks type of input dataframe agains pandas
        if modin_cfg is not None:
            data = data._to_pandas()
            y = y._to_pandas()
            group_id = group_id._to_pandas()


        valid_dataset = catboost.Pool(
            data=data,
            label=y,
            group_id=group_id,
            cat_features=cat_features,
        )
    else:
        valid_dataset = None

    params = {
        "loss_function": "YetiRank",
        # If we already know best iteration, then just use it
        "use_best_model": best_iteration is None,
        "one_hot_max_size": 300,
        "iterations": best_iteration or CFG.n_iterations,
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


def predict(dataset, model):
    feature_columns = get_feature_cols(dataset)

    pred = dataset[["user", "item"]].reset_index(drop=True)

    inf_data = dataset[feature_columns]

    if modin_cfg is not None:
        inf_data = inf_data._to_pandas()

    pred["pred"] = model.predict(inf_data)

    pred = pred.groupby(["user", "item"])["pred"].max().reset_index()
    return (
        pred.sort_values(by=["user", "pred"], ascending=False)
        .reset_index(drop=True)
        .groupby("user")["item"]
        .apply(lambda x: list(x)[:12])
        .reset_index()
    )


def validate_model(
    model, transactions, users, items, candidates_valid, age_shifts, user_features_path
):
    dataset_valid_all = attach_features(
        transactions,
        users,
        items,
        candidates_valid,
        1,
        CFG.train_weeks + 1,
        age_shifts=age_shifts,
        user_features_path=user_features_path,
    )

    pred = predict(dataset_valid_all, model)

    gt = (
        transactions.query("week == 0")
        .groupby("user")["item"]
        .apply(list)
        .reset_index()
        .rename(columns={"item": "gt"})
    )
    merged = gt.merge(pred, on="user", how="left")
    merged["item"] = merged["item"].fillna("").apply(list)

    return mapk(merged["gt"], merged["item"])


def predict_new_week(*, model, transactions, users, items, age_shifts, user_features_path):
    """This function predicts in chunks to avid OOM problem"""
    all_users = users["user"].values
    preds = []
    n_split_prediction = 10
    n_chunk = (len(all_users) + n_split_prediction - 1) // n_split_prediction
    for i in range(0, len(all_users), n_chunk):
        logger.info("chunk: %s", i)
        target_users = all_users[i : i + n_chunk]  # noqa: E203

        candidates = create_candidates(
            users=users,
            transactions=transactions,
            items=items,
            target_users=target_users,
            week=0,
            user_features_path=user_features_path,
            age_shifts=age_shifts,
        )
        candidates = attach_features(
            transactions,
            users,
            items,
            candidates,
            0,
            CFG.train_weeks,
            age_shifts=age_shifts,
            user_features_path=user_features_path,
        )

        preds.append(predict(candidates, model))

    pred = pd.concat(preds).reset_index(drop=True)
    assert len(pred) == len(all_users)
    assert np.array_equal(pred["user"].values, all_users)
    return pred


def prepare_submission(*, pred, working_dir, preprocessed_data_path):
    mp_user = pd.read_pickle(preprocessed_data_path / "mp_customer_id.pkl")
    mp_item = pd.read_pickle(preprocessed_data_path / "mp_article_id.pkl")

    a_user = mp_user["val"].values
    a_item = mp_item["val"].values

    pred["customer_id"] = pred["user"].apply(lambda x: a_user[x])
    pred["prediction"] = pred["item"].apply(lambda x: list(map(lambda y: a_item[y], x)))

    pred["prediction"] = pred["prediction"].apply(lambda x: " ".join(map(str, x)))

    submission = pred[["customer_id", "prediction"]]
    submission.to_csv(working_dir / "submission.csv", index=False)


def train_eval(
    train, valid, transactions, users, items, candidates_valid, age_shifts, user_features_path
):
    
    with tm.timeit('02-train'):
        model = train_model(train=train, valid=valid)
        best_iteration = model.get_best_iteration()

    del train, valid
    gc.collect()

    with tm.timeit('03-eval'):
        metric = validate_model(
            model=model,
            transactions=transactions,
            users=users,
            items=items,
            candidates_valid=candidates_valid,
            age_shifts=age_shifts,
            user_features_path=user_features_path,
        )
    logger.info("mAP@12: %s", metric)
    return best_iteration


def make_submission(candidates, transactions, users, items, best_iteration, age_shifts, paths):
    with tm.timeit('01-make_dataset'):
        train, valid = make_dataset(
            candidates=candidates,
            transactions=transactions,
            users=users,
            items=items,
            begin_shift=1,
            end_shift=0,
            age_shifts=age_shifts,
            user_features_path=paths["user_features"],
        )   

    with tm.timeit('02-train'):
        model = train_model(train=train, best_iteration=best_iteration)

    del train, valid
    del candidates
    gc.collect()

    with tm.timeit('03-predict'):
        pred = predict_new_week(
            model=model,
            transactions=transactions,
            users=users,
            items=items,
            age_shifts=age_shifts,
            user_features_path=paths["user_features"],
        )
    with tm.timeit('04-prepare_sub'):
        prepare_submission(
            pred=pred, working_dir=paths["workdir"], preprocessed_data_path=paths["preprocessed_data"]
        )


def main(raw_data_path, paths):
    with tm.timeit('total'):
        with tm.timeit('01-processing'):
            run_complete_preprocessing(
                raw_data_path=raw_data_path,
                preprocessed_path=paths['preprocessed_data'],
                paths=paths, n_weeks=CFG.train_weeks + 1, use_lfm=CFG.use_lfm
            )
        
        with tm.timeit('02-load_processed'):
            transactions, users, items = load_data(
                preprocessed_data_path=paths["preprocessed_data"]
            )

        with tm.timeit('03-age_shifts'):
            age_shifts = get_age_shifts(transactions=transactions, users=users)

        with tm.timeit('04-weekly_candidates'):
            candidates, candidates_valid = make_weekly_candidates(
                transactions=transactions,
                users=users,
                items=items,
                train_weeks=CFG.train_weeks,
                user_features_path=paths["user_features"],
                age_shifts=age_shifts,
            )

        with tm.timeit('05-make_dataset'):
            train, valid = make_dataset(
                candidates=candidates,
                transactions=transactions,
                users=users,
                items=items,
                begin_shift=1,
                end_shift=1,
                age_shifts=age_shifts,
                user_features_path=paths["user_features"],
            )

        with tm.timeit('06-train_eval'):
            best_iteration = train_eval(
                train=train,
                valid=valid,
                candidates_valid=candidates_valid,
                transactions=transactions,
                users=users,
                items=items,
                age_shifts=age_shifts,
                user_features_path=paths["user_features"],
        )

        del candidates_valid
        gc.collect()

        # with tm.timeit('06-retrain_whole_dataset'):
        #     make_submission(
        #         candidates=candidates,
        #         transactions=transactions,
        #         users=users,
        #         items=items,
        #         best_iteration=best_iteration,
        #         age_shifts=age_shifts,
        #         paths=paths,
        #     )


class Benchmark(BaseBenchmark):
    __unsupported_params__ = ("optimizer", "dfiles_num")

    def run_benchmark(self, parameters):
        paths = get_workdir_paths()
        main(raw_data_path=Path(parameters["data_file"]), paths=paths)

        task2time = tm.get_results()
        print(task2time)

        return BenchmarkResults(task2time)
