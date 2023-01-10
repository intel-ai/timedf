from pathlib import Path

from hm_fashion_recs.preprocess import transform_data, create_user_ohe_agg
from hm_fashion_recs.notebook import get_age_shifts, load_data
from hm_fashion_recs.candidates import make_one_week_candidates, drop_trivial_users
from hm_fashion_recs.fe import attach_features


from hm_fashion_recs.vars import (
    preprocessed_data_path,
    user_features_path,
    lfm_features_path,
    working_dir,
)
from hm_fashion_recs.tm import tm


from utils import check_support


class CFG:
    preprocessed_data_path = preprocessed_data_path
    lfm_features_path = lfm_features_path
    working_dir = working_dir
    user_features_path = user_features_path

    use_lfm = False


def feature_engieering(week):
    with tm.timeit("01-load_data"):
        transactions, users, items = load_data()

    with tm.timeit("02-age_shifts"):
        age_shifts = get_age_shifts(transactions=transactions, users=users)

    with tm.timeit("03-candidates"):
        week_candidates = make_one_week_candidates(
            transactions=transactions,
            users=users,
            items=items,
            week=week,
            user_features_path=CFG.user_features_path,
            age_shifts=age_shifts,
        )

        candidates = drop_trivial_users(week_candidates)

        candidates.to_pickle(CFG.working_dir / "candidates.pkl")

    with tm.timeit("04-attach_features"):
        dataset = attach_features(
            transactions=transactions,
            users=users,
            items=items,
            candidates=candidates,
            # +1 because train data comes one week earlier
            week=week + 1,
            pretrain_week=week + 2,
            age_shifts=age_shifts,
            user_features_path=CFG.user_features_path,
            lfm_features_path=CFG.lfm_features_path if CFG.use_lfm else None,
        )

    dataset["query_group"] = dataset["week"].astype(str) + "_" + dataset["user"].astype(str)
    dataset = dataset.sort_values(by="query_group").reset_index(drop=True)
    return dataset


def main(raw_data_path):
    with tm.timeit("total"):
        with tm.timeit("01-initial_transform"):
            transform_data(input_data_path=raw_data_path, result_path=CFG.preprocessed_data_path)

        week = 0
        with tm.timeit("02-create_user_ohe_agg"):
            create_user_ohe_agg(
                week + 1,
                preprocessed_data_path=CFG.preprocessed_data_path,
                result_path=CFG.user_features_path,
            )

        with tm.timeit("03-fe"):
            feature_engieering(week=week)

    print(tm.get_results())


def run_benchmark(parameters):
    check_support(parameters, unsupported_params=["optimizer", "dfiles_num"])

    raw_data_path = Path(parameters["data_file"].strip("'"))
    main(raw_data_path=raw_data_path)

    task2time = tm.get_results()

    results = [
        {"query_name": b, "result": t, "Backend": parameters["pandas_mode"]}
        for b, t in task2time.items()
    ]

    return {"ETL": results}
