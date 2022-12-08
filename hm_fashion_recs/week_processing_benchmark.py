import os

from hm_fashion_recs.lfm import train_lfm
from hm_fashion_recs.preprocess import transform_data, create_user_ohe_agg
from hm_fashion_recs.notebook import get_age_shifts, load_data
from hm_fashion_recs.candidates import make_one_week_candidates, drop_trivial_users
from hm_fashion_recs.fe import attach_features

from hm_fashion_recs.vars import (
    raw_data_path,
    preprocessed_data_path,
    user_features_path,
    lfm_features_path,
    dim,
    working_dir,
)
from hm_fashion_recs.tm import tm


from utils import (
    check_support,
    import_pandas_into_module_namespace,
    print_results,
    trigger_execution,
    Config,
)


class CFG:
    raw_data_path = raw_data_path
    preprocessed_data_path = preprocessed_data_path
    lfm_features_path = lfm_features_path
    working_dir = working_dir
    user_features_path = user_features_path

    use_lfm = False

    # def update(self, raw_data_path, ):
    #     CFG.


def feature_engieering(week):
    transactions, users, items = load_data()

    age_shifts = get_age_shifts(transactions=transactions, users=users)

    week_candidates = make_one_week_candidates(
        transactions=transactions,
        users=users,
        items=items,
        week=week,
        user_features_path=CFG.user_features_path,
        age_shifts=age_shifts,
    )

    candidates = drop_trivial_users(week_candidates)

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


def main():
    total_timer = tm.timeit("total")
    with tm.timeit("1-initial_transform"):
        transform_data(input_data_path=CFG.raw_data_path, result_path=CFG.preprocessed_data_path)

    week = 0
    with tm.timeit("2-create_user_ohe_agg"):
        create_user_ohe_agg(
            week + 1,
            preprocessed_data_path=CFG.preprocessed_data_path,
            result_path=CFG.user_features_path,
        )

    # if CFG.use_lfm:
    #     with tm.timeit("3-create_lfm_features"):
    #         # features are pretraied, that's why +2
    #         train_lfm(week=week + 2, lfm_features_path=CFG.lfm_features_path, dim=dim)

    with tm.timeit("3-fe"):
        feature_engieering(week=week)
    total_timer.stop()

    print(tm.get_results())


def run_benchmark(parameters):
    check_support(parameters, unsupported_params=["optimizer", "dfiles_num"])

    import_pandas_into_module_namespace(
        namespace=run_benchmark.__globals__,
        mode=parameters["pandas_mode"],
        ray_tmpdir=parameters["ray_tmpdir"],
        ray_memory=parameters["ray_memory"],
    )
    # Update config in case some envs changed after the import
    Config.init(
        MODIN_IMPL="pandas" if parameters["pandas_mode"] == "Pandas" else "modin",
        MODIN_STORAGE_FORMAT=os.getenv("MODIN_STORAGE_FORMAT"),
        MODIN_ENGINE=os.getenv("MODIN_ENGINE"),
    )

    is_hdk_mode = parameters["pandas_mode"] == "Modin_on_hdk"

    main()

    task2time = tm.get_results()

    results = [
        {"query_name": b, "result": t, "Backend": parameters["pandas_mode"]}
        for b, t in task2time.items()
    ]

    return {"ETL": results}


if __name__ == "__main__":
    main()
