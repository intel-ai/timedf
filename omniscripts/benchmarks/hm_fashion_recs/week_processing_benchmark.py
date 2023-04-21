from omniscripts import BenchmarkResults, BaseBenchmark, tm

from .preprocess import transform_data, create_user_ohe_agg
from .hm_utils import load_data, get_workdir_paths
from .candidates import make_one_week_candidates, drop_trivial_users
from .fe import attach_features, get_age_shifts


def feature_engieering(week, paths, use_lfm):
    with tm.timeit("01-load_data"):
        transactions, users, items = load_data(paths["preprocessed_data"])

    with tm.timeit("02-age_shifts"):
        age_shifts = get_age_shifts(transactions=transactions, users=users)

    with tm.timeit("03-candidates"):
        week_candidates = make_one_week_candidates(
            transactions=transactions,
            users=users,
            items=items,
            week=week,
            user_features_path=paths["user_features"],
            age_shifts=age_shifts,
        )

        candidates = drop_trivial_users(week_candidates)

        candidates.to_pickle(paths["workdir"] / "candidates.pkl")

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
            user_features_path=paths["user_features"],
            lfm_features_path=paths["lfm_features"] if use_lfm else None,
        )

    dataset["query_group"] = dataset["week"].astype(str) + "_" + dataset["user"].astype(str)
    dataset = dataset.sort_values(by="query_group").reset_index(drop=True)
    return dataset


def main(paths):
    with tm.timeit("total"):
        with tm.timeit("01-initial_transform"):
            transform_data(
                input_data_path=paths["raw_data_path"], result_path=paths["preprocessed_data"]
            )

        week = 0
        with tm.timeit("02-create_user_ohe_agg"):
            create_user_ohe_agg(
                week + 1,
                preprocessed_data_path=paths["preprocessed_data"],
                result_path=paths["user_features"],
            )

        with tm.timeit("03-fe"):
            feature_engieering(week=week, paths=paths, use_lfm=False)


class Benchmark(BaseBenchmark):
    def run_benchmark(self, parameters):
        paths = get_workdir_paths(parameters["data_file"])
        main(paths)

        task2time = tm.get_results()
        print(task2time)

        return BenchmarkResults(task2time)
