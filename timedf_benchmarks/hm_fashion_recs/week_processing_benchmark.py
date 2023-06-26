import argparse
from timedf import BenchmarkResults, BaseBenchmark, tm

from .preprocess import transform_data, create_user_ohe_agg
from .hm_utils import load_data, get_workdir_paths
from .candidates import make_one_week_candidates, drop_trivial_users
from .fe import attach_features, get_age_shifts


def feature_engieering(week, paths, use_lfm, modin_exp):
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
            modin_exp=modin_exp,
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
            modin_exp=modin_exp,
        )

    dataset["query_group"] = dataset["week"].astype(str) + "_" + dataset["user"].astype(str)
    dataset = dataset.sort_values(by="query_group").reset_index(drop=True)
    return dataset


def main(paths, modin_exp):
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
            feature_engieering(week=week, paths=paths, use_lfm=False, modin_exp=modin_exp)


class Benchmark(BaseBenchmark):
    __params__ = ("modin_exp",)

    def run_benchmark(self, parameters):
        paths = get_workdir_paths(parameters["data_file"])
        modin_exp = parameters["modin_exp"]
        import modin.config.__main__ as cfg_cmd

        cfg_cmd.print_config_help()
        main(paths, modin_exp=modin_exp)

        task2time = tm.get_results()
        print(task2time)

        return BenchmarkResults(task2time)

    def load_data(self, target_dir, reload=False):
        from timedf.tools.kaggle_load import download_dataset

        url = "https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules"
        download_dataset(
            "h-and-m-personalized-fashion-recommendations",
            local_dir=target_dir,
            reload=reload,
            rules_url=url,
        )

    def add_benchmark_args(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-modin_exp",
            default=False,
            action="store_true",
            help="Use experimental modin groupby from "
            "https://modin.readthedocs.io/en/latest/flow/modin/experimental/reshuffling_groupby.html."
            "\nand perform one additional reshuffling",
        )
