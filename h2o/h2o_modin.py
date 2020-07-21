# coding: utf-8
# This script is ported to omniscripts repository from https://github.com/h2oai/db-benchmark
import os
import sys
import traceback
import warnings
from timeit import default_timer as timer
import gc

from utils import (
    import_pandas_into_module_namespace,
    print_results,
    make_chk,
    memory_usage,
    files_names_from_pattern,
    join_to_tbls,
    check_support,
)

warnings.filterwarnings("ignore")


def execute_groupby_query_chk_expr_v1(ans):  # q1, q2
    return [ans["v1"].sum()]


def execute_groupby_query_chk_expr_v2(ans):  # q3
    return [ans["v1"].sum(), ans["v3"].sum()]


def execute_groupby_query_chk_expr_v3(ans):  # q4, q5
    return [ans["v1"].sum(), ans["v2"].sum(), ans["v3"].sum()]


def execute_groupby_query_chk_expr_v4(ans):  # q6
    return [ans["v3"]["median"].sum(), ans["v3"]["std"].sum()]


def execute_groupby_query_chk_expr_v5(ans):  # q7
    return [ans["range_v1_v2"].sum()]


def execute_groupby_query_chk_expr_v6(ans):  # q8
    return [ans["v3"].sum()]


def execute_groupby_query_chk_expr_v7(ans):  # q9
    return [ans["r2"].sum()]


def execute_groupby_query_chk_expr_v8(ans):  # q10
    return [ans["v3"].sum(), ans["v1"].sum()]


def execute_join_query_chk_expr(ans):  # q1, q2, q3, q4, q5
    return [ans["v1"].sum(), ans["v2"].sum()]


groupby_queries_chk_funcs = {
    "groupby_query1": execute_groupby_query_chk_expr_v1,
    "groupby_query2": execute_groupby_query_chk_expr_v1,
    "groupby_query3": execute_groupby_query_chk_expr_v2,
    "groupby_query4": execute_groupby_query_chk_expr_v3,
    "groupby_query5": execute_groupby_query_chk_expr_v3,
    "groupby_query6": execute_groupby_query_chk_expr_v4,
    "groupby_query7": execute_groupby_query_chk_expr_v5,
    "groupby_query8": execute_groupby_query_chk_expr_v6,
    "groupby_query9": execute_groupby_query_chk_expr_v7,
    "groupby_query10": execute_groupby_query_chk_expr_v8,
    "join_query1": execute_join_query_chk_expr,
    "join_query2": execute_join_query_chk_expr,
    "join_query3": execute_join_query_chk_expr,
    "join_query4": execute_join_query_chk_expr,
    "join_query5": execute_join_query_chk_expr,
}


def execute_groupby_query_expr_v1(x, groupby_cols, agg_cols_funcs):  # q1, q2, q3, q4, q5, q6, q10
    return x.groupby(groupby_cols).agg(agg_cols_funcs)


def execute_groupby_query_expr_v2(x, groupby_cols, agg_cols_funcs, range_cols):  # q7
    return (
        x.groupby(groupby_cols)
        .agg(agg_cols_funcs)
        .assign(range_v1_v2=lambda x: x[range_cols[0]] - x[range_cols[0]])[["range_v1_v2"]]
    )


def execute_groupby_query_expr_v3(x, select_cols, sort_col, sort_ascending, groupby_cols):  # q8
    return (
        x[select_cols]
        .sort_values(sort_col, ascending=sort_ascending)
        .groupby(groupby_cols)
        .head(2)
    )


def execute_groupby_query_expr_v4(x, select_cols, groupby_cols, apply_cols):  # q9
    return (
        x[select_cols]
        .groupby(groupby_cols)
        .apply(lambda x: pd.Series({apply_cols[0]: x.corr()[apply_cols[1]][apply_cols[2]] ** 2}))
    )


def execute_join_query_expr(x, y, on, how="inner"):  # q1, q2, q3, q4, q5
    return x.merge(y, how=how, on=on)


queries_funcs = {
    "groupby_query1": execute_groupby_query_expr_v1,
    "groupby_query2": execute_groupby_query_expr_v1,
    "groupby_query3": execute_groupby_query_expr_v1,
    "groupby_query4": execute_groupby_query_expr_v1,
    "groupby_query5": execute_groupby_query_expr_v1,
    "groupby_query6": execute_groupby_query_expr_v1,
    "groupby_query7": execute_groupby_query_expr_v2,
    "groupby_query8": execute_groupby_query_expr_v3,
    "groupby_query9": execute_groupby_query_expr_v4,
    "groupby_query10": execute_groupby_query_expr_v1,
    "join_query1": execute_join_query_expr,
    "join_query2": execute_join_query_expr,
    "join_query3": execute_join_query_expr,
    "join_query4": execute_join_query_expr,
    "join_query5": execute_join_query_expr,
}


def execute_query_run(
    query_args, queries_results, query_name, question, run_number, extended_functionality=False
):
    if run_number not in [1, 2]:
        raise ValueError(f"Accepted run_number values are 1 and 2, actually passed {run_number}")

    gc.collect()
    t_start = timer()
    ans = queries_funcs[query_name](**query_args)
    print(ans.shape)
    queries_results[query_name]["t_run" + str(run_number)] = timer() - t_start
    if extended_functionality:
        m = memory_usage()
        t_start = timer()
        chk = groupby_queries_chk_funcs[query_name](ans)
        queries_results[query_name]["chk_t_run" + str(run_number)] = timer() - t_start
        chk = make_chk(chk)
        print(
            f"query name: {query_name}",
            f"question: {question}",
            "run" + str(run_number),
            f"in_rows: {query_args['x'].shape[0]}",
            f"out_rows: {ans.shape[0]}",
            f"out_cols: {ans.shape[1]}",
            f"time_sec: {queries_results[query_name]['t_run' + str(run_number)]}",
            f"mem_gb: {m}",
            f"chk: {chk}",
            f"chk_time_sec: {queries_results[query_name]['chk_t_run' + str(run_number)]}",
            sep=", ",
        )
    del ans


def execute_query(query_args, queries_results, query_name, question, extended_functionality=False):
    for run_number in [1, 2]:
        execute_query_run(
            query_args,
            queries_results,
            query_name,
            question,
            run_number=run_number,
            extended_functionality=extended_functionality,
        )


def groupby_query1_modin(x, queries_results, extended_functionality):
    query_name = "groupby_query1"
    question = "sum v1 by id1"  # 1
    query_args = {"x": x, "groupby_cols": ["id1"], "agg_cols_funcs": {"v1": "sum"}}

    execute_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
        extended_functionality=extended_functionality,
    )


def groupby_query2_modin(x, queries_results, extended_functionality):
    query_name = "groupby_query2"
    question = "sum v1 by id1:id2"  # 2
    query_args = {"x": x, "groupby_cols": ["id1", "id2"], "agg_cols_funcs": {"v1": "sum"}}

    execute_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
        extended_functionality=extended_functionality,
    )


def groupby_query3_modin(x, queries_results, extended_functionality):
    query_name = "groupby_query3"
    question = "sum v1 mean v3 by id3"  # 3
    query_args = {"x": x, "groupby_cols": ["id3"], "agg_cols_funcs": {"v1": "sum", "v3": "mean"}}

    execute_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
        extended_functionality=extended_functionality,
    )


def groupby_query4_modin(x, queries_results, extended_functionality):
    query_name = "groupby_query4"
    question = "mean v1:v3 by id4"  # 4
    query_args = {
        "x": x,
        "groupby_cols": ["id4"],
        "agg_cols_funcs": {"v1": "mean", "v2": "mean", "v3": "mean"},
    }

    execute_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
        extended_functionality=extended_functionality,
    )


def groupby_query5_modin(x, queries_results, extended_functionality):
    query_name = "groupby_query5"
    question = "sum v1:v3 by id6"  # 5
    query_args = {
        "x": x,
        "groupby_cols": ["id6"],
        "agg_cols_funcs": {"v1": "sum", "v2": "sum", "v3": "sum"},
    }

    execute_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
        extended_functionality=extended_functionality,
    )


def groupby_query6_modin(x, queries_results, extended_functionality):
    query_name = "groupby_query6"
    question = "median v3 sd v3 by id4 id5"  # q6
    query_args = {
        "x": x,
        "groupby_cols": ["id4", "id5"],
        "agg_cols_funcs": {"v3": ["median", "std"]},
    }

    execute_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
        extended_functionality=extended_functionality,
    )


def groupby_query7_modin(x, queries_results, extended_functionality):
    query_name = "groupby_query7"
    question = "max v1 - min v2 by id3"  # q7
    query_args = {
        "x": x,
        "groupby_cols": ["id3"],
        "agg_cols_funcs": {"v1": "max", "v2": "min"},
        "range_cols": ["v1", "v2"],
    }

    execute_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
        extended_functionality=extended_functionality,
    )


def groupby_query8_modin(x, queries_results, extended_functionality):
    query_name = "groupby_query8"
    question = "largest two v3 by id6"  # q8
    query_args = {
        "x": x,
        "select_cols": ["id6", "v3"],
        "sort_col": "v3",
        "sort_ascending": False,
        "groupby_cols": ["id6"],
    }

    execute_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
        extended_functionality=extended_functionality,
    )


def groupby_query9_modin(x, queries_results, extended_functionality):
    query_name = "groupby_query9"
    question = "regression v1 v2 by id2 id4"  # q9
    query_args = {
        "x": x,
        "select_cols": ["id2", "id4", "v1", "v2"],
        "groupby_cols": ["id2", "id4"],
        "apply_cols": ["r2", "v1", "v2"],
    }

    execute_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
        extended_functionality=extended_functionality,
    )


def groupby_query10_modin(x, queries_results, extended_functionality):
    query_name = "groupby_query10"
    question = "sum v3 count by id1:id6"  # q10
    query_args = {
        "x": x,
        "groupby_cols": ["id1", "id2", "id3", "id4", "id5", "id6"],
        "agg_cols_funcs": {"v3": "sum", "v1": "count"},
    }

    execute_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
        extended_functionality=extended_functionality,
    )


def join_query1_modin(x, ys, queries_results, extended_functionality):
    query_name = "join_query1"
    question = "small inner on int"  # q1
    query_args = {
        "x": x,
        "y": ys[0],
        "on": "id1",
    }

    execute_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
        extended_functionality=extended_functionality,
    )


def join_query2_modin(x, ys, queries_results, extended_functionality):
    query_name = "join_query2"
    question = "medium inner on int"  # q2
    query_args = {
        "x": x,
        "y": ys[1],
        "on": "id2",
    }

    execute_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
        extended_functionality=extended_functionality,
    )


def join_query3_modin(x, ys, queries_results, extended_functionality):
    query_name = "join_query3"
    question = "medium outer on int"  # q3
    query_args = {
        "x": x,
        "y": ys[1],
        "how": "left",
        "on": "id2",
    }

    execute_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
        extended_functionality=extended_functionality,
    )


def join_query4_modin(x, ys, queries_results, extended_functionality):
    query_name = "join_query4"
    question = "medium inner on factor"  # q4
    query_args = {
        "x": x,
        "y": ys[1],
        "on": "id5",
    }

    execute_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
        extended_functionality=extended_functionality,
    )


def join_query5_modin(x, ys, queries_results, extended_functionality):
    query_name = "join_query5"
    question = "big inner on int"  # q5
    query_args = {
        "x": x,
        "y": ys[2],
        "on": "id3",
    }

    execute_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
        extended_functionality=extended_functionality,
    )


def queries_modin(filename, pandas_mode, extended_functionality):
    data_files_names = files_names_from_pattern(filename)
    data_for_groupby_queries = []
    data_for_join_queries = []
    for f in data_files_names:
        if f.split("/")[-1].startswith("G1"):
            data_for_groupby_queries.append(f)
        elif f.split("/")[-1].startswith("J1"):
            data_for_join_queries.append(f)
        else:
            raise AttributeError(f"Unrecognized file is passed as -data_file flag argument: {f}")

    groupby_queries_files_number = len(data_for_groupby_queries)
    join_queries_files_number = len(data_for_join_queries)
    accepted_number_of_files_for_join_queries = [0, 1, 4]

    if all([groupby_queries_files_number, join_queries_files_number]):
        raise AttributeError(
            "Only one type of queries (groupby or join) can be executed during one run, but files for both queries are passed with -data_file flag"
        )
    elif groupby_queries_files_number > 1:
        raise AttributeError(
            f"Only one file for one run is accepted for groupby queries, actually passed {groupby_queries_files_number}: {data_for_groupby_queries}"
        )
    elif join_queries_files_number not in accepted_number_of_files_for_join_queries:
        raise AttributeError(
            f"Accepted numbers of files for join queries are {accepted_number_of_files_for_join_queries}, actually passed {join_queries_files_number}: {data_for_join_queries}"
        )
    elif join_queries_files_number and sum("NA" in f for f in data_for_join_queries) != 1:
        raise FileNotFoundError(
            "Data files for join queries should contain file (only one) with NA component in the file name"
        )

    queries_results_fields = ["t_run1", "chk_t_run1", "t_run2", "chk_t_run2"]
    if groupby_queries_files_number:
        print(f"loading dataset {data_for_groupby_queries[0]}")
        t0 = timer()
        x = pd.read_csv(data_for_groupby_queries[0])
        x_data_file_import_time = timer() - t0

        # groupby_query9 currently fails with message:
        # numpy.core._exceptions.UFuncTypeError: ufunc 'subtract' did not contain a loop
        # with signature matching types (dtype('<U2'), dtype('<U2')) -> dtype('<U2')
        queries = {
            "groupby_query1": groupby_query1_modin,
            "groupby_query2": groupby_query2_modin,
            "groupby_query3": groupby_query3_modin,
            "groupby_query4": groupby_query4_modin,
            "groupby_query5": groupby_query5_modin,
            # "groupby_query6": groupby_query6_modin, # NotImplementedError: unsupported aggreagte median
            "groupby_query7": groupby_query7_modin,
            # "groupby_query8": groupby_query8_modin, # NotImplementedError: reindex is not yet suported in DFAlgQueryCompiler
            # "groupby_query9": groupby_query9_modin,
            # "groupby_query10": groupby_query10_modin,
        }
        queries_results = {x: {y: 0.0 for y in queries_results_fields} for x in queries.keys()}
        x_data_file_size = os.path.getsize(data_for_groupby_queries[0]) / 1024 / 1024
        query_data_file_sizes = {x: x_data_file_size for x in queries.keys()}
        query_data_file_import_times = {x: x_data_file_import_time for x in queries.keys()}

        queries_parameters = {
            "x": x,
            "queries_results": queries_results,
            "extended_functionality": extended_functionality,
        }

    if join_queries_files_number:
        data_name = next(
            (f for f in data_for_join_queries if "NA" in f), None
        )  # gets the file name with "NA" component
        data_files_paths, data_files_sizes = join_to_tbls(data_name)

        data_files_import_times = {}
        data_df = {}
        print(f"loading dataset {[path for path in data_files_paths.values()]}")
        for data_id, data_path in data_files_paths.items():
            t0 = timer()
            data_df[data_id] = pd.read_csv(data_path)
            data_files_import_times[data_id] = timer() - t0

        print(len(data_df["x"].index), flush=True)
        print(len(data_df["small"].index), flush=True)
        print(len(data_df["medium"].index), flush=True)
        print(len(data_df["big"].index), flush=True)
        queries = {
            "join_query1": join_query1_modin,
            "join_query2": join_query2_modin,
            "join_query3": join_query3_modin,
            "join_query4": join_query4_modin,
            "join_query5": join_query5_modin,
        }
        queries_results = {x: {y: 0.0 for y in queries_results_fields} for x in queries.keys()}
        queries_parameters = {
            "x": data_df["x"],
            "ys": [data_df["small"], data_df["medium"], data_df["big"]],
            "queries_results": queries_results,
            "extended_functionality": extended_functionality,
        }

        query_data_file_sizes = {
            "join_query1": data_files_sizes["x"] + data_files_sizes["small"],
            "join_query2": data_files_sizes["x"] + data_files_sizes["medium"],
            "join_query3": data_files_sizes["x"] + data_files_sizes["medium"],
            "join_query4": data_files_sizes["x"] + data_files_sizes["medium"],
            "join_query5": data_files_sizes["x"] + data_files_sizes["big"],
        }
        query_data_file_import_times = {
            "join_query1": data_files_import_times["x"] + data_files_import_times["small"],
            "join_query2": data_files_import_times["x"] + data_files_import_times["medium"],
            "join_query3": data_files_import_times["x"] + data_files_import_times["medium"],
            "join_query4": data_files_import_times["x"] + data_files_import_times["medium"],
            "join_query5": data_files_import_times["x"] + data_files_import_times["big"],
        }

    for query_name, query_func in queries.items():
        query_func(**queries_parameters)
        print(f"{pandas_mode} {query_name} results:")
        print_results(results=queries_results[query_name], unit="s")
        queries_results[query_name]["Backend"] = pandas_mode
        queries_results[query_name]["t_readcsv"] = query_data_file_import_times[query_name]
        queries_results[query_name]["dataset_size"] = query_data_file_sizes[query_name]

    return queries_results


def run_benchmark(parameters):
    check_support(
        parameters,
        unsupported_params=[
            "dfiles_num",
            "gpu_memory",
            "no_ml",
            "no_ibis",
            "optimizer",
            "validation",
        ],
    )

    parameters["data_file"] = parameters["data_file"].replace("'", "")

    try:
        if not parameters["no_pandas"]:
            import_pandas_into_module_namespace(
                namespace=run_benchmark.__globals__,
                mode=parameters["pandas_mode"],
                ray_tmpdir=parameters["ray_tmpdir"],
                ray_memory=parameters["ray_memory"],
            )
            queries_results = queries_modin(
                filename=parameters["data_file"],
                pandas_mode=parameters["pandas_mode"],
                extended_functionality=parameters["extended_functionality"],
            )

        return {"ETL": [queries_results], "ML": []}
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
