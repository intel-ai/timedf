import os
from collections import OrderedDict
from timeit import default_timer as timer
from pathlib import Path
from typing import Any, Iterable, Tuple, Union, Dict

import numpy
import pandas

from utils import (
    check_support,
    import_pandas_into_module_namespace,
    load_data_modin_on_omnisci,
    print_results,
)


class Config:
    MODIN_IMPL = os.getenv("MODIN_IMPL")
    MODIN_STORAGE_FORMAT = os.getenv("MODIN_STORAGE_FORMAT")
    MODIN_ENGINE = os.getenv("MODIN_ENGINE")

    @staticmethod
    def init():
        Config.MODIN_IMPL = os.getenv("MODIN_IMPL")
        Config.MODIN_STORAGE_FORMAT = os.getenv("MODIN_STORAGE_FORMAT")
        Config.MODIN_ENGINE = os.getenv("MODIN_ENGINE")


def get_pd():
    return run_benchmark.__globals__["pd"]


def trigger_import(*dfs):
    """
    Trigger import execution for DataFrames obtained by HDK engine.
    Parameters
    ----------
    *dfs : iterable
        DataFrames to trigger import.
    """
    if Config.MODIN_STORAGE_FORMAT != "hdk" or Config.MODIN_IMPL == "pandas":
        return

    from modin.experimental.core.execution.native.implementations.hdk_on_native.db_worker import (
        DbWorker,
    )

    for df in dfs:
        df.shape  # to trigger real execution
        df._query_compiler._modin_frame._partitions[0][0].frame_id = DbWorker().import_arrow_table(
            df._query_compiler._modin_frame._partitions[0][0].get()
        )  # to trigger real execution


def execute(df: pandas.DataFrame, trigger_hdk_import: bool = False):
    """
    Make sure the calculations are finished.
    Parameters
    ----------
    df : modin.pandas.DataFrame or pandas.Datarame
        DataFrame to be executed.
    trigger_hdk_import : bool, default: False
        Whether `df` are obtained by import with HDK engine.
    """
    if trigger_hdk_import:
        trigger_import(df)
        return

    if Config.MODIN_IMPL == "modin":
        if Config.MODIN_STORAGE_FORMAT == "hdk":
            df._query_compiler._modin_frame._execute()
            return

        partitions = df._query_compiler._modin_frame._partitions.flatten()
        if len(partitions) > 0 and hasattr(partitions[0], "wait"):
            all(map(lambda partition: partition.wait(), partitions))
            return

        # compatibility with old Modin versions
        all(map(lambda partition: partition.drain_call_queue() or True, partitions))
        if Config.MODIN_ENGINE == "ray":
            from ray import wait

            all(map(lambda partition: wait([partition._data]), partitions))
        elif Config.MODIN_ENGINE == "dask":
            from dask.distributed import wait

            all(map(lambda partition: wait(partition._data), partitions))
        elif Config.MODIN_ENGINE == "python":
            pass

    elif Config.MODIN_IMPL == "pandas":
        pass


def realize(*dfs):
    """Utility function to trigger execution for lazy pd libraries."""
    for df in dfs:
        if not isinstance(df, (pandas.DataFrame, pandas.Series, numpy.ndarray)):
            execute(df)


def measure_time(func):
    # @wraps(func)
    def wrapper(*args, **kwargs) -> Union[float, Tuple[Any, float]]:
        start = timer()
        res = func(*args, **kwargs)
        if res is None:
            return timer() - start
        else:
            return res, timer() - start

    return wrapper


def clean(ddf, keep_cols: Iterable):
    # replace the extraneous spaces in column names and lower the font type
    tmp = {col: col.strip().lower() for col in list(ddf.columns)}
    ddf = ddf.rename(columns=tmp)

    ddf = ddf.rename(
        columns={
            "tpep_pickup_datetime": "pickup_datetime",
            "tpep_dropoff_datetime": "dropoff_datetime",
            "ratecodeid": "rate_code",
        }
    )
    to_drop = ddf.columns.difference(keep_cols)
    if not to_drop.empty:
        ddf = ddf.drop(columns=to_drop)
    to_fillna = [col for dt, col in zip(ddf.dtypes, ddf.dtypes.index) if dt == "object"]
    if to_fillna:
        ddf[to_fillna] = ddf[to_fillna].fillna("-1")
    return ddf


def read_csv(filepath: Path, *, parse_dates=[], col2dtype: OrderedDict, is_omniscidb_mode: bool):
    pd = get_pd()

    columns_names = list(col2dtype)
    columns_types = [col2dtype[c] for c in columns_names]
    is_gz = ".gz" in filepath.suffixes

    if is_omniscidb_mode:
        if is_gz:
            raise NotImplementedError(
                "Modin_on_omnisci mode doesn't support import of compressed files yet"
            )

        df = load_data_modin_on_omnisci(
            filename=filepath,
            columns_names=columns_names,
            columns_types=columns_types,
            parse_dates=parse_dates,
            skiprows=1,
            pd=pd,
        )
    else:
        df = pd.read_csv(
            filepath,
            dtype=col2dtype,
            parse_dates=parse_dates,
            # use_gzip=is_gz,
        )
    return df


@measure_time
def load_data(dirpath: str, is_omniscidb_mode):
    dirpath: Path = Path(dirpath.strip("'\""))
    data_types_2014 = OrderedDict(
        [
            (" tolls_amount", "float64"),
            (" surcharge", "float64"),
            (" store_and_fwd_flag", "object"),
            (" tip_amount", "float64"),
            ("tolls_amount", "float64"),
        ]
    )

    data_types_2015 = OrderedDict([("extra", "float64"), ("tolls_amount", "float64")])

    data_types_2016 = OrderedDict([("tip_amount", "float64"), ("tolls_amount", "float64")])

    # Dictionary of required columns and their datatypes
    # Convert to list just to be clear that we only need column names, but keep types just in case
    keep_cols = list(
        {
            "pickup_datetime": "datetime64[s]",
            "dropoff_datetime": "datetime64[s]",
            "passenger_count": "int32",
            "trip_distance": "float32",
            "pickup_longitude": "float32",
            "pickup_latitude": "float32",
            "rate_code": "int32",
            "dropoff_longitude": "float32",
            "dropoff_latitude": "float32",
            "fare_amount": "float32",
        }
    )

    dfs = []
    for name, dtypes, date_cols in (
        ("2014", data_types_2014, [" pickup_datetime", " dropoff_datetime"]),
        ("2015", data_types_2015, ["tpep_pickup_datetime", "tpep_dropoff_datetime"]),
        ("2016", data_types_2016, ["tpep_pickup_datetime", "tpep_dropoff_datetime"]),
    ):
        dfs.extend(
            [
                clean(
                    read_csv(
                        dirpath / filename,
                        parse_dates=date_cols,
                        col2dtype=dtypes,
                        is_omniscidb_mode=is_omniscidb_mode,
                    ),
                    keep_cols,
                )
                for filename in list((dirpath / name).iterdir())
                # for filename in list((dirpath / name).iterdir())[:2]
            ]
        )

    # concatenate multiple DataFrames into one bigger one
    pd = get_pd()
    df = pd.concat(dfs, ignore_index=True)

    # To trigger execution
    realize(df)

    return df


@measure_time
def filter_df(df):
    # apply a list of filter conditions to throw out records with missing or outlier values
    df = df.query(
        "(fare_amount > 1) & \
        (fare_amount < 500) & \
        (passenger_count > 0) & \
        (passenger_count < 6) & \
        (pickup_longitude > -75) & \
        (pickup_longitude < -73) & \
        (dropoff_longitude > -75) & \
        (dropoff_longitude < -73) & \
        (pickup_latitude > 40) & \
        (pickup_latitude < 42) & \
        (dropoff_latitude > 40) & \
        (dropoff_latitude < 42) & \
        (trip_distance > 0) & \
        (trip_distance < 500) & \
        ((trip_distance <= 50) | (fare_amount >= 50)) & \
        ((trip_distance >= 10) | (fare_amount <= 300)) & \
        (dropoff_datetime > pickup_datetime)"
    )

    return df.reset_index(drop=True)


@measure_time
def feature_engineering(df):
    #################################
    # Adding Interesting Features ###
    #################################
    # add features
    df["day"] = df["pickup_datetime"].dt.day

    # calculate the time difference between dropoff and pickup.
    df["diff"] = df["dropoff_datetime"].astype("int64") - df["pickup_datetime"].astype("int64")

    cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    df[[c + "_r" for c in cols]] = df[cols] // (0.01 * 0.01)

    df = df.drop(["pickup_datetime", "dropoff_datetime"], axis=1)

    dlon = df["dropoff_longitude"] - df["pickup_longitude"]
    dlat = df["dropoff_latitude"] - df["pickup_latitude"]
    df["e_distance"] = dlon * dlon + dlat * dlat

    realize(df)

    return df


@measure_time
def split(df):
    #######################
    # Pick a Training Set #
    #######################

    # since we calculated the h_distance let's drop the trip_distance column, and then do model training with XGB.
    df = df.drop("trip_distance", axis=1)

    # this is the original data partition for train and test sets.
    x_train = df[df.day < 25]

    # create a Y_train ddf with just the target variable
    y_train = x_train[["fare_amount"]]
    # drop the target variable from the training ddf
    x_train = x_train.drop("fare_amount", axis=1)

    ###################
    # Pick a Test Set #
    ###################
    x_test = df[df.day >= 25]

    # Create Y_test with just the fare amount
    y_test = x_test[["fare_amount"]]

    # Drop the fare amount from X_test
    x_test = x_test.drop("fare_amount", axis=1)

    realize(x_train, y_train, x_test, y_test)

    return {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}


@measure_time
def train(data: dict, use_modin_xgb: bool):

    if use_modin_xgb:
        import modin.experimental.xgboost as xgb

        # import modin.pandas as pd

        # FIXME: why is that?
        # X_train = pd.DataFrame(X_train)
        # y_train = pd.Series(y_train)
        # X_test = pd.DataFrame(X_test)
        # y_test = pd.Series(y_test)
    else:
        import xgboost as xgb

    dtrain = xgb.DMatrix(data["x_train"], data["y_train"])

    trained_model = xgb.train(
        {
            "learning_rate": 0.3,
            "max_depth": 8,
            "objective": "reg:squarederror",
            "subsample": 0.6,
            "gamma": 1,
            "silent": True,
            "verbose_eval": True,
            "tree_method": "hist",
        },
        dtrain,
        num_boost_round=100,
        # num_boost_round=10,
        evals=[(dtrain, "train")],
    )

    # generate predictions on the test set
    booster = trained_model
    prediction = booster.predict(xgb.DMatrix(data["x_test"]))

    # FIXME: returns an error with Pandas, because array only have 1 axis
    # prediction = prediction.squeeze(axis=1)

    # prediction = pd.Series(booster.predict(xgb.DMatrix(X_test)))

    actual = data["y_test"]["fare_amount"].reset_index(drop=True)
    realize(actual, prediction)
    return None


def compute_skip_rows(gpu_memory):
    # count rows inside test_set.csv
    test_rows = 453653104

    overhead = 1.2
    skip_rows = int((1 - gpu_memory / (32.0 * overhead)) * test_rows)
    return skip_rows


def run_benchmark(parameters):
    # Update config in case some envs changed after the import
    Config.init()
    # FIXME: what is that??
    check_support(parameters, unsupported_params=["optimizer", "dfiles_num"])

    # parameters["data_path"] = parameters["data_file"]
    parameters["gpu_memory"] = parameters["gpu_memory"] or 16
    parameters["no_ml"] = parameters["no_ml"] or False

    # FIXME: do we need this?
    # skip_rows = compute_skip_rows(parameters["gpu_memory"])

    import_pandas_into_module_namespace(
        namespace=run_benchmark.__globals__,
        mode=parameters["pandas_mode"],
        ray_tmpdir=parameters["ray_tmpdir"],
        ray_memory=parameters["ray_memory"],
    )

    benchmark2time = {}

    is_omniscidb_mode = parameters["pandas_mode"] == "Modin_on_omnisci"
    df, benchmark2time["load_data"] = load_data(
        parameters["data_file"], is_omniscidb_mode=is_omniscidb_mode
    )
    df, benchmark2time["filter_df"] = filter_df(df)
    df, benchmark2time["feature_engineering"] = feature_engineering(df)
    print_results(results=benchmark2time, backend=parameters["pandas_mode"], unit="s")

    # benchmark2time["Backend"] = parameters["pandas_mode"]

    backend_name = parameters["pandas_mode"]
    if not parameters["no_ml"]:
        print("using ml with dataframes from Pandas")

        data, benchmark2time["split_time"] = split(df)
        data: Dict[str, Any]

        benchmark2time["train_time"] = train(data, use_modin_xgb=parameters["use_modin_xgb"])

        print_results(results=benchmark2time, backend=parameters["pandas_mode"], unit="s")

        if parameters["use_modin_xgb"]:
            backend_name = backend_name + "_modin_xgb"

    benchmark2time["Backend"] = backend_name

    return {"ETL": [benchmark2time], "ML": [None]}