import polars as pl
from polars import col

from .h2o_utils import H2OBackend


def groupby_q1(x):
    return x.groupby("id1").agg(pl.sum("v1"))


def groupby_q2(x):
    return x.groupby(["id1", "id2"]).agg(pl.sum("v1"))


def groupby_q3(x):
    return x.groupby("id3").agg([pl.sum("v1"), pl.mean("v3").alias("v3_mean")])


def groupby_q4(x):
    return x.groupby("id4").agg([pl.mean("v1"), pl.mean("v2"), pl.mean("v3")])


def groupby_q5(x):
    return x.groupby("id6").agg([pl.sum("v1"), pl.sum("v2"), pl.sum("v3")])


def groupby_q6(x):
    return x.groupby(["id4", "id5"]).agg(
        [pl.median("v3").alias("v3_median"), pl.std("v3").alias("v3_std")]
    )


def groupby_q7(x):
    return x.groupby("id3").agg([(pl.max("v1") - pl.min("v2")).alias("range_v1_v2")])


def groupby_q8(x):
    return (
        x.drop_nulls("v3")
        .sort("v3", reverse=True)
        .groupby("id6")
        .agg(col("v3").head(2).alias("largest2_v3"))
        .explode("largest2_v3")
    )


def groupby_q9(x):
    return x.groupby(["id2", "id4"]).agg((pl.pearson_corr("v1", "v2") ** 2).alias("r2"))


def groupby_q10(x):
    return x.groupby(["id1", "id2", "id3", "id4", "id5", "id6"]).agg(
        [pl.sum("v3").alias("v3"), pl.count("v1").alias("count")]
    )


name2groupby_query = {
    "q01": groupby_q1,
    "q02": groupby_q2,
    "q03": groupby_q3,
    "q04": groupby_q4,
    "q05": groupby_q5,
    "q06": groupby_q6,
    "q07": groupby_q7,
    "q08": groupby_q8,
    "q09": groupby_q9,
    "q10": groupby_q10,
}


def join_q1(data):
    return data["df"].join(data["small"], on="id1")


def join_q2(data):
    return data["df"].join(data["medium"], on="id2")


def join_q3(data):
    return data["df"].join(data["medium"], how="left", on="id2")


def join_q4(data):
    return data["df"].join(data["medium"], on="id5")


def join_q5(data):
    return data["df"].join(data["big"], on="id3")


class H2OBackendImpl(H2OBackend):
    name2groupby_query = name2groupby_query
    name2join_query = {
        "q01": join_q1,
        "q02": join_q2,
        "q03": join_q3,
        "q04": join_q4,
        "q05": join_q5,
    }

    def __init__(self):
        dtypes = {
            **{n: pl.Categorical for n in ["id1", "id2", "id3", "id4", "id5", "id6"]},
            **{n: pl.Float64 for n in ["v1", "v2", "v3"]},
            # "v3":pl.Float64
        }
        super().__init__(dtypes)

    def load_groupby_data(self, paths):
        with pl.StringCache():
            x = pl.read_csv(paths["groupby"], dtypes=self.dtypes["groupby"], low_memory=True)

        return x.lazy()

    def load_join_data(self, paths):
        with pl.StringCache():
            df = pl.read_csv(paths["join_df"], dtypes=self.dtypes["join_df"])
            small = pl.read_csv(paths["join_small"], dtypes=self.dtypes["join_small"])
            medium = pl.read_csv(paths["join_medium"], dtypes=self.dtypes["join_medium"])
            big = pl.read_csv(paths["join_big"], dtypes=self.dtypes["join_big"])
            return {"df": df, "small": small, "medium": medium, "big": big}
