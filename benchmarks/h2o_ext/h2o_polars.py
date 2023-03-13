import polars as pl
from polars import col

from .h2o_utils import H2OBackend


def q1(x):
    ans = x.groupby("id1").agg(pl.sum("v1")).collect()
    # return [ans["v1"].cast(pl.Int64).sum()]


def q2(x):
    ans = x.groupby(["id1", "id2"]).agg(pl.sum("v1")).collect()
    # return [ans["v1"].cast(pl.Int64).sum()]


def q3(x):
    ans = x.groupby("id3").agg([pl.sum("v1"), pl.mean("v3").alias("v3_mean")]).collect()
    # return ans.lazy().select([pl.col("v1").cast(pl.Int64).sum(), pl.col("v3_mean").sum()]).collect().to_numpy()[0]


def q4(x):
    ans = x.groupby("id4").agg([pl.mean("v1"), pl.mean("v2"), pl.mean("v3")]).collect()
    # return ans.lazy().select([pl.col("v1").sum(), pl.col("v2").sum(), pl.col("v3").sum()]).collect().to_numpy()[0]


def q5(x):
    ans = x.groupby("id6").agg([pl.sum("v1"), pl.sum("v2"), pl.sum("v3")]).collect()
    # return ans.lazy().select([pl.col("v1").cast(pl.Int64).sum(), pl.col("v2").cast(pl.Int64).sum(), pl.col("v3").sum()]).collect().to_numpy()[0]


def q6(x):
    ans = (
        x.groupby(["id4", "id5"])
        .agg([pl.median("v3").alias("v3_median"), pl.std("v3").alias("v3_std")])
        .collect()
    )
    # return ans.lazy().select([pl.col("v3_median").sum(), pl.col("v3_std").sum()]).collect().to_numpy()[0]


def q7(x):
    ans = x.groupby("id3").agg([(pl.max("v1") - pl.min("v2")).alias("range_v1_v2")]).collect()
    # return [ans["range_v1_v2"].cast(pl.Int64).sum()]


def q8(x):
    ans = (
        x.drop_nulls("v3")
        .sort("v3", reverse=True)
        .groupby("id6")
        .agg(col("v3").head(2).alias("largest2_v3"))
        .explode("largest2_v3")
        .collect()
    )
    # return [ans["largest2_v3"].sum()]


def q9(x):
    ans = x.groupby(["id2", "id4"]).agg((pl.pearson_corr("v1", "v2") ** 2).alias("r2")).collect()
    # return [ans["r2"].sum()]


def q10(x):
    ans = (
        x.groupby(["id1", "id2", "id3", "id4", "id5", "id6"])
        .agg([pl.sum("v3").alias("v3"), pl.count("v1").alias("count")])
        .collect()
    )
    # return ans.lazy().select([pl.col("v3").sum(), pl.col("count").cast(pl.Int64).sum()]).collect().to_numpy()[0]


name2groupby_query = {
    "q01": q1,
    "q02": q2,
    "q03": q3,
    "q04": q4,
    "q05": q5,
    "q06": q6,
    "q07": q7,
    "q08": q8,
    "q09": q9,
    "q10": q10,
}


def q1(data):
    ans = data["df"].join(data["small"], on="id1")
    # return [ans["v1"].sum(), ans["v2"].sum()]


def q2(data):
    ans = data["df"].join(data["medium"], on="id2")
    # return [ans["v1"].sum(), ans["v2"].sum()]


def q3(data):
    ans = data["df"].join(data["medium"], how="left", on="id2")
    # return [ans["v1"].sum(), ans["v2"].sum()]


def q4(data):
    ans = data["df"].join(data["medium"], on="id5")
    # return [ans["v1"].sum(), ans["v2"].sum()]


def q5(data):
    ans = data["df"].join(data["big"], on="id3")
    # return [ans["v1"].sum(), ans["v2"].sum()]


class H2OBackendImpl(H2OBackend):
    name2groupby_query = name2groupby_query
    name2join_query = {
        "q01": q1,
        "q02": q2,
        "q03": q3,
        "q04": q4,
        "q05": q5,
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
