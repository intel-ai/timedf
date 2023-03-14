import warnings
import time

from omniscripts.pandas_backend import pd, trigger_execution, backend_cfg


from .h2o_utils import H2OBackend

gb_params = {"as_index": False}


def q1(x):
    ans = x.groupby("id1", **gb_params).agg({"v1": "sum"})
    trigger_execution(ans)
    # return [ans["v1"].sum()]


def q2(x):
    ans = x.groupby(["id1", "id2"], **gb_params).agg({"v1": "sum"})
    trigger_execution(ans)
    # return [ans["v1"].sum()]


def q3(x):
    ans = x.groupby("id3", **gb_params).agg({"v1": "sum", "v3": "mean"})
    trigger_execution(ans)
    # return [ans["v1"].sum(), ans["v3"].sum()]


def q4(x):
    ans = x.groupby("id4", **gb_params).agg({"v1": "mean", "v2": "mean", "v3": "mean"})
    trigger_execution(ans)


def q5(x):
    ans = x.groupby("id6", **gb_params).agg({"v1": "sum", "v2": "sum", "v3": "sum"})
    trigger_execution(ans)
    # return [ans["v1"].sum(), ans["v2"].sum(), ans["v3"].sum()]


def q6(x):
    ans = x.groupby(["id4", "id5"], **gb_params).agg({"v3": ["median", "std"]})
    trigger_execution(ans)
    # return [ans["v3"]["median"].sum(), ans["v3"]["std"].sum()]


def q7(x):
    ans = (
        x.groupby("id3", **gb_params)
        .agg({"v1": "max", "v2": "min"})
        .assign(range_v1_v2=lambda x: x["v1"] - x["v2"])[["id3", "range_v1_v2"]]
    )
    trigger_execution(ans)
    # return [ans["range_v1_v2"].sum()]


def q8(x):
    ans = (
        x[~x["v3"].isna()][["id6", "v3"]]
        .sort_values("v3", ascending=False)
        .groupby("id6", **gb_params)
        .head(2)
    )
    trigger_execution(ans)
    # return [ans["v3"].sum()]


def q9(x):
    ans = (
        x[["id2", "id4", "v1", "v2"]]
        .groupby(["id2", "id4"], **gb_params)
        .apply(lambda x: pd.Series({"r2": x.corr()["v1"]["v2"] ** 2}))
    )
    trigger_execution(ans)
    # return [ans['r2'].sum()]


def q10(x):
    if backend_cfg["backend"] == "Modin_on_hdk":
        warnings.warn("HDK doesn't support groupby-Q10, waiting 42.42 seconds")
        time.sleep(42.42)
        return

    ans = x.groupby(
        ["id1", "id2", "id3", "id4", "id5", "id6"],
        **gb_params,
    ).agg({"v3": "sum", "v1": "size"})
    trigger_execution(ans)
    # return [ans["v3"].sum(), ans["v1"].sum()]


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
    ans = data["df"].merge(data["small"], on="id1")
    trigger_execution(ans)
    # return [ans["v1"].sum(), ans["v2"].sum()]


def q2(data):
    ans = data["df"].merge(data["medium"], on="id2")
    trigger_execution(ans)
    # return [ans["v1"].sum(), ans["v2"].sum()]


def q3(data):
    ans = data["df"].merge(data["medium"], how="left", on="id2")
    trigger_execution(ans)
    # return [ans["v1"].sum(), ans["v2"].sum()]


def q4(data):
    ans = data["df"].merge(data["medium"], on="id5")
    trigger_execution(ans)
    # return [ans["v1"].sum(), ans["v2"].sum()]


def q5(data):
    ans = data["df"].merge(data["big"], on="id3")
    trigger_execution(ans)
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
            **{n: "string" for n in ["id1", "id2", "id3", "id4", "id5", "id6"]},
            **{n: "float64" for n in ["v1", "v2", "v3"]},
        }
        super().__init__(dtypes)

    def load_groupby_data(self, paths):
        # return pd.read_csv(paths['groupby'], dtype=self.dtypes['groupby'])
        return pd.read_csv(paths["groupby"])

    def load_join_data(self, paths):
        df = pd.read_csv(paths["join_df"], dtype=self.dtypes["groupby"])
        small = pd.read_csv(paths["join_small"], dtype=self.dtypes["join_small"])
        medium = pd.read_csv(paths["join_medium"], dtype=self.dtypes["join_medium"])
        big = pd.read_csv(paths["join_big"], dtype=self.dtypes["join_big"])

        return {"df": df, "small": small, "medium": medium, "big": big}
