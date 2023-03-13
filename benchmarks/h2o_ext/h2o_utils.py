from pathlib import Path

from omniscripts import TimerManager


tm = TimerManager()


def filter_dict(d, names):
    return {key: val for key, val in d.items() if key in names}


class H2OBackend:
    name2groupby_query = None
    name2join_query = None

    def __init__(self, dtypes):
        name2cols = {
            "groupby": ["id1", "id2", "id3", "id4", "id5", "id6", "v1", "v2", "v3"],
            "join_df": ["id1", "id2", "id3", "id4", "id5", "id6", "v1"],
            "join_small": ["id1", "id4", "v2"],
            "join_medium": ["id1", "id2", "id4", "id5", "v2"],
            "join_big": ["id1", "id2", "id3", "id4", "id5", "id6", "v2"],
        }

        self.dtypes = {name: filter_dict(dtypes, cols) for name, cols in name2cols.items()}

    @staticmethod
    def load_groupby_data(paths):
        pass

    @staticmethod
    def load_join_data(paths):
        pass


def get_load_info(data_path):
    data_path = Path(data_path)

    def join_to_tbls(data_name):
        x_n = int(float(data_name.split("_")[1]))
        y_n = ["{:.0e}".format(x_n / e) for e in [1e6, 1e3, 1]]
        y_n = [data_name.replace("NA", y).replace("+0", "") for y in y_n]
        return y_n

    file_name = "J1_1e7_NA_0_0"
    paths = [data_path / f"{f}.csv" for f in [file_name, *join_to_tbls(file_name)]]

    return {
        "groupby": data_path / "G1_1e7_1e2_0_0.csv",
        "join_df": paths[0],
        "join_small": paths[1],
        "join_medium": paths[2],
        "join_big": paths[3],
    }
    # Hotfix for modin Path reading error
    # return {name: str(p) for name, p in paths.items()}
