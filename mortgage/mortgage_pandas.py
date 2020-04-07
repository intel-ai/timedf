import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path
from timeit import default_timer

import numpy as np


class MortgagePandasBenchmark:
    def __init__(self, mortgage_path, algo):
        self.acq_data_path = mortgage_path + "/acq"
        self.perf_data_path = mortgage_path + "/perf"
        self.col_names_path = mortgage_path + "/names.csv"

        self.t_one_hot_encoding = 0
        self.t_read_csv = 0
        self.t_fillna = 0
        self.t_drop_cols = 0
        self.t_merge = 0
        self.t_conv_dates = 0
        self.t_train = 0.0
        self.t_dmatrix = 0.0

        self.score_mse = 0.0
        self.score_cod = 0.0

        ML_FWS = {"xgb": self.train_xgb, "daal": self.train_daal}
        self.ml_func = ML_FWS[algo]

    def null_workaround(self, df, **kwargs):
        for column, data_type in df.dtypes.items():

            t0 = default_timer()
            if str(data_type) == "category":
                df[column] = df[column].cat.codes
            t1 = default_timer()
            self.t_one_hot_encoding += t1 - t0

            t0 = default_timer()
            if str(data_type) in [
                "int8",
                "int16",
                "int32",
                "int64",
                "float32",
                "float64",
            ]:
                df[column] = df[column].fillna(np.dtype(data_type).type(-1))
            t1 = default_timer()
            self.t_fillna += t1 - t0

        return df

    def run_cpu_workflow(self, quarter=1, year=2000, perf_file="", **kwargs):
        names = self.pd_load_names()
        acq_gdf = self.cpu_load_acquisition_csv(
            acquisition_path=self.acq_data_path
            + "/Acquisition_"
            + str(year)
            + "Q"
            + str(quarter)
            + ".txt"
        )
        t0 = default_timer()
        acq_gdf = acq_gdf.merge(names, how="left", on=["seller_name"])
        t1 = default_timer()
        self.t_merge += t1 - t0

        t0 = default_timer()
        acq_gdf = acq_gdf.drop(["seller_name"], axis=1)
        acq_gdf["seller_name"] = acq_gdf["new"]
        acq_gdf = acq_gdf.drop(["new"], axis=1)
        t1 = default_timer()
        self.t_drop_cols += t1 - t0

        cdf = self.cpu_load_performance_csv(perf_file)

        joined_df = self.create_joined_df(cdf)
        df_features_12 = self.create_12_mon_features(joined_df)
        combined_df = self.combine_joined_12_mon(joined_df, df_features_12)
        del (df_features_12, joined_df)

        perf_df = self.final_performance_delinquency(cdf, combined_df)
        del (cdf, combined_df)

        final_gdf = self.join_perf_acq_gdfs(perf_df, acq_gdf)
        del perf_df
        del acq_gdf

        final_gdf = self.last_mile_cleaning(final_gdf)
        return final_gdf

    def _parse_dtyped_csv(self, fname, dtypes, **kw):
        all_but_dates = {
            col: valtype for (col, valtype) in dtypes.items() if valtype != "datetime64"
        }
        dates_only = [col for (col, valtype) in dtypes.items() if valtype == "datetime64"]
        t0 = default_timer()
        df = pd.read_csv(fname, dtype=all_but_dates, parse_dates=dates_only, **kw)
        t1 = default_timer()
        self.t_read_csv += t1 - t0
        return df

    def cpu_load_performance_csv(self, performance_path, **kwargs):
        cols = [
            "loan_id",
            "monthly_reporting_period",
            "servicer",
            "interest_rate",
            "current_actual_upb",
            "loan_age",
            "remaining_months_to_legal_maturity",
            "adj_remaining_months_to_maturity",
            "maturity_date",
            "msa",
            "current_loan_delinquency_status",
            "mod_flag",
            "zero_balance_code",
            "zero_balance_effective_date",
            "last_paid_installment_date",
            "foreclosed_after",
            "disposition_date",
            "foreclosure_costs",
            "prop_preservation_and_repair_costs",
            "asset_recovery_costs",
            "misc_holding_expenses",
            "holding_taxes",
            "net_sale_proceeds",
            "credit_enhancement_proceeds",
            "repurchase_make_whole_proceeds",
            "other_foreclosure_proceeds",
            "non_interest_bearing_upb",
            "principal_forgiveness_upb",
            "repurchase_make_whole_proceeds_flag",
            "foreclosure_principal_write_off_amount",
            "servicing_activity_indicator",
        ]

        dtypes = OrderedDict(
            [
                ("loan_id", "int64"),
                ("monthly_reporting_period", "datetime64"),
                ("servicer", "category"),
                ("interest_rate", "float64"),
                ("current_actual_upb", "float64"),
                ("loan_age", "float64"),
                ("remaining_months_to_legal_maturity", "float64"),
                ("adj_remaining_months_to_maturity", "float64"),
                ("maturity_date", "datetime64"),
                ("msa", "float64"),
                ("current_loan_delinquency_status", "int32"),
                ("mod_flag", "category"),
                ("zero_balance_code", "category"),
                ("zero_balance_effective_date", "datetime64"),
                ("last_paid_installment_date", "datetime64"),
                ("foreclosed_after", "datetime64"),
                ("disposition_date", "datetime64"),
                ("foreclosure_costs", "float64"),
                ("prop_preservation_and_repair_costs", "float64"),
                ("asset_recovery_costs", "float64"),
                ("misc_holding_expenses", "float64"),
                ("holding_taxes", "float64"),
                ("net_sale_proceeds", "float64"),
                ("credit_enhancement_proceeds", "float64"),
                ("repurchase_make_whole_proceeds", "float64"),
                ("other_foreclosure_proceeds", "float64"),
                ("non_interest_bearing_upb", "float64"),
                ("principal_forgiveness_upb", "float64"),
                ("repurchase_make_whole_proceeds_flag", "category"),
                ("foreclosure_principal_write_off_amount", "float64"),
                ("servicing_activity_indicator", "category"),
            ]
        )

        print(performance_path)
        return self._parse_dtyped_csv(performance_path, dtypes, names=cols, delimiter="|")

    def cpu_load_acquisition_csv(self, acquisition_path, **kwargs):
        cols = [
            "loan_id",
            "orig_channel",
            "seller_name",
            "orig_interest_rate",
            "orig_upb",
            "orig_loan_term",
            "orig_date",
            "first_pay_date",
            "orig_ltv",
            "orig_cltv",
            "num_borrowers",
            "dti",
            "borrower_credit_score",
            "first_home_buyer",
            "loan_purpose",
            "property_type",
            "num_units",
            "occupancy_status",
            "property_state",
            "zip",
            "mortgage_insurance_percent",
            "product_type",
            "coborrow_credit_score",
            "mortgage_insurance_type",
            "relocation_mortgage_indicator",
        ]
        dtypes = OrderedDict(
            [
                ("loan_id", "int64"),
                ("orig_channel", "category"),
                ("seller_name", "category"),
                ("orig_interest_rate", "float64"),
                ("orig_upb", "int64"),
                ("orig_loan_term", "int64"),
                ("orig_date", "datetime64"),
                ("first_pay_date", "datetime64"),
                ("orig_ltv", "float64"),
                ("orig_cltv", "float64"),
                ("num_borrowers", "float64"),
                ("dti", "float64"),
                ("borrower_credit_score", "float64"),
                ("first_home_buyer", "category"),
                ("loan_purpose", "category"),
                ("property_type", "category"),
                ("num_units", "int64"),
                ("occupancy_status", "category"),
                ("property_state", "category"),
                ("zip", "int64"),
                ("mortgage_insurance_percent", "float64"),
                ("product_type", "category"),
                ("coborrow_credit_score", "float64"),
                ("mortgage_insurance_type", "float64"),
                ("relocation_mortgage_indicator", "category"),
            ]
        )
        print(acquisition_path)
        return self._parse_dtyped_csv(
            acquisition_path, dtypes, names=cols, delimiter="|", index_col=False
        )

    def pd_load_names(self, **kwargs):
        cols = ["seller_name", "new"]
        # dtypes = OrderedDict([
        #     ("seller_name", "category"),
        #     ("new", "category"),
        # ])
        t0 = default_timer()
        df = pd.read_csv(self.col_names_path, names=cols, delimiter="|")
        t1 = default_timer()
        self.t_read_csv += t1 - t0
        return df

    def create_joined_df(self, gdf, **kwargs):
        test = gdf.loc[
            :,
            [
                "loan_id",
                "monthly_reporting_period",
                "current_loan_delinquency_status",
                "current_actual_upb",
            ],
        ]
        del gdf
        test["timestamp"] = test["monthly_reporting_period"]

        t0 = default_timer()
        test = test.drop(["monthly_reporting_period"], axis=1)
        t1 = default_timer()
        self.t_drop_cols += t1 - t0

        t0 = default_timer()
        test["timestamp_month"] = test["timestamp"].dt.month
        test["timestamp_year"] = test["timestamp"].dt.year
        t1 = default_timer()
        self.t_conv_dates += t1 - t0
        test["delinquency_12"] = test["current_loan_delinquency_status"]

        t0 = default_timer()
        test = test.drop(["current_loan_delinquency_status"], axis=1)
        t1 = default_timer()
        self.t_drop_cols += t1 - t0

        test["upb_12"] = test["current_actual_upb"]

        t0 = default_timer()
        test = test.drop(["current_actual_upb"], axis=1)
        t1 = default_timer()
        self.t_drop_cols += t1 - t0

        t0 = default_timer()
        test["upb_12"] = test["upb_12"].fillna(999999999)
        test["delinquency_12"] = test["delinquency_12"].fillna(-1)
        t1 = default_timer()
        self.t_fillna += t1 - t0

        t0 = default_timer()
        test["timestamp_year"] = test["timestamp_year"].astype("int32")
        test["timestamp_month"] = test["timestamp_month"].astype("int32")
        t1 = default_timer()
        self.t_conv_dates += t1 - t0

        return test

    def create_12_mon_features(self, joined_df, **kwargs):
        testdfs = []
        n_months = 12
        for y in range(1, n_months + 1):
            tmpdf = joined_df.loc[
                :, ["loan_id", "timestamp_year", "timestamp_month", "delinquency_12", "upb_12",],
            ]

            t0 = default_timer()
            tmpdf["josh_months"] = tmpdf["timestamp_year"] * 12 + tmpdf["timestamp_month"]
            tmpdf["josh_mody_n"] = np.floor(
                (tmpdf["josh_months"].astype("float64") - 24000 - y) / 12
            )
            tmpdf = tmpdf.groupby(["loan_id", "josh_mody_n"], as_index=False).agg(
                {"delinquency_12": "max", "upb_12": "min"}
            )
            tmpdf["delinquency_12"] = (tmpdf["delinquency_12"] > 3).astype("int32")
            tmpdf["delinquency_12"] += (tmpdf["upb_12"] == 0).astype("int32")
            # tmpdf.drop('max_delinquency_12', axis=1)
            # tmpdf['upb_12'] = tmpdf['min_upb_12']
            # tmpdf.drop('min_upb_12', axis=1)
            tmpdf["timestamp_year"] = np.floor(
                ((tmpdf["josh_mody_n"] * n_months) + 24000 + (y - 1)) / 12
            ).astype("int16")
            tmpdf["timestamp_month"] = np.int8(y)
            t1 = default_timer()
            self.t_conv_dates += t1 - t0

            t0 = default_timer()
            tmpdf = tmpdf.drop(["josh_mody_n"], axis=1)
            t1 = default_timer()
            self.t_drop_cols += t1 - t0

            testdfs.append(tmpdf)
            del tmpdf
        del joined_df

        return pd.concat(testdfs)

    def combine_joined_12_mon(self, joined_df, testdf, **kwargs):
        t0 = default_timer()
        joined_df = joined_df.drop(["delinquency_12"], axis=1)
        joined_df = joined_df.drop(["upb_12"], axis=1)
        t1 = default_timer()
        self.t_drop_cols += t1 - t0

        t0 = default_timer()
        joined_df["timestamp_year"] = joined_df["timestamp_year"].astype("int16")
        joined_df["timestamp_month"] = joined_df["timestamp_month"].astype("int8")
        t1 = default_timer()
        self.t_conv_dates += t1 - t0

        t0 = default_timer()
        df = joined_df.merge(
            testdf, how="left", on=["loan_id", "timestamp_year", "timestamp_month"]
        )
        t1 = default_timer()
        self.t_merge += t1 - t0

        return df

    def final_performance_delinquency(self, gdf, joined_df, **kwargs):
        merged = self.null_workaround(gdf)
        joined_df = self.null_workaround(joined_df)

        t0 = default_timer()
        joined_df["timestamp_month"] = joined_df["timestamp_month"].astype("int8")
        joined_df["timestamp_year"] = joined_df["timestamp_year"].astype("int16")
        merged["timestamp_month"] = merged["monthly_reporting_period"].dt.month
        merged["timestamp_month"] = merged["timestamp_month"].astype("int8")
        merged["timestamp_year"] = merged["monthly_reporting_period"].dt.year
        merged["timestamp_year"] = merged["timestamp_year"].astype("int16")
        t1 = default_timer()
        self.t_conv_dates += t1 - t0

        t0 = default_timer()
        merged = merged.merge(
            joined_df, how="left", on=["loan_id", "timestamp_year", "timestamp_month"]
        )
        t1 = default_timer()
        self.t_merge += t1 - t0

        t0 = default_timer()
        merged = merged.drop(["timestamp_year"], axis=1)
        merged = merged.drop(["timestamp_month"], axis=1)
        t1 = default_timer()
        self.t_drop_cols += t1 - t0

        return merged

    def join_perf_acq_gdfs(self, perf, acq, **kwargs):
        perf = self.null_workaround(perf)
        acq = self.null_workaround(acq)

        t0 = default_timer()
        df = perf.merge(acq, how="left", on=["loan_id"])
        t1 = default_timer()
        self.t_merge += t1 - t0

        return df

    def last_mile_cleaning(self, df, **kwargs):
        drop_list = [
            "loan_id",
            "orig_date",
            "first_pay_date",
            "seller_name",
            "monthly_reporting_period",
            "last_paid_installment_date",
            "maturity_date",
            "upb_12",
            "zero_balance_effective_date",
            "foreclosed_after",
            "disposition_date",
            "timestamp",
        ]

        t0 = default_timer()
        for column in drop_list:
            df = df.drop([column], axis=1)
        t1 = default_timer()
        self.t_drop_cols += t1 - t0

        t0 = default_timer()
        for col, dtype in df.dtypes.iteritems():
            if str(dtype) == "category":
                df[col] = df[col].cat.codes
        t1 = default_timer()
        self.t_one_hot_encoding += t1 - t0

        df["delinquency_12"] = df["delinquency_12"] > 0

        t0 = default_timer()
        df["delinquency_12"] = df["delinquency_12"].fillna(False).astype("int32")
        for column in df.columns:
            df[column] = df[column].fillna(np.dtype(str(df[column].dtype)).type(-1))
        t1 = default_timer()
        self.t_fillna += t1 - t0

        return df

    def train_daal(pd_df):
        import daal4py

        dxgb_daal_params = {
            "fptype": "float",
            "maxIterations": 100,
            "maxTreeDepth": 8,
            "minSplitLoss": 0.1,
            "shrinkage": 0.1,
            "observationsPerTreeFraction": 1,
            "lambda_": 1,
            "minObservationsInLeafNode": 1,
            "maxBins": 256,
            "featuresPerNode": 0,
            "minBinSize": 5,
            "memorySavingMode": False,
        }

        t0 = default_timer()
        y = np.ascontiguousarray(pd_df["delinquency_12"], dtype=np.float32).reshape(len(pd_df), 1)
        x = np.ascontiguousarray(pd_df.drop(["delinquency_12"], axis=1), dtype=np.float32)
        t1 = default_timer()
        self.t_dmatrix = t1 - t0
        # print("Convert x,y from 64 to 32:", t1-t0)

        train_algo = daal4py.gbt_regression_training(**dxgb_daal_params)
        t0 = default_timer()
        train_result = train_algo.compute(x, y)
        self.t_train = default_timer() - t0
        # print("TRAINING TIME:", default_timer()-t0)
        return train_result

    def train_xgb(pd_df):
        import xgboost as xgb

        dxgb_cpu_params = {
            "nthread": 56,
            "nround": 100,
            "alpha": 0.9,
            "max_bin": 256,
            "scale_pos_weight": 2,
            "learning_rate": 0.1,
            "subsample": 1,
            "reg_lambda": 1,
            "min_child_weight": 0,
            "max_depth": 8,
            "max_leaves": 2 ** 8,
            "tree_method": "hist",
            "predictor": "cpu_predictor",
        }

        t0 = default_timer()
        y = pd_df["delinquency_12"]
        x = pd_df.drop(["delinquency_12"], axis=1)
        t1 = default_timer()
        self.t_drop_cols += t1 - t0

        dtrain = xgb.DMatrix(x, y)
        self.t_dmatrix = default_timer() - t1

        t0 = default_timer()
        model_xgb = xgb.train(dxgb_cpu_params, dtrain, num_boost_round=dxgb_cpu_params["nround"])
        self.t_train = default_timer() - t0

        # calculate mse and cod
        x_test = xgb.DMatrix(x)
        y_pred = model_xgb.predict(x_test)
        self.score_mse = mse(y, y_pred)
        self.score_cod = cod(y, y_pred)

        return model_xgb

    def mse(y_test, y_pred):
        return ((y_test - y_pred) ** 2).mean()

    def cod(y_test, y_pred):
        y_bar = y_test.mean()
        total = ((y_test - y_bar) ** 2).sum()
        residuals = ((y_test - y_pred) ** 2).sum()
        return 1 - (residuals / total)


def main():
    # Load database reporting functions
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from report import DbReport
    from utils import import_pandas_into_module_namespace

    parser = argparse.ArgumentParser(description="Run Mortgage benchmark using pandas")

    parser.add_argument("-r", default="report_pandas.csv", help="Report file name.")
    parser.add_argument(
        "-df",
        default=1,
        type=int,
        help="Number of datafiles (quarters) to input into database for processing.",
    )
    parser.add_argument(
        "-dp",
        required=True,
        help="Path to root of mortgage datafiles directory (contains names.csv).",
    )
    parser.add_argument(
        "-i",
        dest="iterations",
        default=5,
        type=int,
        help="Number of iterations to run every benchmark. Best result is selected.",
    )
    parser.add_argument(
        "-algo", choices=["xgb", "daal"], default="xgb", help="xgb : xgb; daal: daal"
    )
    parser.add_argument(
        "-pandas_mode",
        choices=["pandas", "modin_on_ray", "modin_on_dask", "modin_on_python"],
        default="pandas",
        help="Specifies which version of Pandas to use: plain Pandas, Modin runing on Ray or on Dask",
    )
    parser.add_argument(
        "-ray_tmpdir",
        default="/tmp",
        help="Location where to keep Ray plasma store. It should have enough space to keep -ray_memory",
    )
    parser.add_argument(
        "-ray_memory",
        default=200 * 1024 * 1024 * 1024,
        help="Size of memory to allocate for Ray plasma store",
    )
    parser.add_argument(
        "-no_ml", action="store_true", help="Do not run machine learning benchmark, only ETL part",
    )

    parser.add_argument("-db-server", default="localhost", help="Host name of MySQL server")
    parser.add_argument("-db-port", default=3306, type=int, help="Port number of MySQL server")
    parser.add_argument(
        "-db-user",
        default="",
        help="Username to use to connect to MySQL database. If user name is specified, script attempts to store results in MySQL database using other -db-* parameters.",
    )
    parser.add_argument(
        "-db-pass", default="omniscidb", help="Password to use to connect to MySQL database",
    )
    parser.add_argument(
        "-db-name", default="omniscidb", help="MySQL database to use to store benchmark results",
    )
    parser.add_argument("-db-table", help="Table to use to store results for this benchmark.")

    parser.add_argument(
        "-commit",
        default="1234567890123456789012345678901234567890",
        help="Commit hash to use to record this benchmark results",
    )

    args = parser.parse_args()

    if args.df <= 0:
        raise Exception(f"Bad number of data files specified {args.df}")

    if args.iterations < 1:
        raise Exception(f"Bad number of iterations specified {args.iterations}")

    import_pandas_into_module_namespace(
        main.__globals__, args.pandas_mode, args.ray_tmpdir, args.ray_memory
    )

    db_reporter = None
    if args.db_user is not "":
        print("Connecting to database")
        db = mysql.connector.connect(
            host=args.db_server,
            port=args.db_port,
            user=args.db_user,
            passwd=args.db_pass,
            db=args.db_name,
        )
        db_reporter = report.DbReport(
            db,
            args.db_table,
            {
                "FilesNumber": "INT UNSIGNED NOT NULL",
                "FragmentSize": "BIGINT UNSIGNED NOT NULL",
                "BenchName": "VARCHAR(500) NOT NULL",
                "BestExecTimeMS": "BIGINT UNSIGNED",
                "BestTotalTimeMS": "BIGINT UNSIGNED",
                "WorstExecTimeMS": "BIGINT UNSIGNED",
                "WorstTotalTimeMS": "BIGINT UNSIGNED",
                "AverageExecTimeMS": "BIGINT UNSIGNED",
                "AverageTotalTimeMS": "BIGINT UNSIGNED",
            },
            {"ScriptName": "mortgage_pandas.py", "CommitHash": args.commit},
        )

    data_directory = args.dp
    benchName = "mortgage_pandas"

    perf_data_path = os.path.join(data_directory, "perf")
    perf_format_path = os.path.join(perf_data_path, "Performance_%sQ%s.txt")
    bestExecTime = float("inf")
    bestTotalTime = float("inf")
    worstExecTime = 0
    worstTotalTime = 0
    avgExecTime = 0
    avgTotalTime = 0

    for iii in range(1, args.iterations + 1):
        dataFilesNumber = 0
        pd_dfs = []
        time_ETL = default_timer()
        mb = MortgagePandasBenchmark(data_directory, args.algo)
        print("RUNNING BENCHMARK NUMBER", benchName, "ITERATION NUMBER", iii)
        for quarter in range(0, args.df):
            year = 2000 + quarter // 4
            perf_file = perf_format_path % (str(year), str(quarter % 4 + 1))

            files = [
                f
                for f in Path(perf_data_path).iterdir()
                if f.match("Performance_%sQ%s.txt*" % (str(year), str(quarter % 4 + 1)))
            ]
            for f in files:
                pd_dfs.append(
                    mb.run_cpu_workflow(year=year, quarter=(quarter % 4 + 1), perf_file=str(f))
                )
            dataFilesNumber += 1

        time_ETL = default_timer() - time_ETL
        print("ITERATION", iii, "ETL TIME: ", time_ETL)

        if bestExecTime > time_ETL:
            bestExecTime = time_ETL
        if worstExecTime < time_ETL:
            worstExecTime = time_ETL
        avgExecTime += time_ETL
        if bestTotalTime > time_ETL:
            bestTotalTime = time_ETL
        if worstTotalTime < time_ETL:
            worstTotalTime = time_ETL
        avgTotalTime += time_ETL

        print("t_readcsv = ", mb.t_read_csv)
        print("t_ETL = ", time_ETL - mb.t_read_csv)
        print("  t_one_hot_encoding = ", mb.t_one_hot_encoding)
        print("  t_fillna = ", mb.t_fillna)
        print("  t_drop_cols = ", mb.t_drop_cols)
        print("  t_merge = ", mb.t_merge)
        print("  t_conv_dates = ", mb.t_conv_dates)

        if not args.no_ml:
            pd_df = pd.concat(pd_dfs)
            xgb_model = mb.ml_func(pd_df)
            print("t_ML = ", mb.t_dmatrix + mb.t_train)
            print("   t_dmatrix = ", mb.t_dmatrix)
            print("   t_train = ", mb.t_train)
            print("Scores: ")
            print("    mse = ", mb.score_mse)
            print("    cod = ", mb.score_cod)

    avgExecTime /= args.iterations
    avgTotalTime /= args.iterations

    try:
        with open(args.r, "w") as report:
            print(
                "BENCHMARK", benchName, "EXEC TIME", bestExecTime, "TOTAL TIME", bestTotalTime,
            )
            print(
                "datafiles,fragment_size,query,query_exec_min,query_total_min,query_exec_max,query_total_max,query_exec_avg,query_total_avg,query_error_info",
                file=report,
                flush=True,
            )
            print(
                dataFilesNumber,
                ",",
                0,
                ",",
                benchName,
                ",",
                bestExecTime,
                ",",
                bestTotalTime,
                ",",
                worstExecTime,
                ",",
                worstTotalTime,
                ",",
                avgExecTime,
                ",",
                avgTotalTime,
                ",",
                "",
                "\n",
                file=report,
                sep="",
                end="",
                flush=True,
            )
            if db_reporter is not None:
                db_reporter.submit(
                    {
                        "FilesNumber": dataFilesNumber,
                        "FragmentSize": 0,
                        "BenchName": benchName,
                        "BestExecTimeMS": bestExecTime,
                        "BestTotalTimeMS": bestTotalTime,
                        "WorstExecTimeMS": worstExecTime,
                        "WorstTotalTimeMS": worstTotalTime,
                        "AverageExecTimeMS": avgExecTime,
                        "AverageTotalTimeMS": avgTotalTime,
                    }
                )
    except IOError as err:
        print("Failed writing report file", args.r, err)


if __name__ == "__main__":
    main()
