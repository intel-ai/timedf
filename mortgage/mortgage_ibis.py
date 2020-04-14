#!/usr/bin/env python3
import sys
import time
from collections import OrderedDict
from timeit import default_timer as timer

import numpy as np
import ibis

from .mortgage_pandas import MortgagePandasBenchmark  # used for loading
from utils import check_fragments_size


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *a, **kw):
        print(f"[mortgage-timers] {self.name} took {timer() - self.start} sec")


# ------------------------------------------------------------------------------------------
def create_joined_df(perf_table):
    delinquency_12_expr = (
        ibis.case()
        .when(
            perf_table["current_loan_delinquency_status"].notnull(),
            perf_table["current_loan_delinquency_status"],
        )
        .else_(-1)
        .end()
    )
    upb_12_expr = (
        ibis.case()
        .when(perf_table["current_actual_upb"].notnull(), perf_table["current_actual_upb"])
        .else_(999999999)
        .end()
    )
    joined_df = perf_table[
        "loan_id",
        perf_table["monthly_reporting_period"].month().name("timestamp_month").cast("int32"),
        perf_table["monthly_reporting_period"].year().name("timestamp_year").cast("int32"),
        delinquency_12_expr.name("delinquency_12"),
        upb_12_expr.name("upb_12"),
    ]
    return joined_df


def create_12_mon_features(joined_df):
    delinq_df = None
    n_months = 12  # should be 12 but we don't have UNION yet :(
    for y in range(1, n_months + 1):
        year_dec = (
            ibis.case().when(joined_df["timestamp_month"] < ibis.literal(y), 1).else_(0).end()
        )
        tmp_df = joined_df[
            "loan_id",
            "delinquency_12",
            "upb_12",
            (joined_df["timestamp_year"] - year_dec).name("timestamp_year"),
        ]

        delinquency_12 = (tmp_df["delinquency_12"].max() > 3).cast("int32") + (
            tmp_df["upb_12"].min() == 0
        ).cast("int32")
        tmp_df = tmp_df.groupby(["loan_id", "timestamp_year"]).aggregate(
            delinquency_12.name("delinquency_12")
        )

        tmp_df = tmp_df.mutate(timestamp_month=ibis.literal(y, "int32"))

        if delinq_df is None:
            delinq_df = tmp_df
        else:
            delinq_df = delinq_df.union(tmp_df)

    return delinq_df


def final_performance_delinquency(perf_table, mon12_df):
    # rename columns, or join fails because it has overlapping keys
    return perf_table.left_join(
        mon12_df.relabel({"loan_id": "mon12_loan_id"}),
        [
            ("loan_id", "mon12_loan_id"),
            perf_table["monthly_reporting_period"].month().cast("int32")
            == mon12_df["timestamp_month"],
            perf_table["monthly_reporting_period"].year().cast("int32")
            == mon12_df["timestamp_year"],
        ],
    )[perf_table, mon12_df["delinquency_12"]]


def join_perf_acq_gdfs(perf_df, acq_table):
    merged = perf_df.inner_join(acq_table, ["loan_id"])

    dropList = {
        "loan_id",
        "orig_date",
        "first_pay_date",
        "seller_name",
        "monthly_reporting_period",
        "last_paid_installment_date",
        "maturity_date",
        "ever_30",
        "ever_90",
        "ever_180",
        "delinquency_30",
        "delinquency_90",
        "delinquency_180",
        "upb_12",
        "zero_balance_effective_date",
        "foreclosed_after",
        "disposition_date",
        "timestamp",
    }

    resultCols = []
    for req in (perf_df, acq_table):
        schema = req.schema()
        for colName in schema:
            if colName in dropList:
                continue
            if isinstance(schema[colName], ibis.expr.datatypes.Category):
                resultCols.append(req[colName].cast("int32"))
            else:
                resultCols.append(req[colName])
    return merged[resultCols]


def run_ibis_workflow(acq_table, perf_table):
    with Timer("create ibis queries"):
        joined_df = create_joined_df(perf_table)
        mon12_df = create_12_mon_features(joined_df)

        perf_df = final_performance_delinquency(perf_table, mon12_df)
        final_gdf = join_perf_acq_gdfs(perf_df, acq_table)

    with Timer("ibis compilation"):
        final_gdf.compile()
        final_gdf.materialize()

    with Timer("execute queries"):
        result = final_gdf.execute()
    return result


def etl_ibis(
    dataset_path,
    dfiles_num,
    acq_schema,
    perf_schema,
    database_name,
    table_prefix,
    omnisci_server_worker,
    delete_old_database,
    create_new_table,
    ipc_connection,
    etl_keys,
    import_mode,
    fragments_size,
):
    etl_times = {key: 0.0 for key in etl_keys}

    fragments_size = check_fragments_size(fragments_size, count_table=2, import_mode=import_mode)

    omnisci_server_worker.create_database(database_name, delete_if_exists=delete_old_database)
    mb = MortgagePandasBenchmark(dataset_path, "xgb")  # used for loading

    # Create table and import data
    if create_new_table:
        t0 = timer()
        if import_mode == "copy-from":
            raise ValueError("COPY FROM does not work with Mortgage dataset")
        elif import_mode == "pandas":
            raise NotImplementedError("Loading mortgage for ibis by Pandas not implemented yet")
        elif import_mode == "fsi":
            year, quarter = MortgagePandasBenchmark.split_year_quarter(dfiles_num)
            omnisci_server_worker._conn.create_table_from_csv(
                f"{table_prefix}_acq",
                f"{mb.acq_data_path}/Acquisition_{year}Q{quarter}.txt",
                acq_schema,
                database_name,
                delimiter=",",
                header="true",
                fragment_size=fragments_size[0],
            )
            # FIXME: handle cases when quarter perf is split in two files
            omnisci_server_worker._conn.create_table_from_csv(
                f"{table_prefix}_perf",
                f"{mb.perf_data_path}/Performance_{year}Q{quarter}.txt",
                perf_schema,
                database_name,
                delimiter=",",
                header=True,
                fragment_size=fragments_size[1],
            )
        etl_times["t_readcsv"] = round((timer() - t0) * 1000)

    # Second connection - this is ibis's ipc connection for DML
    omnisci_server_worker.connect_to_server(database_name, ipc=ipc_connection)
    acq_table = omnisci_server_worker.database(database_name).table(f"{table_prefix}_acq")
    perf_table = omnisci_server_worker.database(database_name).table(f"{table_prefix}_perf")

    t_etl_start = timer()
    ibis_df = run_ibis_workflow(acq_table, perf_table)
    etl_times["t_etl"] = round((timer() - t_etl_start) * 1000)

    return ibis_df, mb, etl_times
