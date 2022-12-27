import argparse
import os
import datetime
import logging
from contextlib import suppress

import sqlalchemy as db
import pandas as pd
import pandas.io.formats.excel

from utils_base_env import add_mysql_arguments

# This is necessary to allow custom header formatting
pandas.io.formats.excel.ExcelFormatter.header_style = None

logger = logging.getLogger(__name__)

# Table names with benchmark results to be processed
tables = [
    "census_etl_jit",
    "census_ml_jit",
    "plasticc_ml_jit",
    "plasticc_etl_jit",
    "taxibench_etl_jit",
]

# Columns in SQL table that contain host-specific information that is expected to be the same for every run, this info will be placed in a separate sheet.
host_cols = [
    "ServerName",
    "Architecture",
    "Machine",
    "Node",
    "OS",
    "CPUCount",
    "CPUModel",
    "CPUMaxMHz",
    "L1dCache",
    "L1iCache",
    "L2Cache",
    "L3Cache",
    "MemTotal",
    "SwapTotal",
    "SwapFree",
    "HugePages_Total",
    "HugePages_Free",
    "Hugepagesize",
    "OmnisciCommitHash",
    "OmniscriptsCommitHash",
    "ModinCommitHash",
    "IbisCommitHash",
]

# Columns in SQL tables that are run-specific, part of them will be compressed (`iteration`) and some need to be presented along benchmark results (`MemFree`)
possible_run_cols = [
    # will be removed
    "id",
    "run_id",
    "date",
    "Iteration",
    # will become a header
    "BackEnd",
    # will be presended along with benchmark results in a hidden form
    "CPUMHz",
    "MemFree",
    "MemAvailable",
    "dataset_size",
    "dfiles_num",
]


class DBLoader:
    def __init__(self, driver, server, user, password, port, name):
        self.engine = db.create_engine(f"{driver}://{user}:{password}@{server}:{port}/{name}")

    def load_latest_results(self, table_name, past_lookup_days=30):
        metadata = db.MetaData()
        table = db.Table(table_name, metadata, autoload=True, autoload_with=self.engine)

        lookup_date = datetime.date.today() - datetime.timedelta(days=past_lookup_days)
        latest = db.select(db.func.max(table.columns.run_id)).group_by(table.columns.BackEnd)
        qry = db.select([table]).filter(
            db.and_(table.columns.run_id.in_(latest), table.columns.date > lookup_date)
        )

        return pd.read_sql(
            qry,
            con=self.engine,
            parse_dates=[
                "date",
            ],
        )


def recognize_cols(df):
    """We parse and recognize 3 types of columns:
    1. `run_cols` - columns, specific for each run, part of them will be compressed (`iteration`) and some need to be presented along benchmark results (`MemFree`)
    2. `host_cols` - host-specific columns, that are supposedly identical across all benchmark runs.
    3. `benchmark_cols` - columns with benchmark time in seconds."""
    columns = list(df.columns)
    run_cols = [c for c in possible_run_cols if c in columns]
    benchmark_cols = [c for c in columns if c not in host_cols and c not in run_cols]
    return run_cols, host_cols, benchmark_cols


def prepare_benchmark_results(df, benchmark_cols, run_cols):
    return (
        df.groupby("BackEnd", as_index=False)
        .agg({**{c: "first" for c in run_cols}, **{c: "min" for c in benchmark_cols}})
        .drop(["Iteration", "run_id", "id", "date"], axis=1)
    )


def write_benchmark(df, writer, table_name, benchmark_cols):
    df = df.T

    def add_chart(i, title, loc):
        # Performance chart
        chart1 = workbook.add_chart({"type": "bar"})
        chart1.add_series(
            {
                "name": [table_name, i, 0],
                "categories": [table_name, 0, 1, 0, len(df.columns)],
                "values": [table_name, i, 1, i, len(df.columns)],
            }
        )

        chart1.set_title({"name": f"Query: {title}"})
        chart1.set_x_axis({"name": "Time, s"})
        chart1.set_y_axis({"name": "Task"})

        # Set an Excel chart style.
        chart1.set_style(2)

        # Insert the chart into the worksheet (with an offset).
        worksheet.insert_chart(loc[0], loc[1], chart1, {"x_offset": 25, "y_offset": 10})

    sheet_name = f"{table_name}"

    workbook = writer.book
    df.to_excel(writer, sheet_name=sheet_name, header=False)
    worksheet = writer.sheets[sheet_name]

    header_format = writer.book.add_format({"bold": True, "align": "left"})
    worksheet.set_column(0, 0, 20, header_format)
    worksheet.set_row(0, None, header_format)

    worksheet.set_column(1, len(df.columns), 20)

    n_rows_run_props = len(df) - len(benchmark_cols) - 1
    # Hide benchmark configuration
    for i in range(n_rows_run_props):
        worksheet.set_row(i + 1, None, None, {"hidden": True})

    for i, name in enumerate(benchmark_cols):
        add_chart(i + n_rows_run_props + 1, title=name, loc=(i * 20, len(df.columns) + 1))


def write_hostinfo(df, writer):
    df.T.to_excel(writer, sheet_name="HostInfo", header=False)
    sheet = writer.sheets["HostInfo"]

    cell_format = writer.book.add_format({"bold": True, "align": "left"})
    sheet.set_column(0, 0, 20, cell_format)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report with benchmark results")
    db = parser.add_argument_group("db")
    add_mysql_arguments(db, etl_ml_tables=False)
    db.add_argument(
        "-db_driver",
        dest="db_driver",
        help="DB driver",
    )

    parser.add_argument(
        "-report_path",
        dest="report_path",
        default="report.xlsx",
        help="Path to the resulting file",
    )
    return parser.parse_args()


def main():
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

    args = parse_args()
    # xlsxwriter will corrupt file if it already exists, so remove it manually
    with suppress(FileNotFoundError):
        os.unlink(args.report_path)

    writer = pd.ExcelWriter(args.report_path, engine="xlsxwriter")

    loader = DBLoader(
        driver=args.db_driver,
        server=args.db_server,
        user=args.db_user,
        password=args.db_pass,
        port=args.db_port,
        name=args.db_name,
    )

    host_params = []
    for table_name in tables:
        logger.info("Processing %s", table_name)
        df = loader.load_latest_results(table_name=table_name)
        run_cols, host_cols, benchmark_cols = recognize_cols(df)
        df[benchmark_cols] = df[benchmark_cols] / 1000
        host_params.append(df[host_cols])

        benchmark_results = prepare_benchmark_results(
            df[run_cols + benchmark_cols], benchmark_cols, run_cols
        )
        write_benchmark(
            benchmark_results, writer=writer, table_name=table_name, benchmark_cols=benchmark_cols
        )

    host_info = pd.concat(host_params).fillna("None").drop_duplicates()
    if len(host_info) != 1:
        raise ValueError(
            "Unexpected variability in host info, expected to be the same across all runs, but discovered different results: "
            f'This map should only contain 1-value: "{host_info.nunique()}"'
        )

    write_hostinfo(host_info, writer=writer)
    writer.close()


if __name__ == "__main__":
    main()
