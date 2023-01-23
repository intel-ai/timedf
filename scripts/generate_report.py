import argparse
import os
import datetime
import logging
from contextlib import suppress

import sqlalchemy as sql
import pandas as pd
import pandas.io.formats.excel

# from report.schema import Iteration as Iter, Measurement as M
from report import Db
# import report.schema as schema
from utils_base_env import add_sql_arguments, DbConfig

# This is necessary to allow custom header formatting
pandas.io.formats.excel.ExcelFormatter.header_style = None

logger = logging.getLogger(__name__)


def recorgnize_cols(df):
    """We parse and recognize 2 types of columns:
    1. `shared_params` - host-specific columns, that are identical across all benchmark runs.
    2. `bench_specific_params` - columns, that vary across runs, they will be reported along with benchmark results.
    """
    df = df.drop(['params'], axis=1).fillna("None").nunique()
    return list(df[df == 1].index)
    

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
        add_chart(
            i + n_rows_run_props + 1,
            title=name,
            loc=(i * 20 + n_rows_run_props, len(df.columns) + 1),
        )


def write_hostinfo(df, writer):
    df.T.to_excel(writer, sheet_name="HostInfo", header=False)
    sheet = writer.sheets["HostInfo"]

    cell_format = writer.book.add_format({"bold": True, "align": "left"})
    sheet.set_column(0, 0, 20, cell_format)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report with benchmark results")
    sql = parser.add_argument_group("db")
    add_sql_arguments(sql)
    parser.add_argument(
        "-report_path",
        dest="report_path",
        default="report.xlsx",
        help="Path to the resulting file",
    )
    parser.add_argument(
        "-node",
        dest="node",
        default=None,
        help="Filter benchmark results by node",
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

    db_config = DbConfig(
        driver=args.db_driver,
        server=args.db_server,
        port=args.db_port,
        user=args.db_user,
        password=args.db_pass,
        name=args.db_name,
    )
    db = Db(engine=db_config.create_engine(future=False))

    iterations = db.load_iterations(node=args.node)
    iterations = iterations.groupby(['benchmark', 'pandas_mode'], as_index=False).last()

    benchmark_col = "benchmark"
    backend_cols = ["pandas_mode"]

    iteration_cols = ["id", "iteration_no", "run_id", "date"] + [benchmark_col]
    run_cols = [c for c in iterations.columns if c not in iteration_cols]

    shared_params = recorgnize_cols(iterations[run_cols])

    for benchmark in iterations[benchmark_col].unique():
        df, measurements = db.load_benchmark_results_agg(benchmark=benchmark, node=args.node)
        df = df.groupby('pandas_mode', as_index=False).last()
        write_benchmark(
            df[backend_cols + [c for c in df.columns if c not in [*shared_params, *iteration_cols]]],
            writer=writer, table_name=benchmark, benchmark_cols=measurements
        )

    host_info = iterations[shared_params].fillna("None").drop_duplicates()
    if len(host_info) != 1:
        raise ValueError(
            "Unexpected variability in host info, expected to be the same across all runs, but discovered different results: "
            f'This map should only contain 1-value: "{host_info.nunique()}"'
        )

    write_hostinfo(host_info, writer=writer)
    writer.close()


if __name__ == "__main__":
    main()
