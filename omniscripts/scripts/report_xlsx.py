import argparse
import os
import logging
from contextlib import suppress

import pandas as pd
import pandas.io.formats.excel

from omniscripts.arg_parser import add_sql_arguments, DbConfig

# This is necessary to allow custom header formatting
pandas.io.formats.excel.ExcelFormatter.header_style = None

logger = logging.getLogger(__name__)


def recorgnize_host_cols(df):
    """We parse and recognize columns that are the same across runs."""
    df = df.drop(["params"], axis=1).fillna("None").nunique()
    return list(df[df == 1].index)


def write_benchmark(df, writer, table_name, benchmark_cols):
    """Writes benchmark results to a new sheet `table_name` for a xlsx writer `writer`.

    Rough structure of the sheet
    -------------------------------------------------------------------------
    | pandas_mode                                    | Pandas | Ray  | HDK  |
    -------------------------------------------------------------------------
    | cpu_mghz                                       | ...... | .... | .... | <- Hidden benchmark- specific run params
    | `other run params specific for this benchmark` | ...... | .... | .... | <- Hidden benchmark- specific run params
    | .............................................. | ...... | .... | .... | <- Hidden benchmark- specific run params
    -------------------------------------------------------------------------
    | query1                                         | 120.12 | 12.1 | 10.1 | **Chart1**
    | query2                                         | 150.12 | 13.1 | 11.1 | **Chart2**
    -------------------------------------------------------------------------
    """
    df = df.T

    def add_chart(i, title, loc):
        """Add performance bar chart with title=`title`, for results from
        column `i` and locate the chart with coordinates `loc`"""
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
    # Write benchmark results to a new sheet
    df.to_excel(writer, sheet_name=sheet_name, header=False)
    worksheet = writer.sheets[sheet_name]

    # Format header for 0-row and 0-column
    header_format = writer.book.add_format({"bold": True, "align": "left"})
    worksheet.set_column(0, 0, 20, header_format)
    worksheet.set_row(0, None, header_format)

    # Set column width for columns with benchmark results (1 to N)
    worksheet.set_column(1, len(df.columns), 20)

    # Hide rows with benchmark run configuration
    n_rows_run_props = len(df) - len(benchmark_cols) - 1
    for i in range(n_rows_run_props):
        worksheet.set_row(i + 1, None, None, {"hidden": True})

    # Add bar charts with each query results
    for i, name in enumerate(benchmark_cols):
        add_chart(
            i + n_rows_run_props + 1,
            title=name,
            loc=(i * 20 + n_rows_run_props, len(df.columns) + 1),
        )


def write_hostinfo(df, writer):
    # Write hostinfo to a new sheet
    df.T.to_excel(writer, sheet_name="HostInfo", header=False)
    sheet = writer.sheets["HostInfo"]
    # add formatting
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
        "-agg",
        dest="agg",
        default="median",
        help="Result aggregation type for runs with several iterations. Median by default",
        choices=["mean", "min", "max", "median"],
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
    # xlsxwriter will corrupt file if it already exists, so need to remove it manually
    with suppress(FileNotFoundError):
        os.remove(args.report_path)

    writer = pd.ExcelWriter(args.report_path, engine="xlsxwriter")

    db_config = DbConfig(
        driver=args.db_driver,
        server=args.db_server,
        port=args.db_port,
        user=args.db_user,
        password=args.db_pass,
        name=args.db_name,
    )
    from omniscripts.report import BenchmarkDb

    db = BenchmarkDb(engine=db_config.create_engine())

    iterations = db.load_iterations(node=args.node)
    iterations = iterations.groupby(["benchmark", "pandas_mode"], as_index=False).last()

    benchmark_col = "benchmark"
    backend_col = "pandas_mode"

    iteration_cols = ["id", "iteration_no", "run_id", "date"] + [benchmark_col]
    run_cols = [c for c in iterations.columns if c not in iteration_cols]

    host_params = recorgnize_host_cols(iterations[run_cols])

    for benchmark in iterations[benchmark_col].unique():
        df, measurements = db.load_benchmark_results_agg(
            benchmark=benchmark, node=args.node, agg=args.agg
        )
        df = df.groupby("pandas_mode", as_index=False).last()
        df = df[[backend_col, *(c for c in df.columns if c not in host_params + iteration_cols)]]
        write_benchmark(df, writer=writer, table_name=benchmark, benchmark_cols=measurements)

    host_info = iterations[host_params].fillna("None").drop_duplicates()
    if len(host_info) != 1:
        raise ValueError(
            "Unexpected variability in host info, expected to be the same across all runs, but discovered different results: "
            f'This map should only contain 1-value: "{host_info.nunique()}"'
        )

    write_hostinfo(host_info, writer=writer)
    writer.close()


if __name__ == "__main__":
    main()
