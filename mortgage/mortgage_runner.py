import warnings

import ibis

from utils import (
    compare_dataframes,
    import_pandas_into_module_namespace,
    print_results,
)
from .mortgage_ibis import etl_ibis
from .mortgage_pandas import etl_pandas, ml

warnings.filterwarnings("ignore")


# Dataset link
# https://rapidsai.github.io/demos/datasets/mortgage-data


def _etl_ibis(parameters, acq_schema, perf_schema, etl_keys, do_validate=False):
    return etl_ibis(
        dataset_path=parameters["data_file"],
        dfiles_num=parameters["dfiles_num"],
        acq_schema=acq_schema,
        perf_schema=perf_schema,
        database_name=parameters["database_name"],
        table_prefix=parameters["table"],
        omnisci_server_worker=parameters["omnisci_server_worker"],
        delete_old_database=not parameters["dnd"] and not do_validate,
        create_new_table=not parameters["dni"] and not do_validate,
        ipc_connection=parameters["ipc_connection"],
        etl_keys=etl_keys,
        import_mode=parameters["import_mode"],
        fragments_size=parameters["fragments_size"],
        leave_category_strings=do_validate,
    )


def _etl_pandas(parameters, acq_schema, perf_schema, etl_keys, do_validate=False):
    return etl_pandas(
        dataset_path=parameters["data_file"],
        dfiles_num=parameters["dfiles_num"],
        acq_schema=acq_schema,
        perf_schema=perf_schema,
        etl_keys=etl_keys,
        leave_category_strings=do_validate,
    )


def _run_ml(df, n_runs, mb, ml_keys, ml_score_keys, backend):
    ml_scores, ml_times = ml(
        df=df, n_runs=n_runs, mb=mb, ml_keys=ml_keys, ml_score_keys=ml_score_keys
    )
    print_results(results=ml_times, backend=backend, unit="s")
    ml_times["Backend"] = backend
    print_results(results=ml_scores, backend=backend)
    ml_scores["Backend"] = backend
    return ml_times


def run_benchmark(parameters):
    parameters["data_file"] = parameters["data_file"].replace("'", "")
    ignored_parameters = {"gpu_memory": parameters["gpu_memory"]}
    warnings.warn(f"Parameters {ignored_parameters} are irnored", RuntimeWarning)
    if parameters["validation"]:
        print("WARNING: Validation not yet supported")
    if not parameters["no_ibis"]:
        if parameters["import_mode"] not in ("fsi",):
            raise ValueError("Unsupported import mode: %s" % parameters["import_mode"])
        if parameters["dfiles_num"] != 1:
            raise NotImplementedError("Loading more than 1 file not implemented yet")

    if not parameters["no_pandas"]:
        import_pandas_into_module_namespace(
            namespace=[run_benchmark.__globals__, etl_pandas.__globals__],
            mode=parameters["pandas_mode"],
            ray_tmpdir=parameters["ray_tmpdir"],
            ray_memory=parameters["ray_memory"],
        )

    acq_schema = ibis.Schema(
        names=(
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
            "year_quarter_ignore",
        ),
        types=(
            "int64",
            "category",
            "string",
            "float64",
            "int64",
            "int64",
            "timestamp",
            "timestamp",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "category",
            "category",
            "category",
            "int64",
            "category",
            "category",
            "int64",
            "float64",
            "category",
            "float64",
            "float64",
            "category",
            "int32",
        ),
    )
    perf_schema = ibis.Schema(
        names=(
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
        ),
        types=(
            "int64",
            "timestamp",
            "category",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "timestamp",
            "float64",
            "int32",
            "category",
            "category",
            "timestamp",
            "timestamp",
            "timestamp",
            "timestamp",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "category",
            "float64",
            "category",
        ),
    )

    etl_keys = ["t_readcsv", "t_etl", "t_connect"]
    ml_keys = ["t_dmatrix", "t_ml", "t_train"]
    ml_score_keys = ["mse_mean", "cod_mean", "mse_dev", "cod_dev"]
    N_RUNS = 1

    result = {"ETL": [], "ML": []}

    if not parameters["no_ibis"]:
        df_ibis, mb_ibis, etl_times_ibis = _etl_ibis(parameters, acq_schema, perf_schema, etl_keys)
        print_results(results=etl_times_ibis, backend="Ibis", unit="s")
        etl_times_ibis["Backend"] = "Ibis"
        result["ETL"].append(etl_times_ibis)
        if not parameters["no_ml"]:
            result["ML"].append(_run_ml(df_ibis, N_RUNS, mb_ibis, ml_keys, ml_score_keys, "Ibis"))

    if not parameters["no_pandas"]:
        df_pd, mb_pd, etl_times_pd = _etl_pandas(parameters, acq_schema, perf_schema, etl_keys)
        print_results(results=etl_times_pd, backend=parameters["pandas_mode"], unit="s")
        etl_times_pd["Backend"] = parameters["pandas_mode"]
        result["ETL"].append(etl_times_pd)

        if not parameters["no_ml"]:
            result["ML"].append(
                _run_ml(df_pd, N_RUNS, mb_pd, ml_keys, ml_score_keys, parameters["pandas_mode"])
            )

    if parameters["validation"]:
        # recompute frames but leave categories as strings
        idf, _, _ = _etl_ibis(parameters, acq_schema, perf_schema, etl_keys, do_validate=True)
        pdf, _, _ = _etl_pandas(parameters, acq_schema, perf_schema, etl_keys, do_validate=True)
        for df in (pdf, idf):
            for colname, coltype in df.dtypes.items():
                if str(coltype) == "category":
                    df[colname] = (
                        df[colname]
                        .cat.reorder_categories(sorted(df[colname].cat.categories), True)
                        .cat.add_categories("N/A")
                        .fillna("N/A")
                    )
        sortBy = sorted(pdf.dtypes.index)
        pdf.sort_values(by=sortBy, axis=0, inplace=True)
        idf.sort_values(by=sortBy, axis=0, inplace=True)
        pdf = pdf.reset_index().drop("index", axis=1)
        idf = idf.reset_index().drop("index", axis=1)

        compare_dataframes((idf,), (pdf,), [], [])
    # pdf['servicer'] = pdf['servicer'].cat.add_categories('N/A').fillna('N/A')

    #        pdb.set_trace()
    #        # df_pd.drop(dropCols, axis=1, inplace=True)
    #        compare_dataframes(
    #            ibis_dfs=(df_ibis,), pandas_dfs=(df_pd,), sort_cols=sortBy, drop_cols=dropCols
    #        )

    return result
