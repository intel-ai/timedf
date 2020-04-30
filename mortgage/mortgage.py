# Derived from https://github.com/fschlimb/scale-out-benchs

import argparse
import os
import pathlib
import sys
import time
from glob import glob  # noqa: F401 (imported, but unused)
from io import StringIO  # noqa: F401 (imported, but unused
import mysql.connector

import numpy as np  # noqa: F401 (imported, but unused. Used in commented code.)
import pandas as pd  # noqa: F401 (imported, but unused. Used in commented code.)
from pandas.api.types import CategoricalDtype  # noqa: F401 (imported, but unused. Used in commented code.)

import report
from pymapd import connect

import_query_template = "COPY %s FROM '%s' WITH (DELIMITER='|');"


def run_pd_workflow(quarter, year, perf_file, fragment_size):
    t1 = time.time()
    # Load names
    con.execute("DROP TABLE IF EXISTS names;")
    pd_load_names(con, fragment_size)
    # Load acquisition
    con.execute("DROP TABLE IF EXISTS acq;")
    acquisition_path = os.path.join(
        data_directory, "acq", "Acquisition_" + str(year) + "Q" + str(quarter) + ".txt"
    )
    pd_load_acquisition_csv(acquisition_path, con, fragment_size)
    # Load perf
    con.execute("DROP TABLE IF EXISTS perf;")
    pd_load_performance_csv(perf_file, con, fragment_size)
    print("read time", (time.time() - t1) * 1000)

    t1 = time.time()
    con.execute("DROP TABLE IF EXISTS acqtemp;")
    con.execute(
        "CREATE TABLE acqtemp AS SELECT loan_id,orig_channel,year_quarter,names.seller_name AS seller_name,new_seller_name FROM acq  LEFT JOIN names ON acq.seller_name = names.seller_name;"
    )
    con.execute("DROP TABLE IF EXISTS acq;")
    con.execute("ALTER TABLE acqtemp RENAME TO acq;")
    con.execute("DROP TABLE IF EXISTS names;")
    # acq_pdf = acq_pdf.merge(names, how='left', on=['seller_name'])
    # acq_pdf.drop(columns=['seller_name'], inplace=True)
    # acq_pdf['seller_name'] = acq_pdf['new_seller_name']
    # acq_pdf.drop(columns=['new_seller_name'], inplace=True)
    # DECLARE @pdf nvarchar(30)
    # SET pdf = perf_df_tmp
    create_ever_features()
    create_delinq_features()
    join_ever_delinq_features()
    create_joined_df()
    create_12_mon_features()
    combine_joined_12_mon()
    final_performance_delinquency()
    join_perf_acq_pdfs()
    print("compute time", (time.time() - t1) * 1000)
    final_pdf = last_mile_cleaning(final_pdf)  # noqa: F821 ("final_pdf" undefined variable)
    exec_time = (time.time() - t1) * 1000
    print("compute time with copy to host", exec_time)
    return final_pdf, exec_time


def pd_load_performance_csv(performance_path, con, fragment_size):
    """ Loads performance data

    Returns
    -------
    PD DataFrame
    """

    # cols = [
    #     "loan_id", "monthly_reporting_period", "servicer", "interest_rate", "current_actual_upb",
    #     "loan_age", "remaining_months_to_legal_maturity", "adj_remaining_months_to_maturity",
    #     "maturity_date", "msa", "current_loan_delinquency_status", "mod_flag", "zero_balance_code",
    #     "zero_balance_effective_date", "last_paid_installment_date", "foreclosed_after",
    #     "disposition_date", "foreclosure_costs", "prop_preservation_and_repair_costs",
    #     "asset_recovery_costs", "misc_holding_expenses", "holding_taxes", "net_sale_proceeds",
    #     "credit_enhancement_proceeds", "repurchase_make_whole_proceeds", "other_foreclosure_proceeds",
    #     "non_interest_bearing_upb", "principal_forgiveness_upb", "repurchase_make_whole_proceeds_flag",
    #     "foreclosure_principal_write_off_amount", "servicing_activity_indicator"
    # ]
    # dtypes = {
    #     "loan_id": np.int64,
    #     "monthly_reporting_period": str,
    #     "servicer": str,
    #     "interest_rate": np.float64,
    #     "current_actual_upb": np.float64,
    #     "loan_age": np.float64,
    #     "remaining_months_to_legal_maturity": np.float64,
    #     "adj_remaining_months_to_maturity": np.float64,
    #     "maturity_date": str,
    #     "msa": np.float64,
    #     "current_loan_delinquency_status": np.int32,
    #     "mod_flag": CategoricalDtype(['N', 'Y']),
    #     "zero_balance_code": CategoricalDtype(['01', '02', '06', '09', '03', '15', '16']),
    #     "zero_balance_effective_date": str,
    #     "last_paid_installment_date": str,
    #     "foreclosed_after": str,
    #     "disposition_date": str,
    #     "foreclosure_costs": np.float64,
    #     "prop_preservation_and_repair_costs": np.float64,
    #     "asset_recovery_costs": np.float64,
    #     "misc_holding_expenses": np.float64,
    #     "holding_taxes": np.float64,
    #     "net_sale_proceeds": np.float64,
    #     "credit_enhancement_proceeds": np.float64,
    #     "repurchase_make_whole_proceeds": np.float64,
    #     "other_foreclosure_proceeds": np.float64,
    #     "non_interest_bearing_upb": np.float64,
    #     "principal_forgiveness_upb": np.float64,
    #     "repurchase_make_whole_proceeds_flag": CategoricalDtype(['N', 'Y']),
    #     "foreclosure_principal_write_off_amount": np.float64,
    #     "servicing_activity_indicator": CategoricalDtype(['N', 'Y']),
    # }

    # print(performance_path)

    # p = pd.read_csv(performance_path, names=cols, delimiter='|', dtype=dtypes, parse_dates=[1,8,13,14,15,16])
    # return p

    create_table_names_temlate = """
CREATE TABLE perf (
    loan_id BIGINT,
    monthly_reporting_period DATE ENCODING DAYS(32),
    servicer TEXT ENCODING DICT(16),
    interest_rate DOUBLE,
    current_actual_upb DOUBLE,
    loan_age DOUBLE,
    remaining_months_to_legal_maturity DOUBLE,
    adj_remaining_months_to_maturity DOUBLE,
    maturity_date TEXT ENCODING DICT(16),
    msa DOUBLE,
    current_loan_delinquency_status INTEGER,
    mod_flag TEXT ENCODING DICT(16),
    zero_balance_code TEXT ENCODING DICT(16),
    zero_balance_effective_date DATE ENCODING DAYS(32),
    last_paid_installment_date DATE ENCODING DAYS(32),
    foreclosed_after DATE ENCODING DAYS(32),
    disposition_date DATE ENCODING DAYS(32),
    foreclosure_costs DOUBLE,
    prop_preservation_and_repair_costs DOUBLE,
    asset_recovery_costs DOUBLE,
    misc_holding_expenses DOUBLE,
    holding_taxes DOUBLE,
    net_sale_proceeds DOUBLE,
    credit_enhancement_proceeds DOUBLE,
    repurchase_make_whole_proceeds DOUBLE,
    other_foreclosure_proceeds DOUBLE,
    non_interest_bearing_upb DOUBLE,
    principal_forgiveness_upb DOUBLE,
    repurchase_make_whole_proceeds_flag TEXT ENCODING DICT(16),
    foreclosure_principal_write_off_amount DOUBLE,
    servicing_activity_indicator TEXT ENCODING DICT(16)
) WITH (FRAGMENT_SIZE= ##FRAGMENT_SIZE## );
"""
    create_table_names = create_table_names_temlate.replace(
        "##FRAGMENT_SIZE##", str(fragment_size)
    )
    import_query = import_query_template % ("perf", performance_path)

    con.execute(create_table_names)
    con.execute(import_query)


def pd_load_acquisition_csv(acquisition_path, con, fragment_size):
    """ Loads acquisition data

    Returns
    -------
    PD DataFrame
    """

    # columns = [
    #     'loan_id', 'orig_channel', 'seller_name', 'orig_interest_rate', 'orig_upb', 'orig_loan_term',
    #     'orig_date', 'first_pay_date', 'orig_ltv', 'orig_cltv', 'num_borrowers', 'dti', 'borrower_credit_score',
    #     'first_home_buyer', 'loan_purpose', 'property_type', 'num_units', 'occupancy_status', 'property_state',
    #     'zip', 'mortgage_insurance_percent', 'product_type', 'coborrow_credit_score', 'mortgage_insurance_type',
    #     'relocation_mortgage_indicator', 'year_quarter'
    # ]
    # dtypes = {
    #     'loan_id': np.int64,
    #     'orig_channel': CategoricalDtype(['B', 'C', 'R']),
    #     'seller_name': str,
    #     'orig_interest_rate': np.float64,
    #     'orig_upb': np.int64,
    #     'orig_loan_term': np.int64,
    #     'orig_date': str,
    #     'first_pay_date': str,
    #     'orig_ltv': np.float64,
    #     'orig_cltv': np.float64,
    #     'num_borrowers': np.float64,
    #     'dti': np.float64,
    #     'borrower_credit_score': np.float64,
    #     'first_home_buyer': CategoricalDtype(['N', 'U', 'Y']),
    #     'loan_purpose': CategoricalDtype(['C', 'P', 'R', 'U']),
    #     'property_type': CategoricalDtype(['CO', 'CP', 'MH', 'PU', 'SF']),
    #     'num_units': np.int64,
    #     'occupancy_status': CategoricalDtype(['I', 'P', 'S']),
    #     'property_state': CategoricalDtype(
    #         ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
    #         'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
    #         'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
    #         'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VI',
    #         'VT', 'WA', 'WI', 'WV', 'WY']),
    #     'zip': np.int64,
    #     'mortgage_insurance_percent': np.float64,
    #     'product_type': CategoricalDtype(['FRM']),
    #     'coborrow_credit_score': np.float64,
    #     'mortgage_insurance_type': np.float64,
    #     'relocation_mortgage_indicator': CategoricalDtype(['N', 'Y']),
    #     'year_quarter': np.int64
    # }

    # print(acquisition_path)

    # a = pd.read_csv(acquisition_path, names=columns, delimiter='|', dtype=dtypes, parse_dates=[6,7], error_bad_lines=True, warn_bad_lines=True, na_filter=True)
    # return a

    create_table_names_temlate = """
CREATE TABLE acq (
    loan_id BIGINT,
    orig_channel TEXT ENCODING DICT(32),
    seller_name TEXT ENCODING DICT(32),
    orig_interest_rate DOUBLE,
    orig_upb BIGINT,
    orig_loan_term BIGINT,
    orig_date DATE ENCODING DAYS(32),
    first_pay_date DATE ENCODING DAYS(32),
    orig_ltv DOUBLE,
    orig_cltv DOUBLE,
    num_borrowers DOUBLE,
    dti DOUBLE,
    borrower_credit_score DOUBLE,
    first_home_buyer TEXT ENCODING DICT(32),
    loan_purpose TEXT ENCODING DICT(32),
    property_type TEXT ENCODING DICT(32),
    num_units BIGINT,
    occupancy_status TEXT ENCODING DICT(32),
    property_state TEXT ENCODING DICT(32),
    zip BIGINT,
    mortgage_insurance_percent DOUBLE,
    product_type TEXT ENCODING DICT(32),
    coborrow_credit_score DOUBLE,
    mortgage_insurance_type DOUBLE,
    relocation_mortgage_indicator TEXT ENCODING DICT(32),
    year_quarter BIGINT
) WITH (FRAGMENT_SIZE= ##FRAGMENT_SIZE## );
"""
    create_table_names = create_table_names_temlate.replace(
        "##FRAGMENT_SIZE##", str(fragment_size)
    )
    import_query = import_query_template % ("acq", acquisition_path)

    con.execute(create_table_names)
    con.execute(import_query)


def pd_load_names(con, fragment_size):
    """ Loads names used for renaming the banks

     Returns
    -------
    PD DataFrame
    """

    # cols = [
    #     'seller_name', 'new_seller_name'
    # ]

    # dtypes = {'seller_name':str, 'new_seller_name':str}

    # n = pd.read_csv(os.path.join(data_directory, "names.csv"), names=cols, delimiter='|', dtype=dtypes)
    # return n

    create_table_names_temlate = """
CREATE TABLE names (
    seller_name TEXT ENCODING DICT(32),
    new_seller_name TEXT ENCODING DICT(32)
) WITH (FRAGMENT_SIZE= ##FRAGMENT_SIZE## );
"""
    create_table_names = create_table_names_temlate.replace(
        "##FRAGMENT_SIZE##", str(fragment_size)
    )
    import_query = import_query_template % ("names", os.path.join(data_directory, "names.csv"))

    con.execute(create_table_names)
    con.execute(import_query)


def create_ever_features():
    # everdf = pdf[['loan_id', 'current_loan_delinquency_status']]
    con.execute("DROP TABLE IF EXISTS everdf;")
    con.execute("DROP TABLE IF EXISTS everdftemp1;")
    # con.create_table('everdf', everdf);
    con.execute(
        "CREATE TABLE everdf AS (SELECT loan_id, current_loan_delinquency_status FROM perf);"
    )
    con.execute(
        "CREATE TABLE everdftemp1 AS (SELECT loan_id, MAX(current_loan_delinquency_status) AS current_loan_delinquency_status FROM everdf GROUP BY loan_id);"
    )
    # everdf = everdf.groupby('loan_id').max()
    con.execute("DROP TABLE IF EXISTS pdf;")
    # del(pdf)
    # con.execute('ALTER TABLE everdftemp1 ADD (ever_30 INT   ENCODING DICT);');
    con.execute("DROP TABLE IF EXISTS ever_30;")
    con.execute("DROP TABLE IF EXISTS ever_90;")
    con.execute("DROP TABLE IF EXISTS ever_180;")
    con.execute("DROP TABLE IF EXISTS ever_30temp;")
    con.execute("DROP TABLE IF EXISTS ever_90temp;")
    con.execute("DROP TABLE IF EXISTS ever_180temp;")
    # con.execute()
    con.execute(
        "CREATE TABLE ever_30 AS SELECT loan_id,current_loan_delinquency_status FROM everdf where current_loan_delinquency_status >= 1;"
    )
    con.execute(
        "UPDATE ever_30 SET current_loan_delinquency_status = CAST(current_loan_delinquency_status AS BIGINT);"
    )
    con.execute(
        "CREATE TABLE ever_90 AS SELECT loan_id,current_loan_delinquency_status FROM everdf where current_loan_delinquency_status >= 3;"
    )
    con.execute(
        "UPDATE ever_90 SET current_loan_delinquency_status = CAST(current_loan_delinquency_status AS BIGINT);"
    )
    con.execute(
        "CREATE TABLE ever_180 AS SELECT loan_id,current_loan_delinquency_status FROM everdf where current_loan_delinquency_status >= 6;"
    )
    con.execute(
        "UPDATE ever_180 SET current_loan_delinquency_status = CAST(current_loan_delinquency_status AS BIGINT);"
    )
    con.execute(
        "CREATE TABLE ever_30temp AS SELECT everdftemp1.loan_id AS loan_id,everdftemp1.current_loan_delinquency_status AS current_loan_delinquency_status ,ever_30.current_loan_delinquency_status AS ever_30 FROM ever_30  LEFT JOIN everdftemp1 ON ever_30.loan_id = everdftemp1.loan_id;"
    )
    con.execute(
        "CREATE TABLE ever_90temp AS SELECT ever_30temp.loan_id AS loan_id,ever_30temp.current_loan_delinquency_status AS current_loan_delinquency_status,ever_30temp.current_loan_delinquency_status AS ever_30 ,ever_90.current_loan_delinquency_status AS ever_90 FROM ever_30temp  LEFT JOIN ever_90 ON ever_90.loan_id = ever_30temp.loan_id;"
    )
    con.execute(
        "CREATE TABLE ever_180temp AS SELECT ever_90temp.loan_id AS loan_id,ever_90temp.current_loan_delinquency_status AS current_loan_delinquency_status,ever_90temp.ever_30 AS ever_30,ever_90temp.ever_90 ,ever_180.current_loan_delinquency_status AS ever_180 FROM ever_90temp  LEFT JOIN ever_180 ON ever_180.loan_id = ever_90temp.loan_id;"
    )
    con.execute("DROP TABLE IF EXISTS everdftemp1;")
    con.execute("DROP TABLE IF EXISTS mergetest;")
    # con.execute('CREATE TABLE mergetest');
    # con.execute('ALTER TABLE ever_30 ADD COLUMN test INT;');
    # con.execute('INSERT INTO ever_30 (test) select current_loan_delinquency_status FROM everdf;');
    con.execute("DROP TABLE IF EXISTS everdf1;")
    # con.execute('DROP TABLE IF EXISTS acq;');
    con.execute("ALTER TABLE ever_180temp  RENAME TO everdf1;")
    # con.execute('ALTER TABLE everdftemp1 RENAME TO everdf;');
    # con.execute('CREATE TABLE everdftemp1 AS SELECT loan_id FROM everdf;');
    # con.execute('CREATE TABLE everdftemp1 AS SELECT loan_id FROM everdf;');
    # con.execute('CREATE TABLE everdftemp1 AS SELECT loan_id FROM everdf;');
    # con.execute('DROP TABLE IF EXISTS everdf;');
    # con.execute('ALTER TABLE ever_180temp  RENAME TO everdftemp1 ;');
    con.execute("DROP TABLE IF EXISTS ever_30;")
    con.execute("DROP TABLE IF EXISTS ever_90;")
    con.execute("DROP TABLE IF EXISTS ever_180;")
    con.execute("DROP TABLE IF EXISTS ever_30temp;")
    con.execute("DROP TABLE IF EXISTS ever_90temp;")
    con.execute("DROP TABLE IF EXISTS ever_180temp;")
    # everdf.drop(columns=['current_loan_delinquency_status'], inplace=True)


def create_delinq_features():
    # delinq_pdf = pdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status']]
    con.execute("DROP TABLE IF EXISTS delinq;")
    con.execute("DROP TABLE IF EXISTS delinq_30 ;")
    con.execute("DROP TABLE IF EXISTS delinq_90 ;")
    con.execute("DROP TABLE IF EXISTS delinq_180 ;")

    # con.create_table('delinq',pdf);
    con.execute(
        "CREATE TABLE delinq AS (SELECT loan_id, monthly_reporting_period, current_loan_delinquency_status FROM perf);"
    )
    # delinq_30 = delinq_pdf.query('current_loan_delinquency_status >= 1')[['loan_id', 'monthly_reporting_period']].groupby('loan_id').min()
    con.execute(
        "CREATE TABLE delinq_30 AS (SELECT loan_id, MIN(monthly_reporting_period) AS monthly_reporting_period, monthly_reporting_period AS delinquency_30 FROM delinq WHERE current_loan_delinquency_status >= 1 GROUP BY loan_id,delinquency_30);"
    )
    # delinq_30['delinquency_30'] = delinq_30['monthly_reporting_period']
    con.execute(
        "CREATE TABLE delinq_90 AS (SELECT loan_id, MIN(monthly_reporting_period) AS monthly_reporting_period, monthly_reporting_period AS delinquency_90 FROM delinq WHERE current_loan_delinquency_status >= 3 GROUP BY loan_id,delinquency_90);"
    )
    con.execute(
        "CREATE TABLE delinq_180 AS (SELECT loan_id, MIN(monthly_reporting_period) AS monthly_reporting_period, monthly_reporting_period AS delinquency_180 FROM delinq WHERE current_loan_delinquency_status >= 6 GROUP BY loan_id,delinquency_180);"
    )

    # delinq_90 = delinq_pdf.query('current_loan_delinquency_status >= 3')[['loan_id', 'monthly_reporting_period']].groupby('loan_id').min()
    # delinq_90['delinquency_90'] = delinq_90['monthly_reporting_period']
    # delinq_90.drop(columns=['monthly_reporting_period'], inplace=True)
    # delinq_180 = delinq_pdf.query('current_loan_delinquency_status >= 6')[['loan_id', 'monthly_reporting_period']].groupby('loan_id').min()
    # delinq_180['delinquency_180'] = delinq_180['monthly_reporting_period']
    # delinq_180.drop(columns=['monthly_reporting_period'], inplace=True)
    con.execute("DROP TABLE IF EXISTS delinq_merge;")
    con.execute(
        "CREATE TABLE delinq_merge AS (SELECT delinq_30.delinquency_30 AS delinquency_30, delinq_30.loan_id AS loan_id, delinq_90.delinquency_90 AS delinquency_90 FROM delinq_30 LEFT JOIN delinq_90 ON delinq_30.loan_id=delinq_90.loan_id);"
    )
    # delinq_merge = delinq_30.merge(delinq_90, how='left', on=['loan_id'])
    # UPDATE UFOs SET shape='ovate' where shape='eggish'; 23:59:59.999
    # con.execute(' EXTRACT MONTH FROM 2018-08-01;');
    con.execute("UPDATE delinq_merge SET delinquency_90 = NULL WHERE delinquency_90 = NULL")
    # delinq_merge['delinquency_90'] = delinq_merge['delinquency_90'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    con.execute("DROP TABLE IF EXISTS delinq_mergetemp;")
    con.execute(
        "CREATE TABLE delinq_mergetemp AS (SELECT delinq_merge.delinquency_30 AS delinquency_30, delinq_merge.loan_id AS loan_id, delinq_merge.delinquency_90 AS delinquency_90, delinq_180.delinquency_180 AS delinquency_180 FROM delinq_merge LEFT JOIN delinq_180 ON delinq_merge.loan_id= delinq_180.loan_id);"
    )
    # con.execute('UPDATE delinq_merge SET  delinquency_90 = NULL where  delinquency_90 = NULL');
    # delinq_merge = delinq_merge.merge(delinq_180, how='left', on=['loan_id'])
    con.execute("DROP TABLE IF EXISTS delinq_merge;")
    con.execute("ALTER TABLE delinq_mergetemp RENAME TO delinq_merge;")
    # t =('test1;');
    # con.execute('UPDATE delinq_merge SET  delinquency_180 = NULL where  delinquency_180 = NULL');
    # print(row=con.execute("SELECT * from delinq_merge"));
    # rows=con.fetchall();
    # delinq_merge['delinquency_180'] = delinq_merge['delinquency_180'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    # del(delinq_30)
    # del(delinq_90)
    # del(delinq_180)
    con.execute("DROP TABLE IF EXISTS delinq_30;")
    con.execute("DROP TABLE IF EXISTS delinq_90;")
    con.execute("DROP TABLE IF EXISTS delinq_180;")


def join_ever_delinq_features():
    # everdf = everdf_tmp.merge(delinq_merge, on=['loan_id'], how='left')
    con.execute("DROP TABLE IF EXISTS ever;")
    con.execute("DROP TABLE IF EXISTS everdf;")
    con.execute("CREATE TABLE ever AS (SELECT * FROM delinq_merge);")
    # con.execute('DRPO TABLE IF EXISTS everdf;');
    # con.create_table('everdf',everdf);
    con.execute(
        "CREATE TABLE everdf AS (SELECT delinq_merge.delinquency_30 AS delinquency_30, delinq_merge.loan_id as Loan_id, delinq_merge.delinquency_90 AS delinquency_90, delinq_merge.delinquency_180 AS delinquency_180 FROM delinq_merge LEFT JOIN ever ON delinq_merge.loan_id=ever.loan_id);"
    )
    # con.execute('UPDATE everdf SET delinquency_90 =CAST (01/01/1970 AS TIMESTAMP(0)) where delinquency_90 = NULL;');
    # everdf['delinquency_30'] = everdf['delinquency_30'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    # everdf['delinquency_90'] = everdf['delinquency_90'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    # everdf['delinquency_180'] = everdf['delinquency_180'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))


def create_joined_df():
    # test = pdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status', 'current_actual_upb']]
    con.execute("DROP TABLE IF EXISTS test;")
    # con.create_table('test', test)
    con.execute(
        "CREATE TABLE test AS (SELECT loan_id, monthly_reporting_period, current_loan_delinquency_status, current_actual_upb FROM perf);"
    )
    # con.execute('DROP TABLE IF EXISTS mtemp;')
    # con.execute('DROP TABLE IF EXISTS ytemp;')
    # con.execute('DROP TABLE IF EXISTS yeartemp;')
    # con.execute('DROP TABLE IF EXISTS monthtemp;')
    con.execute("DROP TABLE IF EXISTS test2;")
    # del(pdf)
    con.execute(
        "CREATE TABLE TEST2 AS SELECT EXTRACT (YEAR FROM monthly_reporting_period) timestamp_year, EXTRACT (MONTH FROM monthly_reporting_period) timestamp_month, loan_id, current_loan_delinquency_status AS delinquency_12,monthly_reporting_period AS timestamp_temp, current_actual_upb AS upb_12 FROM test"
    )
    con.execute("DROP TABLE IF EXISTS test;")
    con.execute("ALTER TABLE TEST2  RENAME TO test;")
    con.execute("UPDATE test SET upb_12 = 999999999 WHERE upb_12 = NULL;")
    con.execute("UPDATE test SET delinquency_12 = -1 WHERE delinquency_12 = NULL;")
    # test['timestamp'] = test['monthly_reporting_period']
    # test.drop(columns=['monthly_reporting_period'], inplace=True)
    # test['timestamp_month'] = test['timestamp'].dt.month
    # test['timestamp_year'] = test['timestamp'].dt.year
    # test['delinquency_12'] = test['current_loan_delinquency_status']
    # test.drop(columns=['current_loan_delinquency_status'], inplace=True)
    # test['upb_12'] = test['current_actual_upb']
    # test.drop(columns=['current_actual_upb'], inplace=True)
    # test['upb_12'] = test['upb_12'].fillna(999999999)
    # test['delinquency_12'] = test['delinquency_12'].fillna(-1)
    con.execute("DROP TABLE IF EXISTS joined_df;")
    con.execute("DROP TABLE IF EXISTS joined_df1;")

    con.execute(
        "CREATE TABLE joined_df AS SELECT test.loan_id AS loan_id,test.timestamp_temp AS timestamp_temp,test.delinquency_12 AS delinquency_12, test.upb_12 AS upb_12,test.timestamp_year AS timestamp_year ,test.timestamp_month AS timestamp_month,everdf.delinquency_30 AS delinquency_30 ,everdf.delinquency_90 AS delinquency_90,everdf.delinquency_180 AS delinquency_180 FROM test LEFT JOIN everdf  ON test.loan_id= everdf.loan_id;"
    )
    con.execute(
        "CREATE TABLE joined_df1 AS (SELECT joined_df.loan_id AS loan_id, joined_df.timestamp_temp AS timestamp_temp,joined_df.delinquency_12 AS delinquency_12, joined_df.upb_12 AS upb_12, joined_df.timestamp_year AS timestamp_year, joined_df.timestamp_month AS timestamp_month, joined_df.delinquency_30 AS delinquency_30, joined_df.delinquency_90 AS delinquency_90, joined_df.delinquency_180 AS delinquency_180, everdf1.ever_30 AS ever_30, everdf1.ever_90 AS ever_90, everdf1.ever_180 AS ever_180 FROM joined_df LEFT JOIN everdf1 ON joined_df.loan_id=everdf1.loan_id);"
    )

    # con.execute('DROP TABLE IF EXISTS everdf;')
    con.execute("DROP TABLE IF EXISTS test;")
    con.execute("DROP TABLE IF EXISTS joined_df;")
    con.execute("ALTER TABLE joined_df1 RENAME TO joined_df;")
    # del(everdf)
    # del(test)
    con.execute("UPDATE joined_df SET ever_30 = -1 WHERE ever_30 = NULL;")
    con.execute("UPDATE joined_df SET ever_90 = -1 WHERE ever_90 = NULL;")
    con.execute("UPDATE joined_df SET ever_180 = -1 WHERE ever_180 = NULL;")
    # con.execute('UPDATE joined_df SET delinquency_90 = -1  where delinquency_90 = NULL;');
    # con.execute('UPDATE joined_df SET delinquency_180 = -1  where delinquency_180 = NULL;');
    # joined_df['ever_30'] = joined_df['ever_30'].fillna(-1)
    # joined_df['ever_90'] = joined_df['ever_90'].fillna(-1)
    # joined_df['ever_180'] = joined_df['ever_180'].fillna(-1)
    # joined_df['delinquency_30'] = joined_df['delinquency_30'].fillna(-1)
    # joined_df['delinquency_90'] = joined_df['delinquency_90'].fillna(-1)
    # joined_df['delinquency_180'] = joined_df['delinquency_180'].fillna(-1)
    con.execute("UPDATE joined_df SET upb_12 = 999999999 WHERE upb_12 = NULL;")
    # Ijoined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int32')
    # joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int32')

    # return joined_df
    # mapd_cursor =con.cursor()
    # query="SELECT loan_id, timestamp_temp,delinquency_12,upb_12,timestamp_year ,timestamp_month,delinquency_30 ,delinquency_90,delinquency_180,ever_30,ever_90,ever_180 from joined_df"
    # mapd_cursor.execute(query)
    # result=mapd_cursor.fetchall()
    # df =pd.DataFrame(result)
    # joined_df =df
    # df = con.select_ipc_gpu(query)
    # df =con.select_ipc_gpu(query)
    # df.head()
    # print("hahaaa")
    # print(result)
    # print(df)
    # return joined_df


def create_12_mon_features():
    testdfs = []  # noqa: F841 (assigned, but unused. Used in commented code.)
    n_months = 12
    for y in range(1, n_months + 1):
        string = str(y)
        months_string = str(n_months)
        # print(string)
        con.execute("DROP TABLE IF EXISTS tmpdf;")
        # con.execute('DROP TABLE IF EXISTS josh_monthstemp;')
        con.execute("DROP TABLE IF EXISTS delinq_12 ;")
        # con.execute('DROP TABLE IF EXISTS josh_monthsjoin ;');
        con.execute("DROP TABLE IF EXISTS josh_mody_ntemp ;")
        con.execute("DROP TABLE IF EXISTS finaltbl ;")
        # con.execute('SET string = y ;');
        con.execute(
            "CREATE TABLE tmpdf AS SELECT loan_id,timestamp_year,((timestamp_year * 12)+timestamp_month) AS josh_months,timestamp_month,delinquency_12,upb_12 from joined_df;"
        )
        # tmpdf = joined_df[['loan_id', 'timestamp_year', 'timestamp_month', 'delinquency_12', 'upb_12']]
        # con.execute('CREATE TABLE josh_monthstemp AS SELECT ((timestamp_year * 12)+timestamp_month) AS josh_months,loan_id FROM tmpdf;');
        # con.execute('CREATE TABLE josh_monthsjoin  AS SELECT tmpdf.loan_id AS loan_id, tmpdf.timestamp_year AS timestamp_year ,tmpdf.timestamp_month AS timestamp_month,tmpdf.delinquency_12 AS delinquency_12,tmpdf.upb_12 AS upb_12,josh_monthstemp.josh_months AS josh_months FROM tmpdf  LEFT JOIN josh_monthstemp  ON josh_monthstemp.loan_id= tmpdf.loan_id;');

        # tmpdf['josh_months'] = tmpdf['timestamp_year'] * 12 + tmpdf['timestamp_month']
        con.execute(
            "CREATE TABLE delinq_12 AS SELECT FLOOR((josh_months-24000-"
            + string
            + ")/12) AS josh_mody_n,loan_id from tmpdf;"
        )
        con.execute(
            "CREATE TABLE josh_mody_ntemp  AS SELECT tmpdf.timestamp_year AS timestamp_year ,tmpdf.timestamp_month AS timestamp_month,((delinquency_12 > 3) AND (upb_12 =0)) AS delinquency_12,tmpdf.upb_12 AS upb_12,tmpdf.josh_months AS josh_months ,tmpdf.loan_id AS loan_id, delinq_12.josh_mody_n AS josh_mody_n FROM tmpdf  LEFT JOIN delinq_12  ON tmpdf.loan_id= delinq_12.loan_id;"
        )

        # tmpdf['josh_mody_n'] = np.floor((tmpdf['josh_months'].astype('float64') - 24000 - y) / 12)

        # tmpdf = tmpdf.groupby(['loan_id', 'josh_mody_n'], as_index=False).agg({'delinquency_12': 'max','upb_12': 'min'})
        # con.execute('DROP TABLE IF EXISTS delinq_12temp;')
        con.execute("DROP TABLE IF EXISTS timestamp_yeartemp;")
        # con.execute('DROP TABLE IF EXISTS delinq_12_two;')
        # con.execute('CREATE TABLE delinq_12temp AS SELECT ((delinquency_12 >3) AND (upb_12 = 0)) AS delinquency_12 ,loan_id FROM tmpdf;');
        # con.execute('CREATE TABLE delinq_12_two  AS SELECT josh_mody_ntemp.timestamp_year AS timestamp_year ,josh_mody_ntemp.timestamp_month AS timestamp_month,josh_mody_ntemp.delinquency_12 AS delinquency_12,josh_mody_ntemp.upb_12 AS upb_12,josh_mody_ntemp.josh_months AS josh_months,josh_mody_ntemp.loan_id AS loan_id ,josh_mody_ntemp.josh_mody_n AS josh_mody_n,delinq_12temp.delinquency_12 AS delinquency_12 FROM josh_mody_ntemp  LEFT JOIN delinq_12temp  ON josh_mody_ntemp.loan_id= delinq_12temp.loan_id;');
        # tmpdf['delinquency_12'] = (tmpdf['delinquency_12']>3).astype('int32')
        # tmpdf['delinquency_12'] +=(tmpdf['upb_12']==0).astype('int32')
        # tmpdf['timestamp_year'] = np.floor(((tmpdf['josh_mody_n'] * n_months) + 24000 + (y - 1)) / 12).astype('int16')
        con.execute(
            "CREATE TABLE timestamp_yeartemp AS SELECT ((josh_mody_n * "
            + months_string
            + ") +2400 +("
            + string
            + " -1)/12) AS timestamp_year,loan_id from josh_mody_ntemp ;"
        )
        con.execute(
            "CREATE TABLE finaltbl AS SELECT josh_mody_ntemp.timestamp_month AS timestamp_month,josh_mody_ntemp.delinquency_12 AS delinquency_12,josh_mody_ntemp.upb_12 AS upb_12,josh_mody_ntemp.josh_months AS josh_months,josh_mody_ntemp.loan_id AS loan_id ,josh_mody_ntemp.josh_mody_n AS josh_mody_n,josh_mody_ntemp.delinquency_12 AS delinquency_12,timestamp_yeartemp.timestamp_year AS timestamp_year FROM josh_mody_ntemp  LEFT JOIN timestamp_yeartemp  ON timestamp_yeartemp.loan_id= josh_mody_ntemp.loan_id;"
        )
        con.execute("DROP TABLE IF EXISTS tmpdf;")
        con.execute("DROP TABLE IF EXISTS josh_monthstemp;")
        con.execute("DROP TABLE IF EXISTS delinq_12 ;")
        con.execute("DROP TABLE IF EXISTS josh_monthsjoin ;")
        con.execute("DROP TABLE IF EXISTS josh_mody_ntemp ;")
        con.execute("DROP TABLE IF EXISTS testdfs;")
        # con.execute('DROP TABLE IF EXISTS joined_df ;');
        con.execute("ALTER TABLE finaltbl RENAME TO testdfs;")
        # tmpdf['timestamp_month'] = np.int8(y)
    # tmpdf.drop(columns=['josh_mody_n'], inplace=True)
    # testdfs.append(tmpdf)
    # del(tmpdf)
    # del(joined_df)


# return pd.concat(testdfs)


def combine_joined_12_mon():
    # joined_df.drop(columns=['delinquency_12', 'upb_12'], inplace=True)
    # joined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int16')
    # joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int8')
    con.execute("DROP TABLE IF EXISTS join_final ;")
    con.execute(
        "CREATE TABLE join_final AS SELECT testdfs.timestamp_year AS timestamp_year ,testdfs.timestamp_month AS timestamp_month,testdfs.loan_id AS loan_id  FROM testdfs  LEFT JOIN joined_df  ON testdfs.loan_id= joined_df.loan_id;"
    )
    con.execute("DROP TABLE IF EXISTS joined_df ;")
    con.execute("ALTER TABLE join_final RENAME TO joined_df;")
    con.execute("DROP TABLE IF EXISTS join_final ;")

    # return joined_df.merge(testdf, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'])


def final_performance_delinquency():
    # merged = pdf[['loan_id', 'monthly_reporting_period']]
    # everdf1 = pdf[['loan_id', 'current_loan_delinquency_status']]
    con.execute("DROP TABLE IF EXISTS mergedtemp;")
    con.execute("CREATE TABLE mergedtemp AS (SELECT loan_id, monthly_reporting_period FROM perf);")
    # con.create_table('mergedtemp', merged)
    # merged['timestamp_month'] = merged['monthly_reporting_period'].dt.month
    # merged['timestamp_month'] = merged['timestamp_month'].astype('int8')
    # merged['timestamp_year'] = merged['monthly_reporting_period'].dt.year
    # merged['timestamp_year'] = merged['timestamp_year'].astype('int16')
    # merged = merged.merge(joined_df, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'])
    con.execute("DROP TABLE IF EXISTS merged_temp")
    con.execute("DROP TABLE IF EXISTS merged")
    # con.execute('SELECT loan_id,monthly_reporting_period FROM merged;');
    con.execute(
        "CREATE TABLE merged_temp AS SELECT EXTRACT (MONTH FROM monthly_reporting_period) timestamp_month,EXTRACT(YEAR FROM monthly_reporting_period) timestamp_year,loan_id FROM mergedtemp ;"
    )
    con.execute(
        "CREATE TABLE merged AS SELECT joined_df.timestamp_year AS time_stamp_year,joined_df.timestamp_month AS time_stamp_month,joined_df.loan_id AS loan_id  FROM merged_temp LEFT JOIN joined_df ON merged_temp.loan_id=joined_df.loan_id AND merged_temp.timestamp_year=joined_df.timestamp_year AND merged_temp.timestamp_month=joined_df.timestamp_month"
    )
    # con.execute('UPDATE merged SET time_stamp_year = CAST (time_stamp_year  AS DATE ENCODING FIXED(16));');
    con.execute("DROP TABLE IF EXISTS merged_temp")
    # merged.drop(columns=['timestamp_year'], inplace=True)
    # merged.drop(columns=['timestamp_month'], inplace=True)


def join_perf_acq_pdfs():
    # return perf.merge(acq, how='left', on=['loan_id'])
    con.execute("DROP TABLE IF EXISTS tempperf ")
    con.execute(
        "CREATE TABLE tempperf AS SELECT acq.loan_id AS loan_id,acq.seller_name FROM acq LEFT JOIN perf ON acq.loan_id = perf.loan_id;"
    )


def last_mile_cleaning(df, **kwargs):
    # for col, dtype in df.dtypes.iteritems():
    #    if str(dtype)=='category':
    #        df[col] = df[col].cat.codes
    # df['delinquency_12'] = df['delinquency_12'] > 0
    # df['delinquency_12'] = df['delinquency_12'].fillna(False).astype('int32')
    return df  # .to_arrow(index=False)


# Load database reporting functions
pathToReportDir = os.path.join(pathlib.Path(__file__).parent, "..", "report")
print(pathToReportDir)
sys.path.insert(1, pathToReportDir)

parser = argparse.ArgumentParser(description="Run Mortgage benchmark using pandas")

parser.add_argument(
    "-fs",
    dest="fragment_size",
    action="append",
    type=int,
    help="Fragment size to use for created table. Multiple values are allowed and encouraged.",
)
parser.add_argument("-r", default="report_pandas.csv", help="Report file name.")
parser.add_argument(
    "-df",
    default=1,
    type=int,
    help="Number of datafiles (quarters) to input into database for processing.",
)
parser.add_argument(
    "-dp", required=True, help="Path to root of mortgage datafiles directory (contains names.csv)."
)
parser.add_argument(
    "-i",
    dest="iterations",
    default=5,
    type=int,
    help="Number of iterations to run every benchmark. Best result is selected.",
)
parser.add_argument(
    "-port",
    default=62074,
    type=int,
    help="TCP port that omnisql client should use to connect to server",
)

parser.add_argument("-db-server", default="localhost", help="Host name of MySQL server")
parser.add_argument("-db-port", default=3306, type=int, help="Port number of MySQL server")
parser.add_argument(
    "-db-user",
    default="",
    help="Username to use to connect to MySQL database. If user name is specified, script attempts to store results in MySQL database using other -db-* parameters.",
)
parser.add_argument(
    "-db-pass", default="omniscidb", help="Password to use to connect to MySQL database"
)
parser.add_argument(
    "-db-name", default="omniscidb", help="MySQL database to use to store benchmark results"
)
parser.add_argument("-db-table", help="Table to use to store results for this benchmark.")

parser.add_argument(
    "-commit",
    default="1234567890123456789012345678901234567890",
    help="Commit hash to use to record this benchmark results",
)

args = parser.parse_args()

if args.df <= 0:
    print("Bad number of data files specified", args.df)
    sys.exit(1)

if args.iterations < 1:
    print("Bad number of iterations specified", args.t)

con = connect(
    user="admin", password="HyperInteractive", host="localhost", dbname="omnisci", port=args.port
)

db_reporter = None
if args.db_user != "":
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
benchName = "mortgage"

perf_data_path = os.path.join(data_directory, "perf")
perf_format_path = os.path.join(perf_data_path, "Performance_%sQ%s.txt")
bestExecTime = float("inf")
bestTotalTime = float("inf")
worstExecTime = 0
worstTotalTime = 0
avgExecTime = 0
avgTotalTime = 0

for fs in args.fragment_size:
    for iii in range(1, args.iterations + 1):
        dataFilesNumber = 0
        time_ETL = time.time()
        exec_time_total = 0
        print("RUNNING BENCHMARK NUMBER", benchName, "ITERATION NUMBER", iii)
        for quarter in range(0, args.df):
            year = 2000 + quarter // 4
            perf_file = perf_format_path % (str(year), str(quarter % 4 + 1))

            files = [
                f
                for f in pathlib.Path(perf_data_path).iterdir()
                if f.match("Performance_%sQ%s.txt*" % (str(year), str(quarter % 4 + 1)))
            ]
            for f in files:
                dataframe, exec_time = run_pd_workflow(
                    year=year, quarter=(quarter % 4 + 1), perf_file=str(f), fragment_size=fs
                )
                exec_time_total += exec_time
            dataFilesNumber += 1
        time_ETL_end = time.time()
        ttt = (time_ETL_end - time_ETL) * 1000
        print("ITERATION", iii, "EXEC TIME: ", exec_time_total, "TOTAL TIME: ", ttt)

        if bestExecTime > exec_time_total:
            bestExecTime = exec_time_total
        if worstExecTime < exec_time_total:
            worstExecTime = exec_time_total
        avgExecTime += exec_time_total
        if bestTotalTime > ttt:
            bestTotalTime = ttt
        if worstTotalTime < ttt:
            bestTotalTime = ttt
        avgTotalTime += ttt

avgExecTime /= args.iterations
avgTotalTime /= args.iterations

try:
    with open(args.r, "w") as report:
        print("BENCHMARK", benchName, "EXEC TIME", bestExecTime, "TOTAL TIME", bestTotalTime)
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
