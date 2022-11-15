# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import sys
#from utils import measure  # not avail?!

#import cudf, dask_cudf
import cudf
from timeit import default_timer as timer

#from dask.distributed import Client, wait
#from dask_cuda import LocalCUDACluster

#cluster = LocalCUDACluster()
#client = Client(cluster)
#client

def read(filename):
    columns_names = [
        "trip_id",
        "vendor_id",
        "pickup_datetime",
        "dropoff_datetime",
        "store_and_fwd_flag",
        "rate_code_id",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "ehail_fee",
        "improvement_surcharge",
        "total_amount",
        "payment_type",
        "trip_type",
        "pickup",
        "dropoff",
        "cab_type",
        "precipitation",
        "snow_depth",
        "snowfall",
        "max_temperature",
        "min_temperature",
        "average_wind_speed",
        "pickup_nyct2010_gid",
        "pickup_ctlabel",
        "pickup_borocode",
        "pickup_boroname",
        "pickup_ct2010",
        "pickup_boroct2010",
        "pickup_cdeligibil",
        "pickup_ntacode",
        "pickup_ntaname",
        "pickup_puma",
        "dropoff_nyct2010_gid",
        "dropoff_ctlabel",
        "dropoff_borocode",
        "dropoff_boroname",
        "dropoff_ct2010",
        "dropoff_boroct2010",
        "dropoff_cdeligibil",
        "dropoff_ntacode",
        "dropoff_ntaname",
        "dropoff_puma",
    ]
    # use string instead of category
    columns_types = [
        "int64",
        "string",
        "timestamp",
        "timestamp",
        "string",
        "int64",
        "float64",
        "float64",
        "float64",
        "float64",
        "int64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "string",
        "float64",
        "string",
        "string",
        "string",
        "float64",
        "int64",
        "float64",
        "int64",
        "int64",
        "float64",
        "float64",
        "float64",
        "float64",
        "string",
        "float64",
        "float64",
        "string",
        "string",
        "string",
        "float64",
        "float64",
        "float64",
        "float64",
        "string",
        "float64",
        "float64",
        "string",
        "string",
        "string",
        "float64",
    ]

    dtypes = {columns_names[i]: columns_types[i] for i in range(len(columns_names))}
    all_but_dates = {
        col: valtype
        for (col, valtype) in dtypes.items()
        if valtype not in ["timestamp"]
    }
    dates_only = [col for (col, valtype) in dtypes.items() if valtype in ["timestamp"]]

    df = cudf.read_csv(
        filename,
        names=columns_names,
        dtype=all_but_dates,
        parse_dates=dates_only,
    )

        
    # Read multiples files from directory
    #import glob
    #all_files = glob.glob("/data/datag/200M/*.csv")
    #df = cudf.concat((cudf.read_csv(f, names=columns_names, dtype=all_but_dates, parse_dates=dates_only,) for f in all_files))	

    return df


def q1_hdk(df):
    q1_pandas_output = df.groupby("cab_type").size()
    q1_pandas_output.shape  # to trigger real execution
    return q1_pandas_output


def q2_hdk(df):
    q2_pandas_output = df.groupby("passenger_count").agg({"total_amount": "mean"})
    q2_pandas_output.shape  # to trigger real execution
    return q2_pandas_output


def q3_hdk(df):
    df["pickup_datetime"] = df["pickup_datetime"].dt.year
    q3_pandas_output = df.groupby(["passenger_count", "pickup_datetime"]).size()
    q3_pandas_output.shape  # to trigger real execution
    return q3_pandas_output


def q4_hdk(df):
    df["pickup_datetime"] = df["pickup_datetime"].dt.year
    df["trip_distance"] = df["trip_distance"].astype("int64")
    q4_pandas_output = (
        df.groupby(["passenger_count", "pickup_datetime", "trip_distance"], sort=False)
        .size()
        .reset_index()
        .sort_values(
            by=["pickup_datetime", 0], ignore_index=True, ascending=[True, False]
        )
    )
    q4_pandas_output.shape  # to trigger real execution
    return q4_pandas_output


def main():
    if len(sys.argv) != 2:
        print(
            f"USAGE: docker run --rm -v /path/to/dataset:/dataset python nyc-taxi-hdk.py <data file name starting with /dataset>"
        )
        return
    df = read(sys.argv[1])

    t1 = timer()
    q1_hdk(df)
    q1_time = timer() - t1

    t2 = timer()
    q2_hdk(df)
    q2_time = timer() - t2

    t3 = timer()
    q3_hdk(df.copy())
    q3_time = timer() - t3

    t4 = timer()
    q4_hdk(df.copy())
    q4_time = timer() - t4

    print("q1 time (ms)= ", q1_time*1e3)
    print("q2 time (ms)= ", q2_time*1e3)
    print("q3 time (ms)= ", q3_time*1e3)
    print("q4 time (ms)= ", q4_time*1e3)

if __name__ == "__main__":
    main()
