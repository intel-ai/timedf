```python
!pip install "modin[ray] @ git+https://github.com/modin-project/modin"
# !pip install ray==1.13.0
!pip install ray==2.0.0
```

    Collecting modin[ray]@ git+https://github.com/modin-project/modin
      Cloning https://github.com/modin-project/modin to /tmp/pip-install-0syqny3g/modin_150a06794cb24dcfb4ee8e41c59309c7
      Running command git clone --filter=blob:none -q https://github.com/modin-project/modin /tmp/pip-install-0syqny3g/modin_150a06794cb24dcfb4ee8e41c59309c7
      Resolved https://github.com/modin-project/modin to commit 11ba4811e6db11740e11fd33d3cdfba8ce5bec54
      Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting pandas==1.5.1
      Using cached pandas-1.5.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.2 MB)
    Requirement already satisfied: packaging in /opt/conda/lib/python3.9/site-packages (from modin[ray]@ git+https://github.com/modin-project/modin) (21.3)
    Requirement already satisfied: numpy>=1.18.5 in /opt/conda/lib/python3.9/site-packages (from modin[ray]@ git+https://github.com/modin-project/modin) (1.23.3)
    Requirement already satisfied: fsspec in /opt/conda/lib/python3.9/site-packages (from modin[ray]@ git+https://github.com/modin-project/modin) (2022.8.2)
    Requirement already satisfied: psutil in /opt/conda/lib/python3.9/site-packages (from modin[ray]@ git+https://github.com/modin-project/modin) (5.9.2)
    Requirement already satisfied: ray[default]>=1.4.0 in /opt/conda/lib/python3.9/site-packages (from modin[ray]@ git+https://github.com/modin-project/modin) (2.0.0)
    Requirement already satisfied: pyarrow>=4.0.1 in /opt/conda/lib/python3.9/site-packages (from modin[ray]@ git+https://github.com/modin-project/modin) (9.0.0)
    Requirement already satisfied: redis<4.0.0,>=3.5.0 in /opt/conda/lib/python3.9/site-packages (from modin[ray]@ git+https://github.com/modin-project/modin) (3.5.3)
    Requirement already satisfied: python-dateutil>=2.8.1 in /opt/conda/lib/python3.9/site-packages (from pandas==1.5.1->modin[ray]@ git+https://github.com/modin-project/modin) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.9/site-packages (from pandas==1.5.1->modin[ray]@ git+https://github.com/modin-project/modin) (2021.3)
    Requirement already satisfied: grpcio<=1.43.0,>=1.28.1 in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (1.43.0)
    Requirement already satisfied: protobuf<4.0.0,>=3.15.3 in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (3.20.3)
    Requirement already satisfied: frozenlist in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (1.3.1)
    Requirement already satisfied: click<=8.0.4,>=7.0 in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (8.0.4)
    Requirement already satisfied: jsonschema in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (4.1.2)
    Requirement already satisfied: attrs in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (21.2.0)
    Requirement already satisfied: filelock in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (3.8.0)
    Requirement already satisfied: msgpack<2.0.0,>=1.0.0 in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (1.0.4)
    Requirement already satisfied: aiosignal in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (1.2.0)
    Requirement already satisfied: requests in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (2.26.0)
    Requirement already satisfied: pyyaml in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (6.0)
    Requirement already satisfied: virtualenv in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (20.16.5)
    Requirement already satisfied: pydantic in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (1.10.2)
    Requirement already satisfied: opencensus in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (0.11.0)
    Requirement already satisfied: colorful in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (0.5.4)
    Requirement already satisfied: py-spy>=0.2.0 in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (0.3.14)
    Requirement already satisfied: aiohttp-cors in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (0.7.0)
    Requirement already satisfied: prometheus-client<0.14.0,>=0.7.1 in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (0.12.0)
    Requirement already satisfied: aiohttp>=3.7 in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (3.8.3)
    Requirement already satisfied: smart-open in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (6.2.0)
    Requirement already satisfied: gpustat>=1.0.0b1 in /opt/conda/lib/python3.9/site-packages (from ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (1.0.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/conda/lib/python3.9/site-packages (from packaging->modin[ray]@ git+https://github.com/modin-project/modin) (2.4.7)
    Requirement already satisfied: multidict<7.0,>=4.5 in /opt/conda/lib/python3.9/site-packages (from aiohttp>=3.7->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (6.0.2)
    Requirement already satisfied: charset-normalizer<3.0,>=2.0 in /opt/conda/lib/python3.9/site-packages (from aiohttp>=3.7->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (2.0.0)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /opt/conda/lib/python3.9/site-packages (from aiohttp>=3.7->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (4.0.2)
    Requirement already satisfied: yarl<2.0,>=1.0 in /opt/conda/lib/python3.9/site-packages (from aiohttp>=3.7->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (1.8.1)
    Requirement already satisfied: nvidia-ml-py<=11.495.46,>=11.450.129 in /opt/conda/lib/python3.9/site-packages (from gpustat>=1.0.0b1->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (11.495.46)
    Requirement already satisfied: blessed>=1.17.1 in /opt/conda/lib/python3.9/site-packages (from gpustat>=1.0.0b1->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (1.19.1)
    Requirement already satisfied: six>=1.7 in /opt/conda/lib/python3.9/site-packages (from gpustat>=1.0.0b1->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (1.16.0)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /opt/conda/lib/python3.9/site-packages (from jsonschema->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (0.17.3)
    Requirement already satisfied: google-api-core<3.0.0,>=1.0.0 in /opt/conda/lib/python3.9/site-packages (from opencensus->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (2.10.2)
    Requirement already satisfied: opencensus-context>=0.1.3 in /opt/conda/lib/python3.9/site-packages (from opencensus->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (0.1.3)
    Requirement already satisfied: typing-extensions>=4.1.0 in /opt/conda/lib/python3.9/site-packages (from pydantic->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (4.4.0)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.9/site-packages (from requests->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (2021.10.8)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/conda/lib/python3.9/site-packages (from requests->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (1.26.7)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.9/site-packages (from requests->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (3.1)
    Requirement already satisfied: distlib<1,>=0.3.5 in /opt/conda/lib/python3.9/site-packages (from virtualenv->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (0.3.6)
    Requirement already satisfied: platformdirs<3,>=2.4 in /opt/conda/lib/python3.9/site-packages (from virtualenv->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (2.5.2)
    Requirement already satisfied: wcwidth>=0.1.4 in /opt/conda/lib/python3.9/site-packages (from blessed>=1.17.1->gpustat>=1.0.0b1->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (0.2.5)
    Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.56.2 in /opt/conda/lib/python3.9/site-packages (from google-api-core<3.0.0,>=1.0.0->opencensus->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (1.56.4)
    Requirement already satisfied: google-auth<3.0dev,>=1.25.0 in /opt/conda/lib/python3.9/site-packages (from google-api-core<3.0.0,>=1.0.0->opencensus->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (2.12.0)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/conda/lib/python3.9/site-packages (from google-auth<3.0dev,>=1.25.0->google-api-core<3.0.0,>=1.0.0->opencensus->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (5.2.0)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/conda/lib/python3.9/site-packages (from google-auth<3.0dev,>=1.25.0->google-api-core<3.0.0,>=1.0.0->opencensus->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (0.2.8)
    Requirement already satisfied: rsa<5,>=3.1.4 in /opt/conda/lib/python3.9/site-packages (from google-auth<3.0dev,>=1.25.0->google-api-core<3.0.0,>=1.0.0->opencensus->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (4.9)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /opt/conda/lib/python3.9/site-packages (from pyasn1-modules>=0.2.1->google-auth<3.0dev,>=1.25.0->google-api-core<3.0.0,>=1.0.0->opencensus->ray[default]>=1.4.0->modin[ray]@ git+https://github.com/modin-project/modin) (0.4.8)
    Building wheels for collected packages: modin
      Building wheel for modin (setup.py) ... [?25ldone
    [?25h  Created wheel for modin: filename=modin-0.16.0+24.g11ba481-py3-none-any.whl size=959350 sha256=bf2fe2f287c88b5315fc34497b4e27ee199561cf51f261b5a3e46feb356b4328
      Stored in directory: /tmp/pip-ephem-wheel-cache-s4v4rzln/wheels/39/a3/8f/1456565b4629a11de16bef3ee4e3ee7f98562aa1220e080e68
    Successfully built modin
    Installing collected packages: pandas, modin
      Attempting uninstall: pandas
        Found existing installation: pandas 1.4.4
        Uninstalling pandas-1.4.4:
          Successfully uninstalled pandas-1.4.4
      Attempting uninstall: modin
        Found existing installation: modin 0.7.3+1275.g9fe020ba
        Uninstalling modin-0.7.3+1275.g9fe020ba:
          Successfully uninstalled modin-0.7.3+1275.g9fe020ba
    Successfully installed modin-0.16.0+24.g11ba481 pandas-1.5.1
    Requirement already satisfied: ray==2.0.0 in /opt/conda/lib/python3.9/site-packages (2.0.0)
    Requirement already satisfied: click<=8.0.4,>=7.0 in /opt/conda/lib/python3.9/site-packages (from ray==2.0.0) (8.0.4)
    Requirement already satisfied: attrs in /opt/conda/lib/python3.9/site-packages (from ray==2.0.0) (21.2.0)
    Requirement already satisfied: frozenlist in /opt/conda/lib/python3.9/site-packages (from ray==2.0.0) (1.3.1)
    Requirement already satisfied: pyyaml in /opt/conda/lib/python3.9/site-packages (from ray==2.0.0) (6.0)
    Requirement already satisfied: numpy>=1.19.3 in /opt/conda/lib/python3.9/site-packages (from ray==2.0.0) (1.23.3)
    Requirement already satisfied: grpcio<=1.43.0,>=1.28.1 in /opt/conda/lib/python3.9/site-packages (from ray==2.0.0) (1.43.0)
    Requirement already satisfied: jsonschema in /opt/conda/lib/python3.9/site-packages (from ray==2.0.0) (4.1.2)
    Requirement already satisfied: protobuf<4.0.0,>=3.15.3 in /opt/conda/lib/python3.9/site-packages (from ray==2.0.0) (3.20.3)
    Requirement already satisfied: msgpack<2.0.0,>=1.0.0 in /opt/conda/lib/python3.9/site-packages (from ray==2.0.0) (1.0.4)
    Requirement already satisfied: filelock in /opt/conda/lib/python3.9/site-packages (from ray==2.0.0) (3.8.0)
    Requirement already satisfied: requests in /opt/conda/lib/python3.9/site-packages (from ray==2.0.0) (2.26.0)
    Requirement already satisfied: virtualenv in /opt/conda/lib/python3.9/site-packages (from ray==2.0.0) (20.16.5)
    Requirement already satisfied: aiosignal in /opt/conda/lib/python3.9/site-packages (from ray==2.0.0) (1.2.0)
    Requirement already satisfied: six>=1.5.2 in /opt/conda/lib/python3.9/site-packages (from grpcio<=1.43.0,>=1.28.1->ray==2.0.0) (1.16.0)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /opt/conda/lib/python3.9/site-packages (from jsonschema->ray==2.0.0) (0.17.3)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /opt/conda/lib/python3.9/site-packages (from requests->ray==2.0.0) (2.0.0)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.9/site-packages (from requests->ray==2.0.0) (2021.10.8)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/conda/lib/python3.9/site-packages (from requests->ray==2.0.0) (1.26.7)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.9/site-packages (from requests->ray==2.0.0) (3.1)
    Requirement already satisfied: distlib<1,>=0.3.5 in /opt/conda/lib/python3.9/site-packages (from virtualenv->ray==2.0.0) (0.3.6)
    Requirement already satisfied: platformdirs<3,>=2.4 in /opt/conda/lib/python3.9/site-packages (from virtualenv->ray==2.0.0) (2.5.2)



```python
import modin
modin.__version__
```




    '0.16.0+24.g11ba481'




```python
import os
os.environ["PREFECT_API_URL"]="http://x1infra.jf.intel.com/api"
os.environ["__MODIN_AUTOIMPORT_PANDAS__"]="1"

MODIN_ENGINE = 'ray'
RAY_URL = 'ray://ray-head-svc.ray:10001'
# RAY_URL = 'ray://ray-ray-head.ray:10001'
# RAY_URL = 'ray://ray-ray-head.ray.svc.cluster.local:10001'
AWS_ACCESS_KEY_ID = 'x1miniouser'
AWS_SECRET_ACCESS_KEY = 'x1miniopass'

# import pandas as pd
import modin.pandas as pd
import modin.config as cfg

os.environ["MODIN_ENGINE"] = MODIN_ENGINE

import ray
ray.shutdown()
extra_init_kw = {
    "runtime_env": {
        'pip': ["modin[ray] @ git+https://github.com/modin-project/modin", 'ray==2.0.0'],
        # 'pip': ['modin[ray] @ git+https://github.com/intel-ai/modin@taxi-modin', 'ray==1.13.0'],
        # "env_vars": {
            # "__MODIN_AUTOIMPORT_PANDAS__": "1",
            # 'HTTP_PROXY': 'http://proxy-us.intel.com:912',
            # 'HTTPS_PROXY': 'http://proxy-us.intel.com:912',
            # 'NO_PROXY': 'intel.com,.intel.com,10.0.0.0/8,192.168.0.0/16,localhost,127.0.0.0/8,134.134.0.0/16,172.16.0.0/16,cluster.local',
        # }
    }
}
ray.init(RAY_URL, **extra_init_kw)
print("CPU =", ray.cluster_resources()["CPU"])
print("ray version =", ray.__version__)
cfg.NPartitions.put(120)
```

    CPU = 256.0
    ray version = 2.0.0



```python
##########################################################################
### Predicting NYC Taxi Fares with Intel Optimizations on Full Dataset ###
##########################################################################

import glob
from timeit import default_timer as dt
# import modin.pandas as pd


###########################
### Inspecting the Data ###
###########################

####################
### Data Cleanup ###
####################

start_whole = dt()

start = dt()

data_types_2014 = {
    ' tolls_amount': 'float64',
    ' surcharge': 'float64',
    ' store_and_fwd_flag': 'object',
    ' tip_amount': 'float64',
    'tolls_amount': 'float64',
}

data_types_2015 = {
    'extra': 'float64',
    'tolls_amount': 'float64',
}

data_types_2016 = {
    'tip_amount': 'float64',
    'tolls_amount': 'float64',
}

#Dictionary of required columns and their datatypes
must_haves = {
     'pickup_datetime': 'datetime64[s]',
     'dropoff_datetime': 'datetime64[s]',
     'passenger_count': 'int32',
     'trip_distance': 'float32',
     'pickup_longitude': 'float32',
     'pickup_latitude': 'float32',
     'rate_code': 'int32',
     'dropoff_longitude': 'float32',
     'dropoff_latitude': 'float32',
     'fare_amount': 'float32'
    }


def clean(ddf, must_haves):
    # replace the extraneous spaces in column names and lower the font type
    tmp = {col:col.strip().lower() for col in list(ddf.columns)}
    ddf = ddf.rename(columns=tmp)

    ddf = ddf.rename(columns={
        'tpep_pickup_datetime': 'pickup_datetime',
        'tpep_dropoff_datetime': 'dropoff_datetime',
        'ratecodeid': 'rate_code'
    })
    to_drop = ddf.columns.difference(must_haves.keys())
    if not to_drop.empty:
        ddf = ddf.drop(columns=to_drop)
    to_fillna = [col for dt, col in zip(ddf.dtypes, ddf.dtypes.index) if dt == "object"]
    if to_fillna:
        ddf[to_fillna] = ddf[to_fillna].fillna('-1')
    # for col in ddf.columns:
    #     if col not in must_haves:
    #         ddf = ddf.drop(columns=col)
    #         continue
    #     if ddf[col].dtype == 'object':
    #         ddf[col] = ddf[col].fillna('-1')
    return ddf


def read_csv_from_minio(minio_file_name, parse_dates=[], dtype=None):    
    file = pd.read_csv(
                    minio_file_name,
                    parse_dates=parse_dates,
                    dtype=dtype,
                    storage_options={
                        'key': AWS_ACCESS_KEY_ID,
                        'secret': AWS_SECRET_ACCESS_KEY,
                        'client_kwargs': {
                            'endpoint_url': 'https://minio.minio',
                            'verify': False,
                        },
                        'config_kwargs': {
                            'region_name': 'us-west-1',
                            'signature_version': 's3v4',
                        }
                    }
    )
    return file

import s3fs
fs = s3fs.S3FileSystem(
    key='x1miniouser',
    secret='x1miniopass',
    client_kwargs={
        'endpoint_url': 'https://minio.minio',
        'verify': False,
    },
    config_kwargs={
        'region_name': 'us-west-1',
        'signature_version': 's3v4',
    },
)

base_path = 's3://yellow-taxi-dataset/'

df_2014 = [
    clean(read_csv_from_minio('s3://' + x, parse_dates=[' pickup_datetime', ' dropoff_datetime'], dtype=data_types_2014), must_haves)
    for x in fs.glob(base_path+'2014/*.csv')]

df_2015 = [
    clean(read_csv_from_minio('s3://' + x, parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], dtype=data_types_2015), must_haves)
    for x in fs.glob(base_path+'2015/*.csv')]

end = dt()
print("Data Cleanup: ", end - start)
```


```python
##############################################
### Handling 2016's Mid-Year Schema Change ###
##############################################

start = dt()

months = [str(x).rjust(2, '0') for x in range(1, 7)]
valid_files = [base_path+'2016/yellow_tripdata_2016-'+month+'.csv' for month in months]

#read & clean 2016 data and concat all DFs
df_2016 = [
    clean(read_csv_from_minio(x, parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], dtype=data_types_2016), must_haves)
    for x in valid_files]

#concatenate multiple DataFrames into one bigger one
taxi_df = pd.concat(df_2014 + df_2015 + df_2016, ignore_index=True)

end = dt()
print("Handling 2016's Mid-Year Schema Change: ", end - start)
```


```python
#######################################
### Exploratory Data Analysis (EDA) ###
#######################################

start = dt()

# apply a list of filter conditions to throw out records with missing or outlier values
taxi_df  = taxi_df.query("(fare_amount > 1) & \
    (fare_amount < 500) & \
    (passenger_count > 0) & \
    (passenger_count < 6) & \
    (pickup_longitude > -75) & \
    (pickup_longitude < -73) & \
    (dropoff_longitude > -75) & \
    (dropoff_longitude < -73) & \
    (pickup_latitude > 40) & \
    (pickup_latitude < 42) & \
    (dropoff_latitude > 40) & \
    (dropoff_latitude < 42) & \
    (trip_distance > 0) & \
    (trip_distance < 500) & \
    ((trip_distance <= 50) | (fare_amount >= 50)) & \
    ((trip_distance >= 10) | (fare_amount <= 300)) & \
    (dropoff_datetime > pickup_datetime)")


# reset_index and drop index column
taxi_df = taxi_df.reset_index(drop=True)

end = dt()
print("Exploratory Data Analysis (EDA): ", end - start)
```


```python
# ###################################
# ### Adding Interesting Features ###
# ###################################

# start = dt()

# ## add features
# taxi_df['day'] = taxi_df['pickup_datetime'].dt.day

# #calculate the time difference between dropoff and pickup.
# taxi_df['diff'] = taxi_df['dropoff_datetime'].astype('int64') - taxi_df['pickup_datetime'].astype('int64')

# taxi_df['pickup_latitude_r'] = taxi_df['pickup_latitude']//.01*.01
# taxi_df['pickup_longitude_r'] = taxi_df['pickup_longitude']//.01*.01
# taxi_df['dropoff_latitude_r'] = taxi_df['dropoff_latitude']//.01*.01
# taxi_df['dropoff_longitude_r'] = taxi_df['dropoff_longitude']//.01*.01

# # taxi_df[["pickup_latitude_r", "pickup_longitude_r", "dropoff_latitude_r", "dropoff_longitude_r"]] = taxi_df[["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]] // (.01*.01)

# # multiplied = taxi_df[["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]] // (.01*.01)

# # taxi_df['pickup_latitude_r'] = multiplied['pickup_latitude']
# # taxi_df['pickup_longitude_r'] = multiplied['pickup_longitude']
# # taxi_df['dropoff_latitude_r'] = multiplied['dropoff_latitude']
# # taxi_df['dropoff_longitude_r'] = multiplied['dropoff_longitude']

# # taxi_df = taxi_df.drop('pickup_datetime', axis=1)
# # taxi_df = taxi_df.drop('dropoff_datetime', axis=1)
# taxi_df = taxi_df.drop(['pickup_datetime', 'dropoff_datetime'], axis=1)

# dlon = taxi_df['dropoff_longitude'] - taxi_df['pickup_longitude']
# dlat = taxi_df['dropoff_latitude'] - taxi_df['pickup_latitude']
# taxi_df['e_distance'] = dlon * dlon + dlat * dlat

# end = dt()
# print("Adding Interesting Features: ", end - start)


start = dt()

## add features
taxi_df['day'] = taxi_df['pickup_datetime'].dt.day

#calculate the time difference between dropoff and pickup.
taxi_df['diff'] = taxi_df['dropoff_datetime'].astype('int64') - taxi_df['pickup_datetime'].astype('int64')

taxi_df[
    [
        "pickup_longitude_r",
        "pickup_latitude_r",
        "dropoff_longitude_r",
        "dropoff_latitude_r",
    ]
] = taxi_df[["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]] // (0.01 * 0.01)

taxi_df = taxi_df.drop(['pickup_datetime', 'dropoff_datetime'], axis=1)

dlon = taxi_df['dropoff_longitude'] - taxi_df['pickup_longitude']
dlat = taxi_df['dropoff_latitude'] - taxi_df['pickup_latitude']
taxi_df['e_distance'] = dlon * dlon + dlat * dlat
end = dt()
print("Adding Interesting Features: ", end - start)
```


```python
###########################
### Pick a Training Set ###
###########################

start = dt()

#since we calculated the h_distance let's drop the trip_distance column, and then do model training with XGB.
taxi_df = taxi_df.drop('trip_distance', axis=1)

# this is the original data partition for train and test sets.
X_train = taxi_df[taxi_df.day < 25]

# create a Y_train ddf with just the target variable
Y_train = X_train[['fare_amount']]
# Y_train = X_train[['fare_amount']]._to_pandas()
# drop the target variable from the training ddf
X_train = X_train.drop("fare_amount", axis=1)
# X_train = X_train.drop("fare_amount", axis=1)._to_pandas()
# X_train = X_train[X_train.columns.difference(['fare_amount'])]
# X_train = X_train[X_train.columns.difference(['fare_amount'])]._to_pandas()

end = dt()
print("Pick a Training Set: ", end - start)
```


```python
#######################
### Pick a Test Set ###
#######################

from modin.config import LogMode, LogMemoryInterval, LogFileSize
# LogMode.enable()

start = dt()

X_test = taxi_df[taxi_df.day >= 25]

# Create Y_test with just the fare amount
Y_test = X_test[['fare_amount']]

# Drop the fare amount from X_test
X_test = X_test.drop("fare_amount", axis=1)

end = dt()

print("Pick a Test Set: ", end - start)

end_whole = dt()
print("whole time: ", end_whole - start_whole)
```


```python
ray.shutdown()
```


```python

```
