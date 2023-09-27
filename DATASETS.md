# Datasets

This is a comprehensive list of public datasets used by this repository. Before loading any of the datasets make sure you've read and accepted dataset rules. Loading instructions are available below.

| timedf name         | Name (Link/Source)                                                                                             | License                                                                                         |
| ------------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| plasticc            | [plasticc](https://www.kaggle.com/competitions/PLAsTiCC-2018)                                                  | [rules](https://www.kaggle.com/competitions/PLAsTiCC-2018/rules)                                |
| ny_taxi, ny_taxi_ml | [NYC TLC yellow taxi trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)               | [rules](https://www.nyc.gov/home/terms-of-use.page)                                             |

## Loading datasets

### plasticc

```
cd $TARGET_DIR
curl -Z -OOOOO https://modin-datasets.s3.amazonaws.com/plasticc/{test_set,test_set_metadata,test_set_skiprows,training_set,training_set_metadata}.csv
```

### ny_taxi

```
cd $TARGET_DIR
# For run with default params you only need this file
curl -O https://modin-datasets.s3.amazonaws.com/taxi/trips_xaa.csv
# You need these files only if you run benchmark with parameter -dfiles_num > 1, not needed for default params
curl -Z -OOOOOOOOOOOOOOOOOOO https://modin-datasets.s3.amazonaws.com/taxi/trips_xa{b..t}.csv
```

### ny_taxi_ml

```
cd $TARGET_DIR
wget https://modin-datasets.s3.amazonaws.com/ny_taxi_ml/ny_taxi_ml.tar.gz
tar -xvf ny_taxi_ml.tar.gz
mv ny_taxi_ml/201{4,5,6}/* ./
rm -rf ny_taxi_ml
```

