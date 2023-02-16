### Intro
Benchmark with winning solution to optiver [realized volatility competition](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction). The dataset contains financial data such as orders and trades. The goal is to predict volatility in the next 10 minutes. The datset was originally obscured by authors and winning solutions resored original data to increase their score. That's why solution uses tools like nearest neighbours. 

### Data
The dataset doesn't really contain test set, it was not available to participants, they submited code that could read 
test set only on the host machines. So we'll not describe test set here. Available data:
Files:
- `book_train.parquet` - buy and trade orders (someone was ready to buy or sell)
- `trade_train.parquet` - stock trades (someone was ready to buy **and** someone was ready sell **with intersecting price**)
- `train.csv` - contains target (volatility) for each `stock_id` and `time_id`

#### Data obscurity
Dataset contains normalized prices and shuffled `time_ids`, winning solutions recovered correct prices 
and order of `time_id` to improve quality of solution.

### Files
1. `preprocess.py` - initial preprocessing of raw data
2. `prepare_dataset.py` - preparation of dataset for training
3. `benchmark.py` - file with benchmark, uses 2 files above
4. `train.py` - training part, currently not part of the benchmark. Trains LGB and DL models.
5. `optiver_utils.py` - utility functions

### Sources
- Competition: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction
- Original notebook with Apache 2 license: https://www.kaggle.com/code/nyanpn/1st-place-public-2nd-place-solution/notebook
- Description of solution: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970
