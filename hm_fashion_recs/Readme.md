### Into
Kaggle H&M fashion competition represents typical recommendation task for e-commerce. We need to predict what users are likely to buy in the next week based on past activity. We are provided with customer features, item features and transaction history.

### Data
- user features
- item features
- past transactions 

### Validity of our source code
This is a benchmark with a solution to H&M fashion recommendation competition. It is based on the following solution: . Authors of the original solution were 11th in that competition and provided source code to their solution under Apache2 license. Original solution requires combination of 4 ipython notebooks, but we only use one of them for simplicity. Authors claim that such solution achieves the score of X, which is enough to score 50 in the competition. Authors of the original code didn't know about modin and hence, didn't try to optimize for it.

### Solution
#### Time split by week
In this solution data (transactions) split is perfomed by weeks, so it's best to understand the logic and terms.
Data contains past user transactions and we are trying to predict transactions for one future week. Past transactions cover about 100 weeks. During the solution we encode transaction week as following: `week=0` means that transaction happened during a week that just ended, `1` means one week ago, `2` means two weeks ago, etc. So the goal of the competition is to predict transactions for `week=-1` and we have transactions for `weeks=[104, 103, ..., 1, 0]` (from past to today) to do so. So filtering like `week >= 1` in the code means that we keep all the past weeks except the most recent one.

#### Preprocessing
Preprocessing is stored in `preprocess.py` file and in `lfm.py`.
Steps of data preprocessing
1. Transform data. We load raw data from `csv` files, perform basic preprocessing and store results into 3 `pickle` files with customer, item and transaction data.
2. Create user OHEs for `week >= X` for `X=0,1,...TRAIN_WEEKS`. We load data from `pickle` files and generate one-hot encoding features for users based on transactions with `week >= X` (so we will use all past transaction until week `X+1`). We do that for `X=0,1,...TRAIN_WEEKS` to use that data for model training, using different points in time.
3. Generate LFM embeddings. We generate `LightFM` embeddings based on different point in time: `week >= X`. These embeddings will be used in the future for feature engineering. By default `week_processing_benchmark.py` is not performing this step and not generating corresponding features to minimize external dependencies.

#### One week data processing
Complete data processing **for a single time split** (`week >= 1`) is stored in `week_processing_benchmark.py` file. It contains feature generation and label extraction starting from raw data. It can be used to measure data processing time for a single time split, which is the most relevant metric for a library like `modin`.

#### Complete benchmark
Complete reproduction is stored in `main.py` file. It will perform complete reproduction of the solution, repeating data processing several times for different time splits, training validation and submission model. This will be much longer than one week data processing and performance will be much harder to interpret.
Steps of the solution:
1. Preprocessing (like described above)
2. Generate candidates with various time splits starting from `week=0` as GT. This step could be part of model training and submission generated, but is separated to avoide work duplication.
2. Find the best number of iterations (`best_iteration`)
    1. Generate train set. We perform *one week data processing* with several splits and concatenate generated features and labels. We will start with `week=1` as GT and keep going in the past for the selected number of weeks `train_weeks`.
    2. Prepare val set from transactions with `week=0` as GT and transactions with `week > 0` as `x_val`.
    2. Train model with generated dataset
    3. Evaluate model performance on the val set
    4. Remember optimal hyperparameters
    3. 
3. Make submission
    1. Generate train set with all available data splits
    2. Train model with `n_iter=best_iteration`
    3. Predict with

#### Candidate generation


### Complete benchmark

### Files
- `fe.py`
- `lfm.py`
- `main.py`
- `metric.py`
- `preprocess.py`
- `schema.py`
- `tm.py`
- `vars.py`
- `week_processing_benchmark.py`