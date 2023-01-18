### Intro
[Kaggle H&M fashion competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) represents typical recommendation task for e-commerce. We need to predict what users are likely to buy in the next week based on past activity. We are provided with customer features, item features and transaction history.

This is a benchmark with a solution to H&M fashion recommendation competition. It is based on the following [solution](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324084) and it's [source code](https://github.com/ryowk/kaggle-h-and-m-personalized-fashion-recommendations). Authors of the original solution were 11th in that competition and provided source code to their solution under Apache license 2.0. Original solution requires combination of 4 ipython notebooks, but we only use one of them for simplicity (`local6.ipynb`). Authors claim that such solution achieves the score of `0.03391`, which is enough to score in top 15 in the competition. Authors of the original code didn't know about modin and hence, didn't try to optimize source code for modin use.

### Data
- user features (`customers.csv` 207MB)
- item features (`articles.csv` 36MB)
- past transactions (`transactions_train.csv` 3.5GB)
- item images (not used, about 30GB)

### Solution
#### Time split by week
In this solution data split (transactions split) is performed by weeks, so it's best to understand the logic and terms.
Data contains past user transactions, and we are trying to predict transactions for one future week. Past transactions cover about 100 weeks. During the solution we encode week of a transaction as following: `week=0` means that transaction happened during a week that just ended, `1` means one week ago, `2` means two weeks ago, etc. So the goal of the competition is to predict transactions for `week=-1` and we have transactions for `weeks=[104, 103, ..., 1, 0]` (from past to today) to do so. So, filtering like `week >= 1` in the code means that we keep all the past weeks except the most recent one.

#### User one-hot encoding (OHE)
During data preprocessing for each user and a chosen week `W` we encode each category of each purchased items during `week >= W` with [one-hot encoding](https://en.wikipedia.org/wiki/One-hot) and calculate mean value this vector for all transactions of that user. Then these user features (dependent on user and time split) are saved on disk. They are used in multiple other places for candidate generation and feature engineering.

#### Preprocessing
Preprocessing is stored in `preprocess.py` file and in `lfm.py`.
Steps of data preprocessing
1. Transform data. We load raw data from `csv` files, perform basic preprocessing and store results into 3 `pickle` files with customer, item and transaction data.
2. Create user OHEs for `week >= X` for `X=0,1,...TRAIN_WEEKS`. We load data from `pickle` files and generate one-hot encoding features for users based on transactions with `week >= X` (so we will use all past transaction until week `X+1`). We do that for `X=0,1,...TRAIN_WEEKS` to use that data for model training, using different points in time. These OHEs will be used for candidate generation and feature engineering.
3. Generate LFM embeddings. We generate `LightFM` embeddings based on different point in time: `week >= X`. These embeddings will be used in the future for feature engineering. By default, `week_processing_benchmark.py` is not performing this step and not generating corresponding features to minimize external dependencies.

To run preprocessing separately from everything else you need to run `preprocess.py` (for steps 1 and 2) and `lfm.py` for LFM embeddings.

#### One week data processing
Complete data processing **for a single time split** (`week >= 1`) is stored in `week_processing_benchmark.py` file. It provides feature generation and label extraction starting from raw data. It can be used to measure data processing time for a single time split, which is the most relevant metric for a library like `modin`. Essentially that is a part of train set generation with X covering `week>=1` and y covering `week=0`. To generate complete train set you would need to perform the same logic with varying time split.
Key steps:
1. Preprocess data as described above
2. Generate candidates: use several heuristics to generate likely candidates for future purchases. In the next stage we will be augmenting these candidates with features, to train model to find the most likely candidates.
3. Attach features to generated candidates.

To run one week processing just run the benchmark as any other benchmark. You don't need to run preprocessing beforehand.

#### Complete benchmark
Complete reproduction, except for data preprocessing is stored in `notebook.py` file. It will perform complete reproduction of the solution, repeating data processing several times for different time splits, training validation and submission model. This will be much longer than one week data processing and performance will be much harder to interpret.
Steps of the solution:
1. Generate candidates with various time splits starting from `week=0` as GT. This step could be part of model training and submission generation but is separated to avoid work duplication.
2. Find the best number of iterations (`best_iteration`)
    1. Generate train set. We perform *one week data processing* with several splits and concatenate generated features and labels. We will start with `week=1` as GT and keep going in the past for the selected number of weeks `train_weeks`.
    2. Prepare val set from transactions with `week=0` as GT and transactions with `week > 0` as `x_val`.
    2. Train model with generated dataset
    3. Evaluate model performance on the val set
    4. Remember optimal number of iterations
3. Make submission
    1. Generate train set with all the available data splits starting from `week=0` as GT
    2. Train model with `n_iter=best_iteration`
    3. Predict for the `week=-1`
    4. Generate submission file

### Files
Scripts:
- `week_processing_benchmark.py` - main benchmark, contains dataset preparation starting from raw data. Intended benchmark, relatively easy to interpret.
- `notebook.py` - script, based on [local6.ipynb](https://github.com/ryowk/kaggle-h-and-m-personalized-fashion-recommendations/blob/main/local6.ipynb), reproducing it. You only need to work with it directly if you want to reproduce the whole benchmark, including model training, could take about 24hrs to complete in full.

Utility files:
- `candidates.py`- functions for candidate generation
- `fe.py` - functions for feature engineering
- `lfm.py` - functions for lightfm embedding training and extraction of embeddings for FE, also script for LFM training.
- `preprocess.py` - functions as well as script for data processing. You need to run it if you want to run `notebook.py`
- `hm_utils.py` - utility functions
- `schema.py` - utility schema
- `tm.py` - configured timer for benchmark measurements
### Links
1. Competition: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
2. Used solution post: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324084
3. Repository with original code: https://github.com/ryowk/kaggle-h-and-m-personalized-fashion-recommendations

### Additional dependencies
If you want to use user embeddings during feature engineering (turned off by default) you will need:
- `lightfm`
- `scipy`

If you want to generate candidates using close OHE vectors (turned off by default) you will need:
- `faiss` - used to speed up this search

If you want to run `notebook.py`, which contains complete benchmark you will additionally need
- `matplotlib` for plotting
- `catboost` for model training
