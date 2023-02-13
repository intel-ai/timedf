from pathlib import Path

from joblib import delayed, Parallel
import pandas as pd
import numpy as np

from tqdm import tqdm

from utils.pandas_backend import pd

from optiver_vol.optiver_utils import print_trace, tm, get_workdir_paths, DATA_DIR


def load_data(stock_id: int, stem: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f'{stem}_train.parquet' / f'stock_id={stock_id}')


def load_all_stocks(stock_ids, stem):
    dfs = []
    for stock_id in tqdm(stock_ids):
        df = load_data(stock_id, stem)
        df['stock_id'] = stock_id
        dfs.append(df)
    return pd.concat(dfs, axis=0)


def calc_wap1(df: pd.DataFrame) -> pd.Series:
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap


def calc_wap2(df: pd.DataFrame) -> pd.Series:
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap


def realized_volatility(series):
    return np.sqrt(np.sum(series**2))


def log_return(series: np.ndarray):
    return np.log(series).diff()


def log_return_df2(series: np.ndarray):
    return np.log(series).diff(2)


def flatten_name(prefix, src_names):
    ret = []
    for c in src_names:
        if c[0] in ['time_id', 'stock_id']:
            ret.append(c[0])
        else:
            ret.append('.'.join([prefix] + list(c)))
    return ret


def make_book_feature(stock_ids):
    gb_cols = ['stock_id', 'time_id']

    with tm.timeit('01-load_books'):
        book = load_all_stocks(stock_ids, 'book')
    
    with tm.timeit('02-wap'):
        book['wap1'] = calc_wap1(book)
        book['wap2'] = calc_wap2(book)

    with tm.timeit('03-groupby_return'):
        book['log_return1'] = book.groupby(gb_cols, group_keys=False)['wap1'].apply(log_return)
        book['log_return2'] = book.groupby(gb_cols, group_keys=False)['wap2'].apply(log_return)
        book['log_return_ask1'] = book.groupby(gb_cols, group_keys=False)['ask_price1'].apply(log_return)
        book['log_return_ask2'] = book.groupby(gb_cols, group_keys=False)['ask_price2'].apply(log_return)
        book['log_return_bid1'] = book.groupby(gb_cols, group_keys=False)['bid_price1'].apply(log_return)
        book['log_return_bid2'] = book.groupby(gb_cols, group_keys=False)['bid_price2'].apply(log_return)

    with tm.timeit('04-features'):
        book['wap_balance'] = abs(book['wap1'] - book['wap2'])
        book['price_spread'] = (book['ask_price1'] - book['bid_price1']) / ((book['ask_price1'] + book['bid_price1']) / 2)
        book['bid_spread'] = book['bid_price1'] - book['bid_price2']
        book['ask_spread'] = book['ask_price1'] - book['ask_price2']
        book['total_volume'] = (book['ask_size1'] + book['ask_size2']) + (book['bid_size1'] + book['bid_size2'])
        book['volume_imbalance'] = abs((book['ask_size1'] + book['ask_size2']) - (book['bid_size1'] + book['bid_size2']))
    
    features = {
        'seconds_in_bucket': ['count'],
        'wap1': [np.sum, np.mean, np.std],
        'wap2': [np.sum, np.mean, np.std],
        'log_return1': [np.sum, realized_volatility, np.mean, np.std],
        'log_return2': [np.sum, realized_volatility, np.mean, np.std],
        'log_return_ask1': [np.sum, realized_volatility, np.mean, np.std],
        'log_return_ask2': [np.sum, realized_volatility, np.mean, np.std],
        'log_return_bid1': [np.sum, realized_volatility, np.mean, np.std],
        'log_return_bid2': [np.sum, realized_volatility, np.mean, np.std],
        'wap_balance': [np.sum, np.mean, np.std],
        'price_spread':[np.sum, np.mean, np.std],
        'bid_spread':[np.sum, np.mean, np.std],
        'ask_spread':[np.sum, np.mean, np.std],
        'total_volume':[np.sum, np.mean, np.std],
        'volume_imbalance':[np.sum, np.mean, np.std]
    }
    
    with tm.timeit('05-groupby_features'):
        agg = book.groupby(gb_cols).agg(features).reset_index(drop=False)
        agg.columns = flatten_name('book', agg.columns)
    
    with tm.timeit('06-groupby_time_buckets'):
        for time in [450, 300, 150]:
            d = book[book['seconds_in_bucket'] >= time].groupby(gb_cols).agg(features).reset_index(drop=False)
            d.columns = flatten_name(f'book_{time}', d.columns)
            agg = pd.merge(agg, d, on=gb_cols, how='left')
    return agg


def make_trade_feature(stock_ids):
    gb_cols = ['stock_id', 'time_id']

    with tm.timeit('01-load_trade'):
        trade = load_all_stocks(stock_ids, 'trade')

    with tm.timeit('02-groupby_return'):
        trade['log_return'] = trade.groupby(gb_cols, group_keys=False)['price'].apply(log_return)

    features = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':['count'],
        'size':[np.sum],
        'order_count':[np.mean],
    }

    with tm.timeit('03-groupby_features'):
        agg = trade.groupby(gb_cols).agg(features).reset_index()
        agg.columns = flatten_name('trade', agg.columns)
    
    with tm.timeit('04-groupby_time_buckets'):
        for time in [450, 300, 150]:
            d = trade[trade['seconds_in_bucket'] >= time].groupby(gb_cols).agg(features).reset_index(drop=False)
            d.columns = flatten_name(f'trade_{time}', d.columns)
            agg = pd.merge(agg, d, on=gb_cols, how='left')
    return agg


def make_book_feature_v2(stock_ids):
    gb_cols = ['stock_id', 'time_id']

    with tm.timeit('01-load_book'):
        book = load_all_stocks(stock_ids, 'book')

    with tm.timeit('02-prices'):
        prices = book[['stock_id', 'time_id', *['bid_price1', 'ask_price1', 'bid_price2', 'ask_price2']]]
    
    with tm.timeit('03-tick_size'):
        def find_smallest_spread(df: pd.DataFrame):
            """This looks like we want to find the smallest difference between prices at a given
             time. So it's like the smallest spread."""
            try:
                price_list = df.values.flatten()
                return min(np.diff(sorted(set(price_list))))
            except Exception:
                print_trace(str(df[gb_cols].iloc[0]))
                return np.nan

        ticks = prices.groupby(gb_cols).apply(find_smallest_spread)
        import pdb
        pdb.set_trace()
        ticks.name = 'tick_size'
        ticks.to_dataframe()
        
    return ticks


def make_features(base):


    return df


def make_features_v2(base):
    stock_ids = set(base['stock_id'])
    book_v2 = make_book_feature_v2(stock_ids)
    return pd.merge(base, book_v2, on=['stock_id', 'time_id'], how='left')


def preprocess(raw_data_path: Path, preprocessed_path: Path):
    with tm.timeit('01-train'):
        with tm.timeit('01-load_train'):
            train = pd.read_csv(raw_data_path / 'train.csv')

        with tm.timeit('02-stock_ids'):
            stock_ids = set(train['stock_id'])
        
        with tm.timeit('03-books'):
            book = make_book_feature(stock_ids)

        with tm.timeit('04-trades'):
            trade = make_trade_feature(stock_ids)

        with tm.timeit('05-books_v2'):
            book_v2 = make_book_feature_v2(stock_ids)
            
        with tm.timeit('06-merge features'):
            df = pd.merge(train, book, on=['stock_id', 'time_id'], how='left')
            df = pd.merge(df, trade, on=['stock_id', 'time_id'], how='left')
            df = pd.merge(df, book_v2, on=['stock_id', 'time_id'], how='left')
    
    # Use copy of training data as test data to imitate 2nd stage RAM usage.
    MEMORY_TEST_MODE = True
    if MEMORY_TEST_MODE:
        with tm.timeit('02-test generation'):
            test_df = df.iloc[:170000].copy()
            test_df['time_id'] += 32767
            test_df['row_id'] = ''

            df = pd.concat([df, test_df.drop('row_id', axis=1)]).reset_index(drop=True)

    df.to_feather(preprocessed_path)  # save cache


if __name__ == '__main__':
    paths = get_workdir_paths()
    preprocess(DATA_DIR, paths['preprocessed'])    
    print(tm.get_results())
