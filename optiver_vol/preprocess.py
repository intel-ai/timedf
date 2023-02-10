from pathlib import Path

from joblib import delayed, Parallel
import pandas as pd
import numpy as np

from utils.pandas_backend import pd

from optiver_vol.optiver_utils import print_trace, tm, get_workdir_paths, DATA_DIR


def load_data(stock_id: int, stem: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f'{stem}_train.parquet' / f'stock_id={stock_id}')


def load_book(stock_id: int) -> pd.DataFrame:
    return load_data(stock_id, 'book')


def load_trade(stock_id: int) -> pd.DataFrame:
    return load_data(stock_id, 'trade')


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


def make_book_feature(stock_id):
    book = load_book(stock_id)

    book['wap1'] = calc_wap1(book)
    book['wap2'] = calc_wap2(book)
    book['log_return1'] = book.groupby(['time_id'], group_keys=False)['wap1'].apply(log_return)
    book['log_return2'] = book.groupby(['time_id'], group_keys=False)['wap2'].apply(log_return)
    book['log_return_ask1'] = book.groupby(['time_id'], group_keys=False)['ask_price1'].apply(log_return)
    book['log_return_ask2'] = book.groupby(['time_id'], group_keys=False)['ask_price2'].apply(log_return)
    book['log_return_bid1'] = book.groupby(['time_id'], group_keys=False)['bid_price1'].apply(log_return)
    book['log_return_bid2'] = book.groupby(['time_id'], group_keys=False)['bid_price2'].apply(log_return)

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
    
    agg = book.groupby('time_id').agg(features).reset_index(drop=False)
    agg.columns = flatten_name('book', agg.columns)
    agg['stock_id'] = stock_id
    
    for time in [450, 300, 150]:
        d = book[book['seconds_in_bucket'] >= time].groupby('time_id').agg(features).reset_index(drop=False)
        d.columns = flatten_name(f'book_{time}', d.columns)
        agg = pd.merge(agg, d, on='time_id', how='left')
    return agg


def make_trade_feature(stock_id):
    trade = load_trade(stock_id)
    trade['log_return'] = trade.groupby('time_id', group_keys=False)['price'].apply(log_return)

    features = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':['count'],
        'size':[np.sum],
        'order_count':[np.mean],
    }

    agg = trade.groupby('time_id').agg(features).reset_index()
    agg.columns = flatten_name('trade', agg.columns)
    agg['stock_id'] = stock_id
        
    for time in [450, 300, 150]:
        d = trade[trade['seconds_in_bucket'] >= time].groupby('time_id').agg(features).reset_index(drop=False)
        d.columns = flatten_name(f'trade_{time}', d.columns)
        agg = pd.merge(agg, d, on='time_id', how='left')
    return agg


def make_book_feature_v2(stock_id):
    book = load_book(stock_id)

    prices = book.set_index('time_id')[['bid_price1', 'ask_price1', 'bid_price2', 'ask_price2']]
    time_ids = list(set(prices.index))

    ticks = {}
    for tid in time_ids:
        try:
            price_list = prices.loc[tid].values.flatten()
            price_diff = sorted(np.diff(sorted(set(price_list))))
            ticks[tid] = price_diff[0]
        except Exception:
            print_trace(f'tid={tid}')
            ticks[tid] = np.nan
        
    dst = pd.DataFrame()
    dst['time_id'] = np.unique(book['time_id'])
    dst['stock_id'] = stock_id
    dst['tick_size'] = dst['time_id'].map(ticks)

    return dst


def make_features(base):
    stock_ids = set(base['stock_id'])
    with tm.timeit('01-books'):
        books = Parallel(n_jobs=-1)(delayed(make_book_feature)(i,) for i in stock_ids)
        book = pd.concat(books)

    with tm.timeit('02-trades'):
        trades = Parallel(n_jobs=-1)(delayed(make_trade_feature)(i,) for i in stock_ids)
        trade = pd.concat(trades)

    with tm.timeit('03-extra features'):
        df = pd.merge(base, book, on=['stock_id', 'time_id'], how='left')
        df = pd.merge(df, trade, on=['stock_id', 'time_id'], how='left')
        #df = make_extra_features(df)

    return df


def make_features_v2(base):
    with tm.timeit('books-v2'):
        stock_ids = set(base['stock_id'])
        books = Parallel(n_jobs=-1)(delayed(make_book_feature_v2)(i,) for i in stock_ids)
        book_v2 = pd.concat(books)
        return pd.merge(base, book_v2, on=['stock_id', 'time_id'], how='left')


def preprocess(raw_data_path: Path, preprocessed_path: Path):
    with tm.timeit('01-train'):
        train = pd.read_csv(raw_data_path / 'train.csv')

        df = make_features(train)
        df = make_features_v2(df)
    
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
