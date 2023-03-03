# GPu reqs
# !pip -q install ../input/pytorchtabnet/pytorch_tabnet-2.0.1-py3-none-any.whl

import gc
from typing import Dict, List, Optional
import pickle

import numpy as np
import scipy.linalg as lin  # needed for matrix inversion
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale

from omniscripts.pandas_backend import pd

from .optiver_utils import print_trace, tm


# ### Nearest-Neighbor Features
N_NEIGHBORS_MAX = 80

class Neighbors:
    def __init__(
        self,
        name: str,
        pivot: pd.DataFrame,
        p: float,
        metric: str = "minkowski",
        metric_params: Optional[Dict] = None,
        exclude_self: bool = False,
    ):
        self.name = name
        self.exclude_self = exclude_self
        self.p = p
        self.metric = metric

        nn = NearestNeighbors(
            n_neighbors=N_NEIGHBORS_MAX, p=p, metric=metric, metric_params=metric_params
        )
        with tm.timeit("knn_train"):
            nn.fit(pivot)

        with tm.timeit("knn_query"):
            _, self.neighbors = nn.kneighbors(pivot, return_distance=True)

        self.columns = self.index = self.feature_values = self.feature_col = None

    def rearrange_feature_values(self, df: pd.DataFrame, feature_col: str) -> None:
        raise NotImplementedError()

    def make_nn_feature(self, n=5, agg=np.mean) -> pd.DataFrame:
        assert self.feature_values is not None, "should call rearrange_feature_values beforehand"

        start = 1 if self.exclude_self else 0

        pivot_aggs = pd.DataFrame(
            agg(self.feature_values[start:n, :, :], axis=0), columns=self.columns, index=self.index
        )

        dst = pivot_aggs.unstack().reset_index()
        dst.columns = [
            "stock_id",
            "time_id",
            f"{self.feature_col}_nn{n}_{self.name}_{agg.__name__}",
        ]
        return dst


class TimeIdNeighbors(Neighbors):
    def rearrange_feature_values(self, df: pd.DataFrame, feature_col: str) -> None:
        with tm.timeit("01-feature pivot"):
            feature_pivot = df.pivot(index="time_id", columns="stock_id", values=feature_col)
            feature_pivot = feature_pivot.fillna(feature_pivot.mean())

        with tm.timeit("02-feature values"):
            feature_values = np.zeros((N_NEIGHBORS_MAX, *feature_pivot.shape))

            for i in range(N_NEIGHBORS_MAX):
                feature_values[i, :, :] += feature_pivot.values[self.neighbors[:, i], :]

        self.columns = list(feature_pivot.columns)
        self.index = list(feature_pivot.index)
        self.feature_values = feature_values
        self.feature_col = feature_col

    def __repr__(self) -> str:
        return f"time-id NN (name={self.name}, metric={self.metric}, p={self.p})"


class StockIdNeighbors(Neighbors):
    def rearrange_feature_values(self, df: pd.DataFrame, feature_col: str) -> None:
        """stock-id based nearest neighbor features"""
        with tm.timeit("01-feature_pivot"):
            feature_pivot = df.pivot(index="time_id", columns="stock_id", values=feature_col)
            feature_pivot = feature_pivot.fillna(feature_pivot.mean())

        with tm.timeit("02-feature_values"):
            feature_values = np.zeros((N_NEIGHBORS_MAX, *feature_pivot.shape))

            for i in range(N_NEIGHBORS_MAX):
                feature_values[i, :, :] += feature_pivot.values[:, self.neighbors[:, i]]

        self.columns = list(feature_pivot.columns)
        self.index = list(feature_pivot.index)
        self.feature_values = feature_values
        self.feature_col = feature_col

    def __repr__(self) -> str:
        return f"stock-id NN (name={self.name}, metric={self.metric}, p={self.p})"


# #### Build Nearest Neighbors
def train_nearest_neighbors(df):
    time_id_neighbors: List[Neighbors] = []
    stock_id_neighbors: List[Neighbors] = []

    with tm.timeit("01-df_pv"):
        df_pv = df[["stock_id", "time_id"]].copy()
        df_pv["price"] = 0.01 / df["tick_size"]
        df_pv["vol"] = df["book.log_return1.realized_volatility"]
        df_pv["trade.tau"] = df["trade.tau"]
        df_pv["trade.size.sum"] = df["book.total_volume.sum"]

    with tm.timeit("02-price_features"):
        with tm.timeit("01-df_pv"):
            # Price features
            pivot = df_pv.pivot(index="time_id", columns="stock_id", values="price")
            pivot = pivot.fillna(pivot.mean())
            pivot = pd.DataFrame(minmax_scale(pivot))

        with tm.timeit("02-time_id_neighbors_time_price_c"):
            time_id_neighbors.append(
                TimeIdNeighbors("time_price_c", pivot, p=2, metric="canberra", exclude_self=True)
            )
        with tm.timeit("03-time_id_neighbors_time_price_m"):
            with tm.timeit("matrix_inversion_problem"):
                vi = lin.inv(np.cov(pivot.values))

            time_id_neighbors.append(
                TimeIdNeighbors(
                    "time_price_m",
                    pivot,
                    p=2,
                    metric="mahalanobis",
                    # metric_params={'V':np.cov(pivot.values.T)}
                    metric_params={"VI": vi},
                )
            )
        with tm.timeit("04-stock_id_neighbors"):
            stock_id_neighbors.append(
                StockIdNeighbors(
                    "stock_price_l1", minmax_scale(pivot.transpose()), p=1, exclude_self=True
                )
            )

    with tm.timeit("03-volume_features"):
        with tm.timeit("01-pivot"):
            pivot = df_pv.pivot(index="time_id", columns="stock_id", values="vol")
            pivot = pivot.fillna(pivot.mean())
            pivot = pd.DataFrame(minmax_scale(pivot))

        with tm.timeit("02-time_id_neighbors"):
            time_id_neighbors.append(TimeIdNeighbors("time_vol_l1", pivot, p=1))

        with tm.timeit("03-stock_id_neighbors"):
            stock_id_neighbors.append(
                StockIdNeighbors(
                    "stock_vol_l1", minmax_scale(pivot.transpose()), p=1, exclude_self=True
                )
            )

    with tm.timeit("04-size_features"):
        with tm.timeit("01-pivot"):
            # size nn features
            pivot = df_pv.pivot(index="time_id", columns="stock_id", values="trade.size.sum")
            pivot = pivot.fillna(pivot.mean())
            pivot = pd.DataFrame(minmax_scale(pivot))

        with tm.timeit("02-time_id_neighbors_time_size_m"):
            with tm.timeit("matrix_inversion_problem"):
                vi = lin.inv(np.cov(pivot.values))
            time_id_neighbors.append(
                TimeIdNeighbors(
                    "time_size_m",
                    pivot,
                    p=2,
                    metric="mahalanobis",
                    metric_params={"VI": vi},
                )
            )
        with tm.timeit("03-time_id_neighbors_time_size_c"):
            time_id_neighbors.append(TimeIdNeighbors("time_size_c", pivot, p=2, metric="canberra"))
    return df_pv, pivot, time_id_neighbors, stock_id_neighbors


def normalize_rank(df):
    # features with large changes over time are converted to relative ranks within time-id
    df["trade.order_count.mean"] = df.groupby("time_id")["trade.order_count.mean"].rank()
    df["book.total_volume.sum"] = df.groupby("time_id")["book.total_volume.sum"].rank()
    df["book.total_volume.mean"] = df.groupby("time_id")["book.total_volume.mean"].rank()
    df["book.total_volume.std"] = df.groupby("time_id")["book.total_volume.std"].rank()

    df["trade.tau"] = df.groupby("time_id")["trade.tau"].rank()

    for dt in [150, 300, 450]:
        df[f"book_{dt}.total_volume.sum"] = df.groupby("time_id")[
            f"book_{dt}.total_volume.sum"
        ].rank()
        df[f"book_{dt}.total_volume.mean"] = df.groupby("time_id")[
            f"book_{dt}.total_volume.mean"
        ].rank()
        df[f"book_{dt}.total_volume.std"] = df.groupby("time_id")[
            f"book_{dt}.total_volume.std"
        ].rank()
        df[f"trade_{dt}.order_count.mean"] = df.groupby("time_id")[
            f"trade_{dt}.order_count.mean"
        ].rank()


def make_nearest_neighbor_feature(
    df: pd.DataFrame, stock_id_neighbors, time_id_neighbors
) -> pd.DataFrame:
    df2 = df.copy()

    feature_cols_stock = {
        "book.log_return1.realized_volatility": [np.mean, np.min, np.max, np.std],
        "trade.seconds_in_bucket.count": [np.mean],
        "trade.tau": [np.mean],
        "trade_150.tau": [np.mean],
        "book.tau": [np.mean],
        "trade.size.sum": [np.mean],
        "book.seconds_in_bucket.count": [np.mean],
    }

    feature_cols = {
        "book.log_return1.realized_volatility": [np.mean, np.min, np.max, np.std],
        "real_price": [np.max, np.mean, np.min],
        "trade.seconds_in_bucket.count": [np.mean],
        "trade.tau": [np.mean],
        "trade.size.sum": [np.mean],
        "book.seconds_in_bucket.count": [np.mean],
        "trade_150.tau_nn20_stock_vol_l1_mean": [np.mean],
        "trade.size.sum_nn20_stock_vol_l1_mean": [np.mean],
    }

    time_id_neigbor_sizes = [3, 5, 10, 20, 40]
    time_id_neigbor_sizes_vol = [2, 3, 5, 10, 20, 40]
    stock_id_neighbor_sizes = [10, 20, 40]

    ndf: Optional[pd.DataFrame] = None

    def _add_ndf(ndf: Optional[pd.DataFrame], dst: pd.DataFrame) -> pd.DataFrame:
        if ndf is None:
            return dst
        else:
            ndf[dst.columns[-1]] = dst[dst.columns[-1]].astype(np.float32)
            return ndf

    # neighbor stock_id
    with tm.timeit('01-stock_id_nn'):
        for feature_col in feature_cols_stock.keys():
            with tm.timeit(f'feature={feature_col}'):
                if feature_col not in df2.columns:
                    print(f"column {feature_col} is skipped")
                    continue

                if not stock_id_neighbors:
                    continue

                for nn in stock_id_neighbors:
                    with tm.timeit(f'rearrange_feature_values {nn}'):
                        nn.rearrange_feature_values(df2, feature_col)

                for agg in feature_cols_stock[feature_col]:
                    for n in stock_id_neighbor_sizes:
                        with tm.timeit(f'make_nn_feature agg={agg} nn={nn}'):
                            for nn in stock_id_neighbors:
                                dst = nn.make_nn_feature(n, agg)
                                ndf = _add_ndf(ndf, dst)
                    
    if ndf is not None:
        df2 = pd.merge(df2, ndf, on=["time_id", "stock_id"], how="left")
    ndf = None

    # neighbor time_id
    
    with tm.timeit('02-time_id_nn'):
        for feature_col in feature_cols.keys():
            with tm.timeit('nn_time_id_feature'):
                if feature_col not in df2.columns:
                    print(f"column {feature_col} is skipped")
                    continue

                for nn in time_id_neighbors:
                    with tm.timeit(f'rearrange_feature_values {nn}'):
                        nn.rearrange_feature_values(df2, feature_col)

                if "volatility" in feature_col:
                    time_id_ns = time_id_neigbor_sizes_vol
                else:
                    time_id_ns = time_id_neigbor_sizes

                for agg in feature_cols[feature_col]:
                    for n in time_id_ns:
                        with tm.timeit(f'make_nn_feature agg={agg} nn={nn}'):
                            for nn in time_id_neighbors:
                                dst = nn.make_nn_feature(n, agg)
                                ndf = _add_ndf(ndf, dst)

    with tm.timeit('03-merge'):
        if ndf is not None:
            df2 = pd.merge(df2, ndf, on=["time_id", "stock_id"], how="left")

    # features further derived from nearest neighbor features
    with tm.timeit('04-nn_extra_features'):
        for sz in time_id_neigbor_sizes:
            denominator = f"real_price_nn{sz}_time_price_c"

            df2[f"real_price_rankmin_{sz}"] = df2["real_price"] / df2[f"{denominator}_amin"]
            df2[f"real_price_rankmax_{sz}"] = df2["real_price"] / df2[f"{denominator}_amax"]
            df2[f"real_price_rankmean_{sz}"] = df2["real_price"] / df2[f"{denominator}_mean"]

        for sz in time_id_neigbor_sizes_vol:
            denominator = f"book.log_return1.realized_volatility_nn{sz}_time_price_c"

            df2[f"vol_rankmin_{sz}"] = (
                df2["book.log_return1.realized_volatility"] / df2[f"{denominator}_amin"]
            )
            df2[f"vol_rankmax_{sz}"] = (
                df2["book.log_return1.realized_volatility"] / df2[f"{denominator}_amax"]
            )

        price_cols = [c for c in df2.columns if "real_price" in c and "rank" not in c]
        for c in price_cols:
            del df2[c]

        for sz in time_id_neigbor_sizes_vol:
            tgt = f"book.log_return1.realized_volatility_nn{sz}_time_price_m_mean"
            df2[f"{tgt}_rank"] = df2.groupby("time_id")[tgt].rank()
    return df2


def skew_correction(df2):
    """Skew correction for NN"""
    cols_to_log = [
        "trade.size.sum",
        "trade_150.size.sum",
        "trade_300.size.sum",
        "trade_450.size.sum",
        "volume_imbalance",
    ]
    for c in df2.columns:
        for check in cols_to_log:
            try:
                if check in c:
                    df2[c] = np.log(df2[c] + 1)
                    break
            except Exception:
                print_trace("log1p")


def rolling_average(df2):
    """Rolling average of RV for similar trading volume"""
    try:
        df2.sort_values(by=["stock_id", "book.total_volume.sum"], inplace=True)
        df2.reset_index(drop=True, inplace=True)

        roll_target = "book.log_return1.realized_volatility"

        for window_size in [3, 10]:
            df2[f"realized_volatility_roll{window_size}_by_book.total_volume.mean"] = (
                df2.groupby("stock_id")[roll_target]
                .rolling(window_size, center=True, min_periods=1)
                .mean()
                .reset_index()
                .sort_values(by=["level_1"])[roll_target]
                .values
            )
    except Exception:
        print_trace("mean RV")


def stock_id_embeddings(df2, pivot, df_pv):
    """stock-id embedding (helps little)"""
    try:
        lda_n = 3
        lda = LatentDirichletAllocation(n_components=lda_n, random_state=0)

        with tm.timeit("LDA train"):
            stock_id_emb = pd.DataFrame(
                lda.fit_transform(pivot.transpose()),
                index=df_pv.pivot("time_id", "stock_id", "vol").columns,
            )

        for i in range(lda_n):
            df2[f"stock_id_emb{i}"] = df2["stock_id"].map(stock_id_emb[i])
    except Exception:
        print_trace("LDA")


def fe(preprocessed_path):
    with tm.timeit("01-load_preprocessed"):
        df = pd.read_feather(preprocessed_path)

    with tm.timeit("02-tau_fe"):
        # the tau itself is meaningless for GBDT, but useful as input to aggregate in Nearest Neighbor features
        df["trade.tau"] = np.sqrt(1 / df["trade.seconds_in_bucket.count"])
        df["trade_150.tau"] = np.sqrt(1 / df["trade_150.seconds_in_bucket.count"])
        df["book.tau"] = np.sqrt(1 / df["book.seconds_in_bucket.count"])
        df["real_price"] = 0.01 / df["tick_size"]

    with tm.timeit("03-knn_fit"):
        df_pv, pivot, time_id_neighbors, stock_id_neighbors = train_nearest_neighbors(df)

    with tm.timeit("04-normalize rank"):
        normalize_rank(df)

    with tm.timeit("05-make_nearest_neighbor_feature"):
        df2 = make_nearest_neighbor_feature(df, stock_id_neighbors, time_id_neighbors)

    with tm.timeit("06-extra features for df2"):
        skew_correction(df2)
        rolling_average(df2)
        stock_id_embeddings(df2, pivot, df_pv)

    return df2


# ## Reverse Engineering time-id Order & Make CV Split
def calc_price2(df):
    tick = sorted(np.diff(sorted(np.unique(df.values.flatten()))))[0]
    return 0.01 / tick


def calc_prices(raw_data_path):
    df = pd.read_parquet(
        raw_data_path / "book_train.parquet",
        columns=["stock_id", "time_id", "ask_price1", "ask_price2", "bid_price1", "bid_price2"],
    )
    return df.groupby(["stock_id", "time_id"]).apply(calc_price2).to_frame("price").reset_index()


def sort_manifold(df, clf):
    df_ = df.set_index("time_id")
    df_ = pd.DataFrame(minmax_scale(df_.fillna(df_.mean())))

    X_compoents = clf.fit_transform(df_)
    return np.argsort(X_compoents[:, 0]), X_compoents


def reconstruct_time_id_order(raw_data_path):
    with tm.timeit("calc prices"):
        df_prices = calc_prices(raw_data_path)
        df_prices = df_prices.pivot(index="time_id", columns="stock_id", values="price")
        df_prices.columns = [f"stock_id={i}" for i in df_prices.columns]
        df_prices = df_prices.reset_index(drop=False)

    with tm.timeit("t-SNE(400) -> 50 (ML-heavy)"):
        clf = TSNE(n_components=1, perplexity=400, random_state=0, n_iter=2000)
        order, X_compoents = sort_manifold(df_prices, clf)

        clf = TSNE(
            n_components=1,
            perplexity=50,
            random_state=0,
            init=X_compoents,
            n_iter=2000,
            method="exact",
        )
        order, X_compoents = sort_manifold(df_prices, clf)

        df_ordered = df_prices.reindex(order).reset_index(drop=True)
        if df_ordered["stock_id=61"].iloc[0] > df_ordered["stock_id=61"].iloc[-1]:
            df_ordered = df_ordered.reindex(df_ordered.index[::-1]).reset_index(drop=True)

    return df_ordered[["time_id"]]


def perform_split(df_train, raw_data_path):
    with tm.timeit("calculate order of time-id"):
        timeid_order = reconstruct_time_id_order(raw_data_path)

    with tm.timeit("make folds"):
        timeid_order["time_id_order"] = np.arange(len(timeid_order))
        df_train["time_id_order"] = df_train["time_id"].map(
            timeid_order.set_index("time_id")["time_id_order"]
        )
        df_train = df_train.sort_values(["time_id_order", "stock_id"]).reset_index(drop=True)

        folds_border = [3830 - 383 * 4, 3830 - 383 * 3, 3830 - 383 * 2, 3830 - 383 * 1]
        time_id_orders = df_train["time_id_order"]

        folds = []
        for i, border in enumerate(folds_border):
            idx_train = np.where(time_id_orders < border)[0]
            idx_valid = np.where((border <= time_id_orders) & (time_id_orders < border + 383))[0]
            folds.append((idx_train, idx_valid))

        del df_train["time_id_order"]

    return folds


def prepare_dataset(paths):
    with tm.timeit("01-fe"):
        df2 = fe(paths["preprocessed"])

    with tm.timeit("02-train_test_split"):
        df_train = df2[~df2.target.isnull()].copy()
        df_test = df2[df2.target.isnull()].copy()

        folds = perform_split(df_train, raw_data_path=paths["raw_data"])

        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

    with tm.timeit("03-save results"):
        df_train.to_feather(paths["train"])
        df_test.to_feather(paths["test"])
        with open(paths["folds"], "wb") as f:
            pickle.dump(folds, f)
