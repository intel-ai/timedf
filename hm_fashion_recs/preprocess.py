"""
- article_id, category_idを含めた全てのカテゴリを0-indexedな連番に変換する(_idxがついたカラムが追加される)
- None, 1のみのカテゴリを0, 1に変換する(カラムは上書きされる)
- 1, 2のみのカテゴリを0, 1に変換する(カラムは上書きされる)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# import modin.pandas as pd
import logging

from utils.pandas_backend import pd

import hm_fashion_recs.schema as schema
from hm_fashion_recs.tm import tm
from hm_fashion_recs.lfm import train_lfm

from hm_fashion_recs.vars import (
    n_weeks,
    raw_data_path,
    preprocessed_data_path,
    user_features_path,
    lfm_features_path,
    dim,
)


logger = logging.getLogger(__name__)


def transform_data(input_data_path, result_path):
    def _count_encoding_dict(df: pd.DataFrame, col_name: str) -> dict[Any, int]:
        v = (
            df.groupby(col_name)
            .size()
            .reset_index(name="size")
            .sort_values(by="size", ascending=False)[col_name]
            .tolist()
        )
        return {x: i for i, x in enumerate(v)}

    def _dict_to_dataframe(mp: dict[Any, int]) -> pd.DataFrame:
        return pd.DataFrame(mp.items(), columns=["val", "idx"])

    def _add_idx_column(
        df: pd.DataFrame, col_name_from: str, col_name_to: str, mp: dict[Any, int]
    ):
        df[col_name_to] = df[col_name_from].apply(lambda x: mp[x]).astype("int64")

    logger.info("start reading dataframes")
    articles = pd.read_csv(input_data_path / "articles.csv", dtype=schema.ARTICLES_ORIGINAL)
    customers = pd.read_csv(input_data_path / "customers.csv", dtype=schema.CUSTOMERS_ORIGINAL)
    transactions = pd.read_csv(
        input_data_path / "transactions_train.csv",
        dtype=schema.TRANSACTIONS_ORIGINAL,
        parse_dates=["t_dat"],
    )

    (result_path / "images").mkdir(exist_ok=True, parents=True)

    # customer_id
    logger.info("start processing customer_id")
    customer_ids = customers.customer_id.values
    mp_customer_id = {x: i for i, x in enumerate(customer_ids)}
    _dict_to_dataframe(mp_customer_id).to_pickle(result_path / "mp_customer_id.pkl")

    # article_id
    logger.info("start processing article_id")
    article_ids = articles.article_id.values
    mp_article_id = {x: i for i, x in enumerate(article_ids)}
    _dict_to_dataframe(mp_article_id).to_pickle(result_path / "mp_article_id.pkl")

    ################
    # customers
    ################
    logger.info("start processing customers")
    _add_idx_column(customers, "customer_id", "user", mp_customer_id)
    # (None, 1) -> (0, 1)
    customers["FN"] = customers["FN"].fillna(0).astype("int64")
    customers["Active"] = customers["Active"].fillna(0).astype("int64")

    # 頻度順に番号を振る(既にintなものも連番のほうが都合が良いので振り直す)
    customers["club_member_status"] = customers["club_member_status"].fillna("NULL")
    customers["fashion_news_frequency"] = customers["fashion_news_frequency"].fillna("NULL")
    count_encoding_columns = ["club_member_status", "fashion_news_frequency"]
    for col_name in count_encoding_columns:
        mp = _count_encoding_dict(customers, col_name)
        _add_idx_column(customers, col_name, f"{col_name}_idx", mp)
    customers.to_pickle(result_path / "users.pkl")

    ################
    # articles
    ################
    logger.info("start processing articles")
    _add_idx_column(articles, "article_id", "item", mp_article_id)
    count_encoding_columns = [
        "product_type_no",
        "product_group_name",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_value_id",
        "perceived_colour_master_id",
        "department_no",
        "index_code",
        "index_group_no",
        "section_no",
        "garment_group_no",
    ]
    for col_name in count_encoding_columns:
        mp = _count_encoding_dict(articles, col_name)
        _add_idx_column(articles, col_name, f"{col_name}_idx", mp)
    articles.to_pickle(result_path / "items.pkl")

    ################
    # transactions
    ################
    logger.info("start processing transactions")
    _add_idx_column(transactions, "customer_id", "user", mp_customer_id)
    _add_idx_column(transactions, "article_id", "item", mp_article_id)
    # (1, 2) -> (0, 1)
    transactions["sales_channel_id"] = transactions["sales_channel_id"] - 1
    # transactions_trainに含まれる最後の1週間を0として、過去に行くに連れてインクリメント
    transactions["week"] = (transactions["t_dat"].max() - transactions["t_dat"]).dt.days // 7
    transactions["day"] = (transactions["t_dat"].max() - transactions["t_dat"]).dt.days
    transactions.to_pickle(result_path / "transactions_train.pkl")


def create_user_ohe_agg(week, preprocessed_data_path, result_path):
    result_path.mkdir(exist_ok=True, parents=True)

    transactions = pd.read_pickle(preprocessed_data_path / "transactions_train.pkl")[
        ["user", "item", "week"]
    ]
    users = pd.read_pickle(preprocessed_data_path / "users.pkl")
    items = pd.read_pickle(preprocessed_data_path / "items.pkl")

    # used to be vaex
    tr = transactions.query(f"week >= {week}")[["user", "item"]]

    target_columns = [c for c in items.columns if c.endswith("_idx")]
    for c in target_columns:
        timer = tm.timeit(str(c))
        save_path = result_path / f"user_ohe_agg_week{week}_{c}.pkl"

        # used to be vaex
        right = pd.get_dummies(items[["item", c]], columns=[c])

        tmp = pd.merge(tr, right, on="item")
        tmp = tmp.drop(columns="item")

        tmp = tmp.groupby("user").agg("mean")

        # used to be vaex
        users = users[["user"]].join(tmp, on="user", how="left")
        users = users.rename(
            columns={c: f"user_ohe_agg_{c}" for c in users.columns if c != "user"}
        )

        users = users.sort_values(by="user").reset_index(drop=True)
        users.to_pickle(save_path)
        print("saved", save_path)
        timer.stop()


def main():
    use_lfm = False

    transform_data(input_data_path=raw_data_path, result_path=preprocessed_data_path)

    for week in range(n_weeks + 1):
        create_user_ohe_agg(
            week, preprocessed_data_path=preprocessed_data_path, result_path=user_features_path
        )

    if use_lfm:

        for week in range(1, n_weeks + 1):
            train_lfm(week=week, lfm_features_path=lfm_features_path, dim=dim)


if __name__ == "__main__":
    main()
