"""
- article_id, category_idを含めた全てのカテゴリを0-indexedな連番に変換する(_idxがついたカラムが追加される)
- None, 1のみのカテゴリを0, 1に変換する(カラムは上書きされる)
- 1, 2のみのカテゴリを0, 1に変換する(カラムは上書きされる)
"""
from __future__ import annotations
from pathlib import Path

from typing import Any

import logging

from omniscripts import tm
from omniscripts.pandas_backend import pd

from . import schema


logger = logging.getLogger(__name__)


def transform_data(input_data_path: Path, result_path: Path):
    """
    - Convert all categories including article_id and category_id to 0-indexed serial numbers (column with _idx is added)
    - Convert categories with only None, 1 to 0, 1 (columns are overwritten)
    - convert 1, 2 only categories to 0, 1 (columns are overwritten)
    """

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

    # Assign numbers in order of frequency
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
    # The last week included in transactions_train is set to 0, the week before is 1 etc.
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
        with tm.timeit(str(c)):
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


def run_complete_preprocessing(raw_data_path: Path, paths, n_weeks, use_lfm=False):
    transform_data(input_data_path=raw_data_path, result_path=paths["preprocessed_data"])

    for week in range(n_weeks + 1):
        create_user_ohe_agg(
            week,
            preprocessed_data_path=paths["preprocessed_data"],
            result_path=paths["user_features"],
        )

    if use_lfm:
        from .lfm import train_lfm

        for week in range(1, n_weeks + 1):
            train_lfm(week=week, lfm_features_path=paths["lfm_features"])
