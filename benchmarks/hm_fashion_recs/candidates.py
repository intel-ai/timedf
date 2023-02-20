from pathlib import Path
import logging

import numpy as np

from utils.pandas_backend import pd

from .tm import tm

logger = logging.getLogger(__name__)


class CFG:
    """Configuration for candidate generaton."""

    # These candidates are generated with faiss library, we turn them off by default
    # to avoid this dependency
    use_ohe_distance_candidates = False
    # If we do use faiss, should we try to use gpu or just cpu?
    use_faiss_gpu = False

    popular_num_items = 60
    popular_weeks = 1

    item2item_num_items = 24
    item2item_num_items_for_same_product_code = 12
    cooc_weeks = 32
    cooc_threshold = 150
    ohe_distance_num_items = 48
    ohe_distance_num_weeks = 20


##############
def create_candidates(
    *,
    transactions: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    age_shifts,
    target_users: np.ndarray,
    week: int,
    user_features_path: Path,
) -> pd.DataFrame:
    """
    transactions
        original transactions (user, item, week)
    target_users
        user for candidate generation
    week
        candidates are generated using only the information available until and including this week
    """
    logger.info(f"create candidates (week: {week})")
    assert len(target_users) == len(set(target_users))

    def create_candidates_repurchase(
        strategy: str,
        week_start: int,
        target_users: np.ndarray,
        max_items_per_user: int = 1234567890,
    ) -> pd.DataFrame:
        tr = transactions.query("user in @target_users and @week_start <= week")[
            ["user", "item", "week", "day"]
        ].drop_duplicates(ignore_index=True)

        gr_day = tr.groupby(["user", "item"])["day"].min().reset_index(name="day")
        gr_week = tr.groupby(["user", "item"])["week"].min().reset_index(name="week")
        gr_volume = tr.groupby(["user", "item"]).size().reset_index(name="volume")

        gr_day["day_rank"] = gr_day.groupby("user")["day"].rank()
        gr_week["week_rank"] = gr_week.groupby("user")["week"].rank()
        gr_volume["volume_rank"] = gr_volume.groupby("user")["volume"].rank(ascending=False)

        candidates = gr_day.merge(gr_week, on=["user", "item"]).merge(
            gr_volume, on=["user", "item"]
        )

        candidates["rank_meta"] = 10**9 * candidates["day_rank"] + candidates["volume_rank"]
        candidates["rank_meta"] = candidates.groupby("user")["rank_meta"].rank(method="min")
        # Sort by dictionary order of size of day and size of volume and leave only top items
        # Specify a large enough number for max_items_per_user if you want to keep all
        candidates = candidates.query("rank_meta <= @max_items_per_user").reset_index(drop=True)

        candidates = candidates[["user", "item", "week_rank", "volume_rank", "rank_meta"]].rename(
            columns={
                "week_rank": f"{strategy}_week_rank",
                "volume_rank": f"{strategy}_volume_rank",
            }
        )

        candidates["strategy"] = strategy
        return candidates.drop_duplicates(ignore_index=True)

    def create_candidates_popular(
        week_start: int, target_users: np.ndarray, num_weeks: int, num_items: int
    ) -> pd.DataFrame:
        tr = transactions.query("@week_start <= week < @week_start + @num_weeks")[
            ["user", "item"]
        ].drop_duplicates(ignore_index=True)
        popular_items = tr["item"].value_counts().index.values[:num_items]
        popular_items = pd.DataFrame(
            {"item": popular_items, "rank": range(num_items), "crossjoinkey": 1}
        )

        candidates = pd.DataFrame({"user": target_users, "crossjoinkey": 1})

        candidates = candidates.merge(popular_items, on="crossjoinkey").drop(
            "crossjoinkey", axis=1
        )
        candidates = candidates.rename(columns={"rank": "pop_rank"})

        candidates["strategy"] = "pop"
        return candidates.drop_duplicates(ignore_index=True)

    def create_candidates_age_popular(
        week_start: int, target_users: np.ndarray, num_weeks: int, num_items: int
    ) -> pd.DataFrame:
        tr = transactions.query("@week_start <= week < @week_start + @num_weeks")[
            ["user", "item"]
        ].drop_duplicates(ignore_index=True)
        tr = tr.merge(users[["user", "age"]])

        pops = []
        for age in range(16, 100):
            low = age - age_shifts[age]  # noqa: F841 used in pandas query
            high = age + age_shifts[age]  # noqa: F841 used in pandas query
            pop = tr.query("@low <= age <= @high")["item"].value_counts().index.values[:num_items]
            pops.append(
                pd.DataFrame({"age": age, "item": pop, "age_popular_rank": range(num_items)})
            )
        pops = pd.concat(pops)

        candidates = (
            users[["user", "age"]].dropna().query("user in @target_users").reset_index(drop=True)
        )

        candidates = candidates.merge(pops, on="age").drop("age", axis=1)

        candidates["strategy"] = "age_pop"
        return candidates.drop_duplicates(ignore_index=True)

    def create_candidates_category_popular(
        base_candidates: pd.DataFrame,
        week_start: int,
        num_weeks: int,
        num_items_per_category: int,
        category: str,
    ) -> pd.DataFrame:
        tr = transactions.query("@week_start <= week < @week_start + @num_weeks")[
            ["user", "item"]
        ].drop_duplicates()

        # TODO: modin bug, that's why we use iloc[]
        LARGE_NUMBER = 1_000_000_000
        tr = tr.iloc[:LARGE_NUMBER].groupby("item").size().reset_index(name="volume")
        tr = tr.merge(items[["item", category]], on="item")
        tr["cat_volume_rank"] = tr.groupby(category)["volume"].rank(ascending=False, method="min")
        tr = tr.query("cat_volume_rank <= @num_items_per_category").reset_index(drop=True)
        tr = tr[["item", category, "cat_volume_rank"]].reset_index(drop=True)

        candidates = base_candidates[["user", "item"]].merge(items[["item", category]], on="item")
        candidates = candidates.groupby(["user", category]).size().reset_index(name="cat_volume")
        candidates = candidates.merge(tr, on=category).drop(category, axis=1)
        candidates["strategy"] = "cat_pop"
        return candidates

    def create_candidates_cooc(
        base_candidates: pd.DataFrame, week_start: int, num_weeks: int, pair_count_threshold: int
    ) -> pd.DataFrame:
        week_end = week_start + num_weeks  # noqa: F841 used in pandas query
        tr = transactions.query("@week_start <= week < @week_end")[
            ["user", "item", "week"]
        ].drop_duplicates(ignore_index=True)
        tr = (
            tr.merge(tr.rename(columns={"item": "item_with", "week": "week_with"}), on="user")
            .query("item != item_with and week <= week_with")[["item", "item_with"]]
            .reset_index(drop=True)
        )
        gr_item_count = tr.groupby("item").size().reset_index(name="item_count")
        gr_pair_count = tr.groupby(["item", "item_with"]).size().reset_index(name="pair_count")
        item2item = gr_pair_count.merge(gr_item_count, on="item")
        item2item["ratio"] = item2item["pair_count"] / item2item["item_count"]
        item2item = item2item.query("pair_count > @pair_count_threshold").reset_index(drop=True)

        candidates = (
            base_candidates.merge(item2item, on="item")
            .drop(["item", "pair_count"], axis=1)
            .rename(columns={"item_with": "item"})
        )
        base_candidates_columns = [c for c in base_candidates.columns if "_" in c]
        base_candidates_replace = {c: f"cooc_{c}" for c in base_candidates_columns}
        candidates = candidates.rename(columns=base_candidates_replace)
        candidates = candidates.rename(
            columns={"ratio": "cooc_ratio", "item_count": "cooc_item_count"}
        )

        candidates["strategy"] = "cooc"
        return candidates.drop_duplicates(ignore_index=True)

    def create_candidates_same_product_code(base_candidates: pd.DataFrame) -> pd.DataFrame:
        item2item = (
            items[["item", "product_code"]]
            .merge(
                items[["item", "product_code"]].rename({"item": "item_with"}, axis=1),
                on="product_code",
            )[["item", "item_with"]]
            .query("item != item_with")
            .reset_index(drop=True)
        )

        candidates = (
            base_candidates.merge(item2item, on="item")
            .drop("item", axis=1)
            .rename(columns={"item_with": "item"})
        )

        candidates["min_rank_meta"] = candidates.groupby(["user", "item"])["rank_meta"].transform(
            "min"
        )
        candidates = candidates.query("rank_meta == min_rank_meta").reset_index(drop=True)

        base_candidates_columns = [c for c in base_candidates.columns if "_" in c]
        base_candidates_replace = {c: f"same_product_code_{c}" for c in base_candidates_columns}
        candidates = candidates.rename(columns=base_candidates_replace)

        candidates["strategy"] = "same_product_code"
        return candidates.drop_duplicates(ignore_index=True)

    def create_candidates_ohe_distance(
        target_users: np.ndarray, week_start: int, num_weeks: int, num_items: int
    ) -> pd.DataFrame:
        import faiss

        users_with_ohe = users[["user"]].query("user in @target_users")
        cols = [c for c in items.columns if c.endswith("_idx")]
        for c in cols:
            tmp = pd.read_pickle(user_features_path / f"user_ohe_agg_week{week_start}_{c}.pkl")
            users_with_ohe = users_with_ohe.merge(tmp, on="user")

        users_with_ohe = users_with_ohe.dropna().reset_index(drop=True)
        limited_users = users_with_ohe["user"].values

        recent_items = transactions.query(  # noqa: F841 used in pandas query
            "@week_start <= week < @week_start + @num_weeks"
        )["item"].unique()
        items_with_ohe = pd.get_dummies(items[["item"] + cols], columns=cols)
        items_with_ohe = items_with_ohe.query("item in @recent_items").reset_index(drop=True)
        limited_items = items_with_ohe["item"].values

        item_cols = [c for c in items_with_ohe.columns if c != "item"]
        user_cols = [f"user_ohe_agg_{c}" for c in item_cols]
        users_with_ohe = users_with_ohe[["user"] + user_cols]
        items_with_ohe = items_with_ohe[["item"] + item_cols]

        a_users = users_with_ohe.drop("user", axis=1).values.astype(np.float32)
        a_items = items_with_ohe.drop("item", axis=1).values.astype(np.float32)
        a_users = np.ascontiguousarray(a_users)
        a_items = np.ascontiguousarray(a_items)
        index = faiss.index_factory(a_items.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        if CFG.use_faiss_gpu:
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)

        index.add(a_items)
        distances, idxs = index.search(a_users, num_items)
        return pd.DataFrame(
            {
                "user": np.repeat(limited_users, num_items),
                "item": limited_items[idxs.flatten()],
                "ohe_distance": distances.flatten(),
                "strategy": "ohe_distance",
            }
        )

    with tm.timeit("01-repurchase"):
        candidates_repurchase = create_candidates_repurchase(
            "repurchase", week, target_users=target_users
        )
    with tm.timeit("02-popular"):
        candidates_popular = create_candidates_popular(
            week_start=week,
            target_users=target_users,
            num_weeks=CFG.popular_weeks,
            num_items=CFG.popular_num_items,
        )
    with tm.timeit("03-age popular"):
        candidates_age_popular = create_candidates_age_popular(
            week_start=week, target_users=target_users, num_weeks=1, num_items=12
        )
    with tm.timeit("04-item2item"):
        candidates_item2item = create_candidates_repurchase(
            "item2item", week, target_users, CFG.item2item_num_items
        )
    with tm.timeit("05-item2item2"):
        candidates_item2item2 = create_candidates_repurchase(
            "item2item2", week, target_users, CFG.item2item_num_items_for_same_product_code
        )
    with tm.timeit("06-cooccurrence"):
        candidates_cooc = create_candidates_cooc(
            candidates_item2item, week, CFG.cooc_weeks, CFG.cooc_threshold
        )
    with tm.timeit("07-same_product_code"):
        candidates_same_product_code = create_candidates_same_product_code(candidates_item2item2)
    if CFG.use_ohe_distance_candidates:
        with tm.timeit("08-ohe distance"):
            candidates_ohe_distance = create_candidates_ohe_distance(
                target_users=target_users,
                week_start=week,
                num_weeks=CFG.ohe_distance_num_weeks,
                num_items=CFG.ohe_distance_num_items,
            )
    else:
        candidates_ohe_distance = pd.DataFrame()
    with tm.timeit("09-category popular"):
        candidates_dept = create_candidates_category_popular(
            candidates_item2item2, week, 1, 6, "department_no_idx"
        )

    def drop_common_user_item(
        candidates_target: pd.DataFrame, candidates_reference: pd.DataFrame
    ) -> pd.DataFrame:
        """Drop candidates_target whose (user, item) pair is in candidates_reference"""
        tmp = candidates_reference[["user", "item"]].reset_index(drop=True)
        tmp["flag"] = 1
        candidates = candidates_target.merge(tmp, on=["user", "item"], how="left")
        return candidates.query("flag != 1").reset_index(drop=True).drop("flag", axis=1)

    candidates_cooc = drop_common_user_item(candidates_cooc, candidates_repurchase)
    candidates_same_product_code = drop_common_user_item(
        candidates_same_product_code, candidates_repurchase
    )
    if len(candidates_ohe_distance) > 0:
        candidates_ohe_distance = drop_common_user_item(
            candidates_ohe_distance, candidates_repurchase
        )
    candidates_dept = drop_common_user_item(candidates_dept, candidates_repurchase)

    candidates = [
        candidates_repurchase,
        candidates_popular,
        candidates_age_popular,
        candidates_cooc,
        candidates_same_product_code,
        candidates_ohe_distance,
        candidates_dept,
    ]
    candidates = pd.concat(candidates)

    logger.info("volume: %s", len(candidates))
    logger.info(
        "duplicates: %s", len(candidates) / len(candidates[["user", "item"]].drop_duplicates())
    )

    volumes = (
        candidates.groupby("strategy")
        .size()
        .reset_index(name="volume")
        .sort_values(by="volume", ascending=False)
        .reset_index(drop=True)
    )
    volumes["ratio"] = volumes["volume"] / volumes["volume"].sum()
    logger.info(volumes)

    meta_columns = [c for c in candidates.columns if c.endswith("_meta")]
    return candidates.drop(meta_columns, axis=1)


def merge_labels(candidates: pd.DataFrame, transactions, week: int) -> pd.DataFrame:
    """Returns labels for candidates for the week specified by week"""
    logger.info("merge labels (week: %s)", week)
    labels = transactions[transactions["week"] == week][["user", "item"]].drop_duplicates(
        ignore_index=True
    )
    labels["y"] = 1
    original_positives = len(labels)
    labels = candidates.merge(labels, on=["user", "item"], how="left")
    labels["y"] = labels["y"].fillna(0)

    remaining_positives_total = (
        labels[["user", "item", "y"]].drop_duplicates(ignore_index=True)["y"].sum()
    )
    recall = remaining_positives_total / original_positives
    logger.info("Recall: %s", recall)

    volumes = candidates.groupby("strategy").size().reset_index(name="volume")
    remaining_positives = labels.groupby("strategy")["y"].sum().reset_index()
    remaining_positives = remaining_positives.merge(volumes, on="strategy")
    remaining_positives["recall"] = remaining_positives["y"] / original_positives
    remaining_positives["hit_ratio"] = remaining_positives["y"] / remaining_positives["volume"]
    remaining_positives = remaining_positives.sort_values(by="y", ascending=False).reset_index(
        drop=True
    )
    logger.info(remaining_positives)

    return labels


def drop_trivial_users(labels):
    """
    In LightGBM's xendgc and lambdarank, users with only positive or negative examples are
    meaningless for learning, and the calculation of metrics is strange, so they are omitted.
    """
    bef = len(labels)
    df = labels[
        labels["user"].isin(
            labels[["user", "y"]]
            .drop_duplicates()
            .groupby("user")
            .size()
            .reset_index(name="sz")
            .query("sz==2")
            .user
        )
    ].reset_index(drop=True)
    aft = len(df)
    logger.info("drop trivial queries: %s -> %s", bef, aft)
    return df


def make_one_week_candidates(transactions, users, items, week, user_features_path, age_shifts):
    target_users = transactions.query("week == @week")["user"].unique()

    candidates = create_candidates(
        transactions=transactions,
        users=users,
        items=items,
        target_users=target_users,
        week=week + 1,
        user_features_path=user_features_path,
        age_shifts=age_shifts,
    )
    candidates = merge_labels(candidates=candidates, transactions=transactions, week=week)
    candidates["week"] = week

    return candidates


def make_weekly_candidates(
    transactions, users, items, train_weeks, user_features_path, age_shifts
):
    #################
    # valid: week=0
    # train: week=1..CFG.train_weeks
    candidates = []
    for week in range(1 + train_weeks):
        week_candidate = make_one_week_candidates(
            transactions=transactions,
            users=users,
            items=items,
            week=week,
            user_features_path=user_features_path,
            age_shifts=age_shifts,
        )

        candidates.append(week_candidate)

    candidates_valid_all = candidates[0].copy()
    for idx, week_candidates in enumerate(candidates):
        candidates[idx] = drop_trivial_users(week_candidates)
    return candidates, candidates_valid_all
