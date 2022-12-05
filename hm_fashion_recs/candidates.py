import faiss
import numpy as np
import pandas as pd

from utils import timer


class CFG:
    # candidates
    popular_num_items = 60
    popular_weeks = 1

    item2item_num_items = 24
    item2item_num_items_for_same_product_code = 12
    cooc_weeks = 32
    cooc_threshold = 150
    ohe_distance_num_items = 48
    ohe_distance_num_weeks = 20


def get_age_shifts(transactions, users):
    """Finds a range for each age bin so that volume in that bean was equal or greater than bin 24 <= age <= 26"""
    tr = transactions[["user", "item"]].merge(users[["user", "age"]], on="user")
    age_volume_threshold = len(tr.query("24 <= age <= 26"))

    age_volumes = {age: len(tr.query("age == @age")) for age in range(16, 100)}

    age_shifts = {}
    for age in range(16, 100):
        for i in range(0, 100):
            low = age - i
            high = age + i
            age_volume = 0
            for j in range(low, high + 1):
                age_volume += age_volumes.get(j, 0)
            if age_volume >= age_volume_threshold:
                age_shifts[age] = i
                break
    print(age_shifts)
    return age_shifts


##############
def create_candidates(
    transactions: pd.DataFrame, target_users: np.ndarray, week: int
) -> pd.DataFrame:
    """
    transactions
        original transactions (user, item, week)
    target_users
        user for candidate generation
    week
        candidates are generated using only the information available until and including this week
    """
    print(f"create candidates (week: {week})")
    assert len(target_users) == len(set(target_users))

    def create_candidates_repurchase(
        strategy: str,
        transactions: pd.DataFrame,
        target_users: np.ndarray,
        week_start: int,
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

        candidates["rank_meta"] = 10 ** 9 * candidates["day_rank"] + candidates["volume_rank"]
        candidates["rank_meta"] = candidates.groupby("user")["rank_meta"].rank(method="min")
        # item2item is a heavy workload and not the mose useful one
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
        transactions: pd.DataFrame,
        target_users: np.ndarray,
        week_start: int,
        num_weeks: int,
        num_items: int,
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
        candidates = candidates.rename(columns={"rank": f"pop_rank"})

        candidates["strategy"] = "pop"
        return candidates.drop_duplicates(ignore_index=True)

    def create_candidates_age_popular(
        transactions: pd.DataFrame,
        users: pd.DataFrame,
        target_users: np.ndarray,
        week_start: int,
        num_weeks: int,
        num_items: int,
    ) -> pd.DataFrame:
        tr = transactions.query("@week_start <= week < @week_start + @num_weeks")[
            ["user", "item"]
        ].drop_duplicates(ignore_index=True)
        tr = tr.merge(users[["user", "age"]])

        pops = []
        for age in range(16, 100):
            low = age - age_shifts[age]
            high = age + age_shifts[age]
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
        transactions: pd.DataFrame,
        items: pd.DataFrame,
        base_candidates: pd.DataFrame,
        week_start: int,
        num_weeks: int,
        num_items_per_category: int,
        category: str,
    ) -> pd.DataFrame:
        tr = transactions.query("@week_start <= week < @week_start + @num_weeks")[
            ["user", "item"]
        ].drop_duplicates()
        tr = tr.groupby("item").size().reset_index(name="volume")
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
        transactions: pd.DataFrame,
        base_candidates: pd.DataFrame,
        week_start: int,
        num_weeks: int,
        pair_count_threshold: int,
    ) -> pd.DataFrame:
        week_end = week_start + num_weeks
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
            columns={"ratio": "cooc_ratio", "item_count": f"cooc_item_count"}
        )

        candidates["strategy"] = "cooc"
        return candidates.drop_duplicates(ignore_index=True)

    def create_candidates_same_product_code(
        items: pd.DataFrame, base_candidates: pd.DataFrame
    ) -> pd.DataFrame:
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
        transactions: pd.DataFrame,
        users: pd.DataFrame,
        items: pd.DataFrame,
        target_users: np.ndarray,
        week_start: int,
        num_weeks: int,
        num_items: int,
    ) -> pd.DataFrame:
        users_with_ohe = users[["user"]].query("user in @target_users")
        cols = [c for c in items.columns if c.endswith("_idx")]
        for c in cols:
            tmp = pd.read_pickle(
                f"artifacts/user_features/user_ohe_agg_dataset{dataset}_week{week_start}_{c}.pkl"
            )
            # cs = [c for c in tmp.columns if c.endswith('_mean')]
            # tmp = tmp[['user'] + cs]
            users_with_ohe = users_with_ohe.merge(tmp, on="user")

        users_with_ohe = users_with_ohe.dropna().reset_index(drop=True)
        limited_users = users_with_ohe["user"].values

        recent_items = transactions.query("@week_start <= week < @week_start + @num_weeks")[
            "item"
        ].unique()
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

    with timer("repurchase"):
        candidates_repurchase = create_candidates_repurchase(
            "repurchase", transactions, target_users, week
        )
    with timer("popular"):
        candidates_popular = create_candidates_popular(
            transactions, target_users, week, CFG.popular_weeks, CFG.popular_num_items
        )
    with timer("age popular"):
        candidates_age_popular = create_candidates_age_popular(
            transactions, users, target_users, week, 1, 12
        )
    with timer("item2item"):
        candidates_item2item = create_candidates_repurchase(
            "item2item", transactions, target_users, week, CFG.item2item_num_items
        )
    with timer("item2item2"):
        candidates_item2item2 = create_candidates_repurchase(
            "item2item2",
            transactions,
            target_users,
            week,
            CFG.item2item_num_items_for_same_product_code,
        )
    with timer("cooccurrence"):
        candidates_cooc = create_candidates_cooc(
            transactions, candidates_item2item, week, CFG.cooc_weeks, CFG.cooc_threshold
        )
    with timer("same_product_code"):
        candidates_same_product_code = create_candidates_same_product_code(
            items, candidates_item2item2
        )
    with timer("ohe distance"):
        candidates_ohe_distance = create_candidates_ohe_distance(
            transactions,
            users,
            items,
            target_users,
            week,
            CFG.ohe_distance_num_weeks,
            CFG.ohe_distance_num_items,
        )
    with timer("category popular"):
        candidates_dept = create_candidates_category_popular(
            transactions, items, candidates_item2item2, week, 1, 6, "department_no_idx"
        )

    def drop_common_user_item(
        candidates_target: pd.DataFrame, candidates_reference: pd.DataFrame
    ) -> pd.DataFrame:
        """
        candidates_targetのうちuser, itemの組がcandidates_referenceにあるものを落とす
        """
        tmp = candidates_reference[["user", "item"]].reset_index(drop=True)
        tmp["flag"] = 1
        candidates = candidates_target.merge(tmp, on=["user", "item"], how="left")
        return candidates.query("flag != 1").reset_index(drop=True).drop("flag", axis=1)

    candidates_cooc = drop_common_user_item(candidates_cooc, candidates_repurchase)
    candidates_same_product_code = drop_common_user_item(
        candidates_same_product_code, candidates_repurchase
    )
    candidates_ohe_distance = drop_common_user_item(candidates_ohe_distance, candidates_repurchase)
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

    print(f"volume: {len(candidates)}")
    print(f"duplicates: {len(candidates) / len(candidates[['user', 'item']].drop_duplicates())}")

    volumes = (
        candidates.groupby("strategy")
        .size()
        .reset_index(name="volume")
        .sort_values(by="volume", ascending=False)
        .reset_index(drop=True)
    )
    volumes["ratio"] = volumes["volume"] / volumes["volume"].sum()
    print(volumes)

    meta_columns = [c for c in candidates.columns if c.endswith("_meta")]
    return candidates.drop(meta_columns, axis=1)


def merge_labels(candidates: pd.DataFrame, week: int) -> pd.DataFrame:
    """
    candidatesに対してweekで指定される週のトランザクションからラベルを付与する
    """
    print(f"merge labels (week: {week})")
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
    print(f"Recall: {recall}")

    volumes = candidates.groupby("strategy").size().reset_index(name="volume")
    remaining_positives = labels.groupby("strategy")["y"].sum().reset_index()
    remaining_positives = remaining_positives.merge(volumes, on="strategy")
    remaining_positives["recall"] = remaining_positives["y"] / original_positives
    remaining_positives["hit_ratio"] = remaining_positives["y"] / remaining_positives["volume"]
    remaining_positives = remaining_positives.sort_values(by="y", ascending=False).reset_index(
        drop=True
    )
    print(remaining_positives)

    return labels


def drop_trivial_users(labels):
    """
    In LightGBM's xendgc and lambdarank, users with only positive or negative examples are meaningless for learning, and the calculation of metrics is strange, so they are omitted.
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
    print(f"drop trivial queries: {bef} -> {aft}")
    return df


def make_candidates(transactions, train_weeks):
    #################
    # valid: week=0
    # train: week=1..CFG.train_weeks
    candidates = []
    for week in range(1 + train_weeks):
        target_users = transactions.query("week == @week")["user"].unique()

        week_candiates = create_candidates(transactions, target_users, week + 1)
        week_candiates = merge_labels(week_candiates, week)
        week_candiates["week"] = week

        candidates.append(week_candiates)

    candidates_valid_all = candidates[0].copy()
    for idx, week_candidates in enumerate(candidates):
        candidates[idx] = drop_trivial_users(week_candidates)
    return candidates, candidates_valid_all
