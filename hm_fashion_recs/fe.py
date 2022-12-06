from pathlib import Path
import numpy as np

# import modin.pandas as pd
import pandas as pd

from lfm import calc_embeddings

from vars import lfm_features_path, dim
from utils import timer


class CFG:
    # features
    user_transaction_feature_weeks = 50
    item_transaction_feature_weeks = 16
    item_age_feature_weeks = 40
    user_volume_feature_weeks = 50
    item_volume_feature_weeks = 20
    user_item_volume_feature_weeks = 16
    age_volume_feature_weeks = 1


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


def attach_features(
    transactions: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    candidates: pd.DataFrame,
    week: int,
    pretrain_week: int,
    age_shifts,
    user_features_path: Path,
) -> pd.DataFrame:
    """
    user, itemに対して特徴を横付けする
    week: これを含めた以前の情報は使って良い
    """
    print(f"attach features (week: {week})")
    n_original = len(candidates)
    df = candidates.copy()
    with timer("user static fetaures"):
        user_features = ["age"]
        df = df.merge(users[["user"] + user_features], on="user")

    with timer("item stacic features"):
        item_features = [c for c in items.columns if c.endswith("idx")]
        df = df.merge(items[["item"] + item_features], on="item")

    with timer("user dynamic features (transactions)"):
        week_end = week + CFG.user_transaction_feature_weeks
        tmp = (
            transactions.query("@week <= week < @week_end")
            .groupby("user")[["price", "sales_channel_id"]]
            .agg(["mean", "std"])
        )
        tmp.columns = ["user_" + "_".join(a) for a in tmp.columns.to_flat_index()]
        df = df.merge(tmp, on="user", how="left")

    with timer("item dynamic features (transactions)"):
        week_end = week + CFG.item_transaction_feature_weeks
        tmp = (
            transactions.query("@week <= week < @week_end")
            .groupby("item")[["price", "sales_channel_id"]]
            .agg(["mean", "std"])
        )
        tmp.columns = ["item_" + "_".join(a) for a in tmp.columns.to_flat_index()]
        df = df.merge(tmp, on="item", how="left")

    with timer("item dynamic features (user features)"):
        week_end = week + CFG.item_age_feature_weeks
        tmp = transactions.query("@week <= week < @week_end").merge(
            users[["user", "age"]], on="user"
        )
        tmp = tmp.groupby("item")["age"].agg(["mean", "std"])
        tmp.columns = [f"age_{a}" for a in tmp.columns.to_flat_index()]
        df = df.merge(tmp, on="item", how="left")

    with timer("item freshness features"):
        tmp = (
            transactions.query("@week <= week")
            .groupby("item")["day"]
            .min()
            .reset_index(name="item_day_min")
        )
        tmp["item_day_min"] -= transactions.query("@week == week")["day"].min()
        df = df.merge(tmp, on="item", how="left")

    with timer("item volume features"):
        week_end = week + CFG.item_volume_feature_weeks
        tmp = (
            transactions.query("@week <= week < @week_end")
            .groupby("item")
            .size()
            .reset_index(name="item_volume")
        )
        df = df.merge(tmp, on="item", how="left")

    with timer("user freshness features"):
        tmp = (
            transactions.query("@week <= week")
            .groupby("user")["day"]
            .min()
            .reset_index(name="user_day_min")
        )
        tmp["user_day_min"] -= transactions.query("@week == week")["day"].min()
        df = df.merge(tmp, on="user", how="left")

    with timer("user volume features"):
        week_end = week + CFG.user_volume_feature_weeks
        tmp = (
            transactions.query("@week <= week < @week_end")
            .groupby("user")
            .size()
            .reset_index(name="user_volume")
        )
        df = df.merge(tmp, on="user", how="left")

    with timer("user-item freshness features"):
        tmp = (
            transactions.query("@week <= week")
            .groupby(["user", "item"])["day"]
            .min()
            .reset_index(name="user_item_day_min")
        )
        tmp["user_item_day_min"] -= transactions.query("@week == week")["day"].min()
        df = df.merge(tmp, on=["item", "user"], how="left")

    with timer("user-item volume features"):
        week_end = week + CFG.user_item_volume_feature_weeks
        tmp = (
            transactions.query("@week <= week < @week_end")
            .groupby(["user", "item"])
            .size()
            .reset_index(name="user_item_volume")
        )
        df = df.merge(tmp, on=["user", "item"], how="left")

    with timer("item age volume features"):
        week_end = week + CFG.age_volume_feature_weeks
        tr = transactions.query("@week <= week < @week_end")[["user", "item"]].merge(
            users[["user", "age"]], on="user"
        )
        item_age_volumes = []
        for age in range(16, 100):
            low = age - age_shifts[age]
            high = age + age_shifts[age]
            tmp = (
                tr.query("@low <= age <= @high")
                .groupby("item")
                .size()
                .reset_index(name="age_volume")
            )
            tmp["age_volume"] = tmp["age_volume"].rank(ascending=False)
            tmp["age"] = age
            item_age_volumes.append(tmp)
        item_age_volumes = pd.concat(item_age_volumes)
        df = df.merge(item_age_volumes, on=["item", "age"], how="left")

    with timer("user category most frequent"):
        for c in ["department_no_idx"]:
            tmp = pd.read_pickle(
                user_features_path / f"user_ohe_agg_week{week}_{c}.pkl"
            )
            cols = [c for c in tmp.columns if c != "user"]
            # tmp = tmp[['user'] + cols]
            tmp[cols] = tmp[cols] / tmp[cols].mean()
            tmp[f"{c}_most_freq_idx"] = np.argmax(tmp[cols].values, axis=1)
            df = df.merge(tmp[["user", f"{c}_most_freq_idx"]])

    with timer("ohe dot products"):
        item_target_cols = [c for c in items.columns if c.endswith("_idx")]

        items_with_ohe = pd.get_dummies(
            items[["item"] + item_target_cols], columns=item_target_cols
        )

        cols = [c for c in items_with_ohe.columns if c != "item"]
        items_with_ohe[cols] = items_with_ohe[cols] / items_with_ohe[cols].mean()

        users_with_ohe = users[["user"]]
        for c in item_target_cols:
            tmp = pd.read_pickle(
                user_features_path / f"user_ohe_agg_week{week}_{c}.pkl"
            )
            assert tmp["user"].tolist() == users_with_ohe["user"].tolist()
            # tmp = tmp[['user'] + [c for c in tmp.columns if c.endswith('_mean')]]
            tmp = tmp.drop("user", axis=1)
            users_with_ohe = pd.concat([users_with_ohe, tmp], axis=1)

        assert items_with_ohe["item"].tolist() == items["item"].tolist()
        assert users_with_ohe["user"].tolist() == users["user"].tolist()

        users_items = df[["user", "item"]].drop_duplicates().reset_index(drop=True)
        n_split = 10
        n_chunk = (len(users_items) + n_split - 1) // n_split
        ohe = []
        for i in range(0, len(users_items), n_chunk):
            users_items_small = users_items.iloc[i : i + n_chunk].reset_index(drop=True)
            users_small = users_items_small["user"].values
            items_small = users_items_small["item"].values

            for item_col in item_target_cols:
                i_cols = [c for c in items_with_ohe.columns if c.startswith(item_col)]
                u_cols = [f"user_ohe_agg_{c}" for c in i_cols]
                print(users_with_ohe.shape)

                users_items_small[f"{item_col}_ohe_score"] = (
                    items_with_ohe[i_cols].values[items_small]
                    * users_with_ohe[u_cols].values[users_small]
                ).sum(axis=1)

            ohe_cols = [f"{col}_ohe_score" for col in item_target_cols]
            users_items_small = users_items_small[["user", "item"] + ohe_cols]

            ohe.append(users_items_small)
        ohe = pd.concat(ohe)
        df = df.merge(ohe, on=["user", "item"])

    with timer("lfm features"):
        seen_users = transactions.query("week >= @pretrain_week")["user"].unique()
        user_reps, _ = calc_embeddings(lfm_features_path, pretrain_week, dim=dim)
        user_reps = user_reps.query("user in @seen_users")
        df = df.merge(user_reps, on="user", how="left")

    assert len(df) == n_original
    return df
