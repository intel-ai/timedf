from hm_fashion_recs.preprocess import transform_data, create_user_ohe_agg
from hm_fashion_recs.train_lfm_features import train_lfm
from hm_fashion_recs.notebook import get_age_shifts, make_weekly_candidates, load_data
from hm_fashion_recs.candidates import make_one_week_candidates, drop_trivial_users
from hm_fashion_recs.fe import attach_features

from hm_fashion_recs.vars import (
    raw_data_path,
    preprocessed_data_path,
    user_features_path,
    lfm_features_path,
    dim,
    working_dir
)
from hm_fashion_recs.tm import tm



def feature_engieering(week):
    transactions, users, items = load_data()

    age_shifts = get_age_shifts(transactions=transactions, users=users)

    week_candidates = make_one_week_candidates(
        transactions=transactions,
        users=users,
        items=items,
        week=week,
        user_features_path=CFG.user_features_path,
        age_shifts=age_shifts
    )

    candidates = drop_trivial_users(week_candidates)

    dataset = attach_features(
        transactions=transactions,
        users=users,
        items=items,
        candidates=candidates,
        # +1 because train data comes one week earlier
        week=week+1,
        pretrain_week=week+2,
        age_shifts=age_shifts,
        user_features_path=CFG.user_features_path,
    )

    dataset["query_group"] = dataset["week"].astype(str) + "_" + dataset["user"].astype(str)
    dataset = dataset.sort_values(by="query_group").reset_index(drop=True)
    return dataset


class CFG:
    raw_data_path = raw_data_path
    preprocessed_data_path = preprocessed_data_path
    lfm_features_path = lfm_features_path
    working_dir = working_dir
    user_features_path = user_features_path
    
    use_lfm = False


if __name__ == "__main__":
    with tm.timeit('1-initial_transform'):
        transform_data(input_data_path=CFG.raw_data_path, result_path=CFG.preprocessed_data_path)

    week = 0
    with tm.timeit('2-create_user_ohe_agg'):
        create_user_ohe_agg(
        week + 1, preprocessed_data_path=CFG.preprocessed_data_path, result_path=CFG.user_features_path
    )

    if CFG.use_lfm:
        with tm.timeit('3-create_lfm_features'):
            # features are pretraied, that's why +2
            train_lfm(week=week+2, lfm_features_path=CFG.lfm_features_path, dim=dim)

    with tm.timeit('4-fe'):
        feature_engieering(week=week)

    print(tm.get_results())
