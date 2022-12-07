from hm_fashion_recs.preprocess import transform_data, create_user_ohe_agg
from hm_fashion_recs.train_lfm_features import train_lfm

from hm_fashion_recs.vars import (
    n_weeks,
    raw_data_path,
    preprocessed_data_path,
    user_features_path,
    lfm_features_path,
    dim,
)


use_lfm = False

if __name__ == "__main__":
    transform_data(input_data_path=raw_data_path, result_path=preprocessed_data_path)

    for week in range(n_weeks + 1):
        create_user_ohe_agg(
            week, preprocessed_data_path=preprocessed_data_path, result_path=user_features_path
        )

    if use_lfm:

        for week in range(1, n_weeks + 1):
            train_lfm(week=week, lfm_features_path=lfm_features_path, dim=dim)
