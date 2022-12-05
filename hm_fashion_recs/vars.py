from pathlib import Path


raw_data_path = Path("/localdisk/benchmark_datasets/hm_fashion_recs/")

working_dir = Path('/localdisk/ekrivov/hm/tmpdir')

working_dir.mkdir(exist_ok=True, parents=True)

preprocessed_data_path = working_dir / 'preprocessed'
artifacts_path = working_dir / 'artifacts'
lfm_features_path = artifacts_path / 'lfm'
user_features_path = artifacts_path / 'user_features'

preprocessed_data_path.mkdir(exist_ok=True, parents=True)
lfm_features_path.mkdir(exist_ok=True, parents=True)
user_features_path.mkdir(exist_ok=True, parents=True)

# n_weeks = 14 # original
n_weeks = 1
# train_weeks = 6 # orignial 
train_weeks = 0
dim = 16