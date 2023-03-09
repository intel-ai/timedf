# GPu reqs
# !pip -q install ../input/pytorchtabnet/pytorch_tabnet-2.0.1-py3-none-any.whl

import gc
import os
import random
from typing import List, Optional, Tuple, Union
import pickle

import numpy as np

# Visualization deps
import matplotlib.pyplot as plt
import seaborn as sns

# Training deps
import lightgbm as lgb

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from omniscripts.pandas_backend import pd

from .optiver_utils import tm, get_workdir_paths

# You need to provide this path
raw_data_path = None
paths = get_workdir_paths(raw_data_path)

df_train = pd.read_feather(paths["train_dataset"])
df_test = pd.read_feather(paths["test_dataset"])
with open(paths["folds"], "rb") as f:
    folds = pickle.load(f)


MEMORY_TEST_MODE = True

# data configurations
USE_PRECOMPUTE_FEATURES = (
    False  # Load precomputed features for train.csv from private dataset (just for speed up)
)

# model & ensemble configurations
PREDICT_CNN = True
PREDICT_MLP = True
PREDICT_GBDT = True
PREDICT_TABNET = False

GBDT_NUM_MODELS = 5  # 3
GBDT_LR = 0.02  # 0.1

NN_VALID_TH = 0.185
NN_MODEL_TOP_N = 3
TAB_MODEL_TOP_N = 3
ENSEMBLE_METHOD = "mean"
NN_NUM_MODELS = 10
TABNET_NUM_MODELS = 5

# for saving quota
IS_1ST_STAGE = True
SHORTCUT_NN_IN_1ST_STAGE = False  # early-stop training to save GPU quota
SHORTCUT_GBDT_IN_1ST_STAGE = False
MEMORY_TEST_MODE = False

# for ablation studies
CV_SPLIT = "time"  # 'time': time-series KFold 'group': GroupKFold by stock-id
USE_PRICE_NN_FEATURES = True  # Use nearest neighbor features that rely on tick size
USE_VOL_NN_FEATURES = (
    True  # Use nearest neighbor features that can be calculated without tick size
)
USE_SIZE_NN_FEATURES = (
    True  # Use nearest neighbor features that can be calculated without tick size
)
USE_RANDOM_NN_FEATURES = False  # Use random index to aggregate neighbors

USE_TIME_ID_NN = True  # Use time-id based neighbors
USE_STOCK_ID_NN = True  # Use stock-id based neighbors

ENABLE_RANK_NORMALIZATION = True  # Enable rank-normalization


# ## LightGBM Training
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


def feval_RMSPE(preds, train_data):
    labels = train_data.get_label()
    return "RMSPE", round(rmspe(y_true=labels, y_pred=preds), 5), False


# from: https://blog.amedama.jp/entry/lightgbm-cv-feature-importance
def plot_importance(cvbooster, figsize=(10, 10)):
    raw_importances = cvbooster.feature_importance(importance_type="gain")
    feature_name = cvbooster.boosters[0].feature_name()
    importance_df = pd.DataFrame(data=raw_importances, columns=feature_name)
    # order by average importance across folds
    sorted_indices = importance_df.mean(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    # plot top-n
    PLOT_TOP_N = 50
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale("log")
    ax.set_ylabel("Feature")
    ax.set_xlabel("Importance")
    sns.boxplot(data=sorted_importance_df[plot_cols], orient="h", ax=ax)
    plt.show()


def get_X(df_src):
    cols = [c for c in df_src.columns if c not in ["time_id", "target", "tick_size"]]
    return df_src[cols]


class EnsembleModel:
    def __init__(self, models: List[lgb.Booster], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights

        features = list(self.models[0].feature_name())

        for m in self.models[1:]:
            assert features == list(m.feature_name())

    def predict(self, x):
        predicted = np.zeros((len(x), len(self.models)))

        for i, m in enumerate(self.models):
            w = self.weights[i] if self.weights is not None else 1
            predicted[:, i] = w * m.predict(x)

        ttl = np.sum(self.weights) if self.weights is not None else len(self.models)
        return np.sum(predicted, axis=1) / ttl

    def feature_name(self) -> List[str]:
        return self.models[0].feature_name()


# %%
lr = GBDT_LR
if SHORTCUT_GBDT_IN_1ST_STAGE and IS_1ST_STAGE:
    # to save GPU quota
    lr = 0.3

params = {
    "objective": "regression",
    "verbose": 0,
    "metric": "",
    "reg_alpha": 5,
    "reg_lambda": 5,
    "min_data_in_leaf": 1000,
    "max_depth": -1,
    "num_leaves": 128,
    "colsample_bytree": 0.3,
    "learning_rate": lr,
}

X = get_X(df_train)
y = df_train["target"]
X.to_feather("X.f")
df_train[["target"]].to_feather("y.f")

gc.collect()

print(X.shape)

if PREDICT_GBDT:
    ds = lgb.Dataset(X, y, weight=1 / np.power(y, 2))

    with tm.timeit("lgb.cv"):
        ret = lgb.cv(
            params,
            ds,
            num_boost_round=8000,
            folds=folds,  # cv,
            feval=feval_RMSPE,
            stratified=False,
            return_cvbooster=True,
            verbose_eval=20,
            early_stopping_rounds=int(40 * 0.1 / lr),
        )

        print(f"# overall RMSPE: {ret['RMSPE-mean'][-1]}")

    best_iteration = len(ret["RMSPE-mean"])
    for i in range(len(folds)):
        y_pred = (
            ret["cvbooster"].boosters[i].predict(X.iloc[folds[i][1]], num_iteration=best_iteration)
        )
        y_true = y.iloc[folds[i][1]]
        print(f"# fold{i} RMSPE: {rmspe(y_true, y_pred)}")

        if i == len(folds) - 1:
            np.save("pred_gbdt.npy", y_pred)

    plot_importance(ret["cvbooster"], figsize=(10, 20))

    boosters = []
    with tm.timeit("retraining"):
        for i in range(GBDT_NUM_MODELS):
            params["seed"] = i
            boosters.append(lgb.train(params, ds, num_boost_round=int(1.1 * best_iteration)))

    booster = EnsembleModel(boosters)
    del ret
    del ds

gc.collect()

# %% [markdown]
# ## NN Training

null_check_cols = [
    "book.log_return1.realized_volatility",
    "book_150.log_return1.realized_volatility",
    "book_300.log_return1.realized_volatility",
    "book_450.log_return1.realized_volatility",
    "trade.log_return.realized_volatility",
    "trade_150.log_return.realized_volatility",
    "trade_300.log_return.realized_volatility",
    "trade_450.log_return.realized_volatility",
]


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def rmspe_metric(y_true, y_pred):
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    return rmspe


def rmspe_loss(y_true, y_pred):
    rmspe = torch.sqrt(torch.mean(torch.square((y_true - y_pred) / y_true)))
    return rmspe


class RMSPE(Metric):
    def __init__(self):
        self._name = "rmspe"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return np.sqrt(np.mean(np.square((y_true - y_score) / y_true)))


def RMSPELoss_Tabnet(y_pred, y_true):
    return torch.sqrt(torch.mean(((y_true - y_pred) / y_true) ** 2)).clone()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TabularDataset(Dataset):
    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: Optional[np.ndarray]):
        super().__init__()
        self.x_num = x_num
        self.x_cat = x_cat
        self.y = y

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x_num[idx], torch.LongTensor(self.x_cat[idx])
        else:
            return self.x_num[idx], torch.LongTensor(self.x_cat[idx]), self.y[idx]


class MLP(nn.Module):
    def __init__(
        self,
        src_num_dim: int,
        n_categories: List[int],
        dropout: float = 0.0,
        hidden: int = 50,
        emb_dim: int = 10,
        dropout_cat: float = 0.2,
        bn: bool = False,
    ):
        super().__init__()

        self.embs = nn.ModuleList([nn.Embedding(x, emb_dim) for x in n_categories])
        self.cat_dim = emb_dim * len(n_categories)
        self.dropout_cat = nn.Dropout(dropout_cat)

        if bn:
            self.sequence = nn.Sequential(
                nn.Linear(src_num_dim + self.cat_dim, hidden),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.sequence = nn.Sequential(
                nn.Linear(src_num_dim + self.cat_dim, hidden),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, x_num, x_cat):
        embs = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embs)]
        x_cat_emb = self.dropout_cat(torch.cat(embs, 1))
        x_all = torch.cat([x_num, x_cat_emb], 1)
        x = self.sequence(x_all)
        return torch.squeeze(x)


class CNN(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        n_categories: List[int],
        emb_dim: int = 10,
        dropout_cat: float = 0.2,
        channel_1: int = 256,
        channel_2: int = 512,
        channel_3: int = 512,
        dropout_top: float = 0.1,
        dropout_mid: float = 0.3,
        dropout_bottom: float = 0.2,
        weight_norm: bool = True,
        two_stage: bool = True,
        celu: bool = True,
        kernel1: int = 5,
        leaky_relu: bool = False,
    ):
        super().__init__()

        num_targets = 1

        cha_1_reshape = int(hidden_size / channel_1)
        cha_po_1 = int(hidden_size / channel_1 / 2)
        cha_po_2 = int(hidden_size / channel_1 / 2 / 2) * channel_3

        self.cat_dim = emb_dim * len(n_categories)
        self.cha_1 = channel_1
        self.cha_2 = channel_2
        self.cha_3 = channel_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2
        self.two_stage = two_stage

        self.expand = nn.Sequential(
            nn.BatchNorm1d(num_features + self.cat_dim),
            nn.Dropout(dropout_top),
            nn.utils.weight_norm(nn.Linear(num_features + self.cat_dim, hidden_size), dim=None),
            nn.CELU(0.06) if celu else nn.ReLU(),
        )

        def _norm(layer, dim=None):
            return nn.utils.weight_norm(layer, dim=dim) if weight_norm else layer

        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(channel_1),
            nn.Dropout(dropout_top),
            _norm(
                nn.Conv1d(
                    channel_1,
                    channel_2,
                    kernel_size=kernel1,
                    stride=1,
                    padding=kernel1 // 2,
                    bias=False,
                )
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=cha_po_1),
            nn.BatchNorm1d(channel_2),
            nn.Dropout(dropout_top),
            _norm(nn.Conv1d(channel_2, channel_2, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(),
        )

        if self.two_stage:
            self.conv2 = nn.Sequential(
                nn.BatchNorm1d(channel_2),
                nn.Dropout(dropout_mid),
                _norm(
                    nn.Conv1d(channel_2, channel_2, kernel_size=3, stride=1, padding=1, bias=True)
                ),
                nn.ReLU(),
                nn.BatchNorm1d(channel_2),
                nn.Dropout(dropout_bottom),
                _norm(
                    nn.Conv1d(channel_2, channel_3, kernel_size=5, stride=1, padding=2, bias=True)
                ),
                nn.ReLU(),
            )

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        if leaky_relu:
            self.dense = nn.Sequential(
                nn.BatchNorm1d(cha_po_2),
                nn.Dropout(dropout_bottom),
                _norm(nn.Linear(cha_po_2, num_targets), dim=0),
                nn.LeakyReLU(),
            )
        else:
            self.dense = nn.Sequential(
                nn.BatchNorm1d(cha_po_2),
                nn.Dropout(dropout_bottom),
                _norm(nn.Linear(cha_po_2, num_targets), dim=0),
            )

        self.embs = nn.ModuleList([nn.Embedding(x, emb_dim) for x in n_categories])
        self.cat_dim = emb_dim * len(n_categories)
        self.dropout_cat = nn.Dropout(dropout_cat)

    def forward(self, x_num, x_cat):
        embs = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embs)]
        x_cat_emb = self.dropout_cat(torch.cat(embs, 1))
        x = torch.cat([x_num, x_cat_emb], 1)

        x = self.expand(x)

        x = x.reshape(x.shape[0], self.cha_1, self.cha_1_reshape)

        x = self.conv1(x)

        if self.two_stage:
            x = self.conv2(x) * x

        x = self.max_po_c2(x)
        x = self.flt(x)
        x = self.dense(x)

        return torch.squeeze(x)


def preprocess_nn(
    X: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    scaler_type: str = "standard",
    n_pca: int = -1,
    na_cols: bool = True,
):
    if na_cols:
        # for c in X.columns:
        for c in null_check_cols:
            if c in X.columns:
                X[f"{c}_isnull"] = X[c].isnull().astype(int)

    cat_cols = [c for c in X.columns if c in ["time_id", "stock_id"]]
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_num = X[num_cols].values.astype(np.float32)
    X_cat = np.nan_to_num(X[cat_cols].values.astype(np.int32))

    def _pca(X_num_):
        if n_pca > 0:
            pca = PCA(n_components=n_pca, random_state=0)
            return pca.fit_transform(X_num)
        return X_num

    if scaler is None:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
        X_num = np.nan_to_num(X_num, posinf=0, neginf=0)
        return _pca(X_num), X_cat, cat_cols, scaler
    else:
        X_num = scaler.transform(X_num)  # TODO: infでも大丈夫？
        X_num = np.nan_to_num(X_num, posinf=0, neginf=0)
        return _pca(X_num), X_cat, cat_cols


def train_epoch(
    data_loader: DataLoader, model: nn.Module, optimizer, scheduler, device, clip_grad: float = 1.5
):
    model.train()
    losses = AverageMeter()
    step = 0

    for x_num, x_cat, y in tqdm(data_loader, position=0, leave=True, desc="Training"):
        batch_size = x_num.size(0)
        x_num = x_num.to(device, dtype=torch.float)
        x_cat = x_cat.to(device)
        y = y.to(device, dtype=torch.float)

        loss = rmspe_loss(y, model(x_num, x_cat))
        losses.update(loss.detach().cpu().numpy(), batch_size)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        step += 1

    return losses.avg


def evaluate(data_loader: DataLoader, model, device):
    model.eval()

    losses = AverageMeter()

    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for x_num, x_cat, y in tqdm(data_loader, position=0, leave=True, desc="Evaluating"):
            batch_size = x_num.size(0)
            x_num = x_num.to(device, dtype=torch.float)
            x_cat = x_cat.to(device)
            y = y.to(device, dtype=torch.float)

            with torch.no_grad():
                output = model(x_num, x_cat)

            loss = rmspe_loss(y, output)
            # record loss
            losses.update(loss.detach().cpu().numpy(), batch_size)

            targets = y.detach().cpu().numpy()
            output = output.detach().cpu().numpy()

            final_targets.append(targets)
            final_outputs.append(output)

    final_targets = np.concatenate(final_targets)
    final_outputs = np.concatenate(final_outputs)

    try:
        metric = rmspe_metric(final_targets, final_outputs)
    except BaseException:
        metric = None

    return final_outputs, final_targets, losses.avg, metric


def predict_nn(
    X: pd.DataFrame,
    model: Union[List[MLP], MLP],
    scaler: StandardScaler,
    device,
    ensemble_method="mean",
):
    if not isinstance(model, list):
        model = [model]

    for m in model:
        m.eval()
    X_num, X_cat, cat_cols = preprocess_nn(X.copy(), scaler=scaler)
    valid_dataset = TabularDataset(X_num, X_cat, None)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=512, shuffle=False, num_workers=4
    )

    final_outputs = []

    with torch.no_grad():
        for x_num, x_cat in tqdm(valid_loader, position=0, leave=True, desc="Evaluating"):
            x_num = x_num.to(device, dtype=torch.float)
            x_cat = x_cat.to(device)

            outputs = []
            with torch.no_grad():
                for m in model:
                    output = m(x_num, x_cat)
                    outputs.append(output.detach().cpu().numpy())

            if ensemble_method == "median":
                pred = np.nanmedian(np.array(outputs), axis=0)
            else:
                pred = np.array(outputs).mean(axis=0)
            final_outputs.append(pred)

    final_outputs = np.concatenate(final_outputs)
    return final_outputs


def predict_tabnet(
    X: pd.DataFrame,
    model: Union[List[TabNetRegressor], TabNetRegressor],
    scaler: StandardScaler,
    ensemble_method="mean",
):
    if not isinstance(model, list):
        model = [model]

    X_num, X_cat, cat_cols = preprocess_nn(X.copy(), scaler=scaler)
    X_processed = np.concatenate([X_cat, X_num], axis=1)

    predicted = []
    for m in model:
        predicted.append(m.predict(X_processed))

    if ensemble_method == "median":
        pred = np.nanmedian(np.array(predicted), axis=0)
    else:
        pred = np.array(predicted).mean(axis=0)

    return pred


def train_tabnet(
    X: pd.DataFrame,
    y: pd.DataFrame,
    folds: List[Tuple],
    batch_size: int = 1024,
    lr: float = 1e-3,
    model_path: str = "fold_{}.pth",
    scaler_type: str = "standard",
    output_dir: str = "artifacts",
    epochs: int = 250,
    seed: int = 42,
    n_pca: int = -1,
    na_cols: bool = True,
    patience: int = 10,
    factor: float = 0.5,
    gamma: float = 2.0,
    lambda_sparse: float = 8.0,
    n_steps: int = 2,
    scheduler_type: str = "cosine",
    n_a: int = 16,
):
    seed_everything(seed)

    os.makedirs(output_dir, exist_ok=True)

    y = y.values.astype(np.float32)
    X_num, X_cat, cat_cols, scaler = preprocess_nn(
        X.copy(), scaler_type=scaler_type, n_pca=n_pca, na_cols=na_cols
    )

    best_losses = []
    best_predictions = []

    for cv_idx, (train_idx, valid_idx) in enumerate(folds):
        X_tr, X_va = X_num[train_idx], X_num[valid_idx]
        X_tr_cat, X_va_cat = X_cat[train_idx], X_cat[valid_idx]
        y_tr, y_va = y[train_idx], y[valid_idx]
        y_tr = y_tr.reshape(-1, 1)
        y_va = y_va.reshape(-1, 1)
        X_tr = np.concatenate([X_tr_cat, X_tr], axis=1)
        X_va = np.concatenate([X_va_cat, X_va], axis=1)

        cat_idxs = [0]
        cat_dims = [128]

        if scheduler_type == "cosine":
            scheduler_params = dict(T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False)
            scheduler_fn = CosineAnnealingWarmRestarts
        else:
            scheduler_params = {
                "mode": "min",
                "min_lr": 1e-7,
                "patience": patience,
                "factor": factor,
                "verbose": True,
            }
            scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau

        model = TabNetRegressor(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=1,
            n_d=n_a,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=2,
            n_shared=2,
            lambda_sparse=lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": lr},
            mask_type="entmax",
            scheduler_fn=scheduler_fn,
            scheduler_params=scheduler_params,
            seed=seed,
            verbose=10
            # device_name=device,
            # clip_value=1.5
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            max_epochs=epochs,
            patience=50,
            batch_size=1024 * 20,
            virtual_batch_size=batch_size,
            num_workers=4,
            drop_last=False,
            eval_metric=[RMSPE],
            loss_fn=RMSPELoss_Tabnet,
        )

        path = os.path.join(output_dir, model_path.format(cv_idx))
        model.save_model(path)

        predicted = model.predict(X_va)

        rmspe = rmspe_metric(y_va, predicted)
        best_losses.append(rmspe)
        best_predictions.append(predicted)

    return best_losses, best_predictions, scaler, model


def train_nn(
    X: pd.DataFrame,
    y: pd.DataFrame,
    folds: List[Tuple],
    device,
    emb_dim: int = 25,
    batch_size: int = 1024,
    model_type: str = "mlp",
    mlp_dropout: float = 0.0,
    mlp_hidden: int = 64,
    mlp_bn: bool = False,
    cnn_hidden: int = 64,
    cnn_channel1: int = 32,
    cnn_channel2: int = 32,
    cnn_channel3: int = 32,
    cnn_kernel1: int = 5,
    cnn_celu: bool = False,
    cnn_weight_norm: bool = False,
    dropout_emb: bool = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    model_path: str = "fold_{}.pth",
    scaler_type: str = "standard",
    output_dir: str = "artifacts",
    scheduler_type: str = "onecycle",
    optimizer_type: str = "adam",
    max_lr: float = 0.01,
    epochs: int = 30,
    seed: int = 42,
    n_pca: int = -1,
    batch_double_freq: int = 50,
    cnn_dropout: float = 0.1,
    na_cols: bool = True,
    cnn_leaky_relu: bool = False,
    patience: int = 8,
    factor: float = 0.5,
):
    seed_everything(seed)

    os.makedirs(output_dir, exist_ok=True)

    y = y.values.astype(np.float32)
    X_num, X_cat, cat_cols, scaler = preprocess_nn(
        X.copy(), scaler_type=scaler_type, n_pca=n_pca, na_cols=na_cols
    )

    best_losses = []
    best_predictions = []

    for cv_idx, (train_idx, valid_idx) in enumerate(folds):
        X_tr, X_va = X_num[train_idx], X_num[valid_idx]
        X_tr_cat, X_va_cat = X_cat[train_idx], X_cat[valid_idx]
        y_tr, y_va = y[train_idx], y[valid_idx]

        cur_batch = batch_size
        best_loss = 1e10
        best_prediction = None

        print(f"fold {cv_idx} train: {X_tr.shape}, valid: {X_va.shape}")

        train_dataset = TabularDataset(X_tr, X_tr_cat, y_tr)
        valid_dataset = TabularDataset(X_va, X_va_cat, y_va)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cur_batch, shuffle=True, num_workers=4
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=cur_batch, shuffle=False, num_workers=4
        )

        if model_type == "mlp":
            model = MLP(
                X_tr.shape[1],
                n_categories=[128],
                dropout=mlp_dropout,
                hidden=mlp_hidden,
                emb_dim=emb_dim,
                dropout_cat=dropout_emb,
                bn=mlp_bn,
            )
        elif model_type == "cnn":
            model = CNN(
                X_tr.shape[1],
                hidden_size=cnn_hidden,
                n_categories=[128],
                emb_dim=emb_dim,
                dropout_cat=dropout_emb,
                channel_1=cnn_channel1,
                channel_2=cnn_channel2,
                channel_3=cnn_channel3,
                two_stage=False,
                kernel1=cnn_kernel1,
                celu=cnn_celu,
                dropout_top=cnn_dropout,
                dropout_mid=cnn_dropout,
                dropout_bottom=cnn_dropout,
                weight_norm=cnn_weight_norm,
                leaky_relu=cnn_leaky_relu,
            )
        else:
            raise NotImplementedError()
        model = model.to(device)

        if optimizer_type == "adamw":
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError()

        scheduler = epoch_scheduler = None
        if scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=opt,
                pct_start=0.1,
                div_factor=1e3,
                max_lr=max_lr,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
            )
        elif scheduler_type == "reduce":
            epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=opt,
                mode="min",
                min_lr=1e-7,
                patience=patience,
                verbose=True,
                factor=factor,
            )

        for epoch in range(epochs):
            if epoch > 0 and epoch % batch_double_freq == 0:
                cur_batch = cur_batch * 2
                print(f"batch: {cur_batch}")
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=cur_batch, shuffle=True, num_workers=4
                )
            train_loss = train_epoch(train_loader, model, opt, scheduler, device)
            predictions, valid_targets, valid_loss, rmspe = evaluate(
                valid_loader, model, device=device
            )
            print(f"epoch {epoch}, train loss: {train_loss:.3f}, valid rmspe: {rmspe:.3f}")

            if epoch_scheduler is not None:
                epoch_scheduler.step(rmspe)

            if rmspe < best_loss:
                print(f"new best:{rmspe}")
                best_loss = rmspe
                best_prediction = predictions
                torch.save(model, os.path.join(output_dir, model_path.format(cv_idx)))

        best_predictions.append(best_prediction)
        best_losses.append(best_loss)
        del (
            model,
            train_dataset,
            valid_dataset,
            train_loader,
            valid_loader,
            X_tr,
            X_va,
            X_tr_cat,
            X_va_cat,
            y_tr,
            y_va,
            opt,
        )
        if scheduler is not None:
            del scheduler
        gc.collect()

    return best_losses, best_predictions, scaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

del df_train
gc.collect()


def get_top_n_models(models, scores, top_n):
    if len(models) <= top_n:
        print("number of models are less than top_n. all models will be used")
        return models
    sorted_ = [(y, x) for y, x in sorted(zip(scores, models), key=lambda pair: pair[0])]
    print(f"scores(sorted): {[y for y, _ in sorted_]}")
    return [x for _, x in sorted_][:top_n]


if PREDICT_MLP:
    model_paths = []
    scores = []

    if SHORTCUT_NN_IN_1ST_STAGE and IS_1ST_STAGE:
        print("shortcut to save quota...")
        epochs = 3
        valid_th = 100
    else:
        epochs = 30
        valid_th = NN_VALID_TH

    for i in range(NN_NUM_MODELS):
        # MLP
        nn_losses, nn_preds, scaler = train_nn(
            X,
            y,
            [folds[-1]],
            device=device,
            batch_size=512,
            mlp_bn=True,
            mlp_hidden=256,
            mlp_dropout=0.0,
            emb_dim=30,
            epochs=epochs,
            lr=0.002,
            max_lr=0.0055,
            weight_decay=1e-7,
            model_path="mlp_fold_{}" + f"_seed{i}.pth",
            seed=i,
        )
        if nn_losses[0] < NN_VALID_TH:
            print(f"model of seed {i} added.")
            scores.append(nn_losses[0])
            model_paths.append(f"artifacts/mlp_fold_0_seed{i}.pth")
            np.save(f"pred_mlp_seed{i}.npy", nn_preds[0])

    model_paths = get_top_n_models(model_paths, scores, NN_MODEL_TOP_N)
    mlp_model = [torch.load(path, device) for path in model_paths]
    print(f"total {len(mlp_model)} models will be used.")
if PREDICT_CNN:
    model_paths = []
    scores = []

    if SHORTCUT_NN_IN_1ST_STAGE and IS_1ST_STAGE:
        print("shortcut to save quota...")
        epochs = 3
        valid_th = 100
    else:
        epochs = 50
        valid_th = NN_VALID_TH

    for i in range(NN_NUM_MODELS):
        nn_losses, nn_preds, scaler = train_nn(
            X,
            y,
            [folds[-1]],
            device=device,
            cnn_hidden=8 * 128,
            batch_size=1280,
            model_type="cnn",
            emb_dim=30,
            epochs=epochs,  # epochs,
            cnn_channel1=128,
            cnn_channel2=3 * 128,
            cnn_channel3=3 * 128,
            lr=0.00038,  # 0.0011,
            max_lr=0.0013,
            weight_decay=6.5e-6,
            optimizer_type="adam",
            scheduler_type="reduce",
            model_path="cnn_fold_{}" + f"_seed{i}.pth",
            seed=i,
            cnn_dropout=0.0,
            cnn_weight_norm=True,
            cnn_leaky_relu=False,
            patience=8,
            factor=0.3,
        )
        if nn_losses[0] < valid_th:
            model_paths.append(f"artifacts/cnn_fold_0_seed{i}.pth")
            scores.append(nn_losses[0])
            np.save(f"pred_cnn_seed{i}.npy", nn_preds[0])

    model_paths = get_top_n_models(model_paths, scores, NN_MODEL_TOP_N)
    cnn_model = [torch.load(path, device) for path in model_paths]
    print(f"total {len(cnn_model)} models will be used.")

if PREDICT_TABNET:
    tab_model = []
    scores = []

    if SHORTCUT_NN_IN_1ST_STAGE and IS_1ST_STAGE:
        print("shortcut to save quota...")
        epochs = 10
        valid_th = 1000
    else:
        print("train full")
        epochs = 250
        valid_th = NN_VALID_TH

    for i in range(TABNET_NUM_MODELS):
        nn_losses, nn_preds, scaler, model = train_tabnet(
            X,
            y,
            [folds[-1]],
            batch_size=1280,
            epochs=epochs,  # epochs,
            lr=0.04,
            patience=50,
            factor=0.5,
            gamma=1.6,
            lambda_sparse=3.55e-6,
            seed=i,
            n_a=36,
        )
        if nn_losses[0] < valid_th:
            tab_model.append(model)
            scores.append(nn_losses[0])
            np.save(f"pred_tab_seed{i}.npy", nn_preds[0])
            model.save_model(f"artifacts/tabnet_fold_0_seed{i}")

    tab_model = get_top_n_models(tab_model, scores, TAB_MODEL_TOP_N)
    print(f"total {len(tab_model)} models will be used.")
