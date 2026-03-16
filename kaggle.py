import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


def target_encode(train, test, target, cat_cols, n_splits=5, alpha=20):
    train = train.copy()
    test = test.copy()
    global_mean = train[target].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for col in cat_cols:
        oof = np.zeros(len(train))
        test_enc = np.zeros(len(test))

        for train_idx, val_idx in kf.split(train):
            tr, val = train.iloc[train_idx], train.iloc[val_idx]

            stats = tr.groupby(col)[target].agg(['mean', 'count'])

            smooth = (
                    (stats['mean'] * stats['count'] + global_mean * alpha) /
                    (stats['count'] + alpha)
            )

            oof[val_idx] = val[col].map(smooth).fillna(global_mean)
            test_enc += test[col].map(smooth).fillna(global_mean) / n_splits

        train[col + "_te"] = oof
        test[col + "_te"] = test_enc

    return train, test


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
zov = pd.read_csv("test.csv")

train_data = train_data.drop("id", axis=1)
test_data = test_data.drop("id", axis=1)

Y = (train_data["Heart Disease"] == "Presence").astype(int)
train_data = train_data.drop("Heart Disease", axis=1)
train_data["target"] = Y

num_feat = ["Age", "BP", "Cholesterol", "Max HR", "ST depression", "Number of vessels fluro"]
cat_feat = ["Sex", "Chest pain type", "FBS over 120", "EKG results", "Exercise angina", "Slope of ST", "Thallium"]

for col in num_feat:
    train_data[col] = train_data[col].fillna(train_data[col].median())
    test_data[col] = test_data[col].fillna(train_data[col].median())

for col in cat_feat:
    train_data[col] = train_data[col].fillna("None")
    test_data[col] = test_data[col].fillna("None")

X, X_test = target_encode(
    train_data,
    test_data,
    target="target",
    cat_cols=cat_feat
)
X["Age_BP"] = X["Age"] * X["BP"]
X["HR_Age"] = X["Max HR"] / X["Age"]
X["ST_BP"] = X["ST depression"] * X["BP"]
X["Chol_HR"] = X["Cholesterol"] / X["Max HR"]
X["ChestPain_Age"] = X["Chest pain type_te"] * X["Age"]
X["Thal_ST"] = X["Thallium_te"] * X["ST depression"]

X_test["Age_BP"] = X_test["Age"] * X_test["BP"]
X_test["HR_Age"] = X_test["Max HR"] / X_test["Age"]
X_test["ST_BP"] = X_test["ST depression"] * X_test["BP"]
X_test["Chol_HR"] = X_test["Cholesterol"] / X_test["Max HR"]
X_test["ChestPain_Age"] = X_test["Chest pain type_te"] * X_test["Age"]
X_test["Thal_ST"] = X_test["Thallium_te"] * X_test["ST depression"]

X = X.drop("target", axis=1)

X[num_feat] = X[num_feat].astype(np.float32)
X_test[num_feat] = X_test[num_feat].astype(np.float32)

X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

params = {
    "loss_function": "Logloss",
    "eval_metric": "Accuracy",
    "iterations": 4000,
    "learning_rate": 0.06,
    "depth": 7,
    "l2_leaf_reg": 7,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.5,
    "random_strength": 2,
    "border_count": 254,
    "od_type": "Iter",
    "od_wait": 200,
    "random_seed": 42,
    "task_type" : "GPU",
    "devices" : "0",
    "verbose": 200
}

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X))
fold_test_preds = np.zeros((n_splits, len(X_test)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y)):
    print(f"\n===== Fold {fold + 1} =====")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = Y.iloc[train_idx], Y.iloc[val_idx]

    model = CatBoostClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_feat,
        early_stopping_rounds=1000,
        use_best_model=True,
        verbose=200
    )

    oof_preds[val_idx] = (model.predict_proba(X_val)[:, 1] > 0.5)
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits
    fold_test_preds[fold] = model.predict_proba(X_test)[:, 1]

oof_acc = accuracy_score(Y, oof_preds)
print("\nOOF Accuracy:", oof_acc)


mean_preds = fold_test_preds.mean(axis=0)
std_preds  = fold_test_preds.std(axis=0)

confident_mask = (
    ((mean_preds > 0.95) | (mean_preds < 0.05)) &
    (std_preds < 0.03)
)

print("Pseudo samples selected:", confident_mask.sum())

pseudo_X = X_test[confident_mask].copy()
pseudo_y = (mean_preds[confident_mask] > 0.5).astype(int)
pseudo_y = pd.Series(pseudo_y, index=pseudo_X.index)

X_aug = pd.concat([X, pseudo_X], axis=0)
y_aug = pd.concat([Y, pseudo_y], axis=0)

print("Augmented train size:", len(X_aug))

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from lightgbm import early_stopping, log_evaluation
from xgboost.callback import EarlyStopping

cat_oof = np.zeros(len(X_aug))
lgb_oof = np.zeros(len(X_aug))
xgb_oof = np.zeros(len(X_aug))

cat_test = np.zeros(len(X_test))
lgb_test = np.zeros(len(X_test))
xgb_test = np.zeros(len(X_test))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold,(tr_idx,val_idx) in enumerate(skf.split(X_aug,y_aug)):

    model = CatBoostClassifier(**params)

    model.fit(
        X_aug.iloc[tr_idx], y_aug.iloc[tr_idx],
        eval_set=(X_aug.iloc[val_idx], y_aug.iloc[val_idx]),
        cat_features=cat_feat,
        early_stopping_rounds=500,
        verbose=300
    )

    cat_oof[val_idx] = model.predict_proba(X_aug.iloc[val_idx])[:,1]
    cat_test += model.predict_proba(X_test)[:,1]/5
lgb = LGBMClassifier(
    n_estimators=10000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
X_aug_lgb = X_aug.drop(columns=cat_feat)
X_test_lgb = X_test.drop(columns=cat_feat)

for fold,(tr_idx,val_idx) in enumerate(skf.split(X_aug_lgb,y_aug)):

    lgb.fit(
        X_aug_lgb.iloc[tr_idx], y_aug.iloc[tr_idx],
        eval_set=[(X_aug_lgb.iloc[val_idx], y_aug.iloc[val_idx])],
        eval_metric="binary_logloss",
        callbacks=[
            early_stopping(400),
            log_evaluation(100)
        ]
    )

    lgb_oof[val_idx] = lgb.predict_proba(X_aug_lgb.iloc[val_idx],num_iteration=lgb.best_iteration_)[:, 1]
    lgb_test += lgb.predict_proba(X_test_lgb,num_iteration=lgb.best_iteration_)[:, 1] / 5
xgb = XGBClassifier(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    device="cuda",
    early_stopping_rounds=300,
    random_state=42
)


X_aug_lgb


for fold,(tr_idx,val_idx) in enumerate(skf.split(X_aug_lgb ,y_aug)):
    xgb.fit(
        X_aug_lgb.iloc[tr_idx], y_aug.iloc[tr_idx],
        eval_set=[(X_aug_lgb.iloc[val_idx], y_aug.iloc[val_idx])],
        verbose=200
    )
    xgb_oof[val_idx] = xgb.predict_proba(
        X_aug_lgb.iloc[val_idx],
        iteration_range=(0, xgb.best_iteration + 1)
    )[:, 1]

    xgb_test += xgb.predict_proba(
        X_test_lgb,
        iteration_range=(0, xgb.best_iteration + 1)
    )[:, 1] / 5


meta_X = np.vstack([
    cat_oof,
    lgb_oof,
    xgb_oof
]).T

meta_test = np.vstack([
    cat_test,
    lgb_test,
    xgb_test
]).T

from sklearn.linear_model import LogisticRegression

meta = LogisticRegression()
meta.fit(meta_X, y_aug)

final_pred = meta.predict_proba(meta_test)[:,1]




submission = pd.DataFrame(
    np.column_stack((zov["id"].astype(int), final_pred)),
    columns=["id", "Heart Disease"]
)
submission.to_csv("submission4.csv", index=False)