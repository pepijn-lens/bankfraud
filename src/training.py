"""
Tree-based machine learning for bank account fraud detection.

Follows the method from the research paper:
- NearMiss undersampling (sampling_strategy=0.1) for imbalanced data
- Train/test split 75/25 after resampling, stratified
- Tree-based models (RandomForest, XGBoost, LightGBM): no feature scaling
- Drop constant features (e.g. device_fraud_count from Variance Threshold)
- encoded_features: column indices for features with 2 <= nunique < 10
"""

import pickle
from collections import Counter
from pathlib import Path
import os

import pandas as pd
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.constants import DATA_DIR, TARGET_COL
from src.load_data import get_data, preprocess_global
from src.eval import test_classifier, plot_roc_curves


def _get_data_path() -> Path:
    """Resolve data path: prefer data/2/Base.csv (Kaggle layout), else data/Base.csv."""
    p = DATA_DIR / "2" / "Base.csv"
    if p.exists():
        return p
    return DATA_DIR / "Base.csv"


def _get_encoded_feature_indices(X: pd.DataFrame) -> list[int]:
    """
    Indices of encoded (categorical-like) features: 2 <= nunique < 10.
    Matches: X_train.columns.get_loc(str(feature)) for feature in ... if 2 <= nunique < 10
    """
    return [
        X.columns.get_loc(str(f))
        for f in X.columns
        if 2 <= X[f].nunique() < 10
    ]


def prepare_data(
    sampling_strategy: float = 0.1,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """
    Load data, preprocess, drop constant features, NearMiss undersample, and split.

    Returns:
        X_train, X_test, y_train, y_test, encoded_features (column indices)
    """
    path = _get_data_path()
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = get_data(DATA_DIR)

    if 8 in df["month"]:
        df_test = df[df["month"] == 7]
        df_test.to_csv(f"{DATA_DIR}/test.csv")
        df = df[df["month"] != 7]
        print(f"Saved month 8 from the dataset for testing to {DATA_DIR}/test.csv")

    df = preprocess_global(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # NearMiss undersampling
    print(f"Dataset samples per class {Counter(y)}")
    nm = NearMiss(sampling_strategy=sampling_strategy, n_jobs=-1)
    X_nm, y_nm = nm.fit_resample(X, y)
    print(f"Resampled dataset shape {Counter(y_nm)}")

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_nm, y_nm, test_size=test_size, random_state=random_state, stratify=y_nm
    )

    encoded_features = _get_encoded_feature_indices(X_train)
    return X_train, X_test, y_train, y_test, encoded_features


def train_classifier(model, X_train, y_train):
    """
    Train a tree-based classifier. No scaling; resampling is done before split.

    Parameters:
        model: sklearn/lightgbm/xgboost classifier (RandomForest, XGB, LGBM).
        X_train: training features.
        y_train: training target.

    Returns:
        Fitted model.
    """
    model.fit(X_train, y_train)
    return model


def main():
    X_train, X_test, y_train, y_test, encoded_features = prepare_data(
        sampling_strategy=0.1,
        test_size=0.25,
        random_state=42,
    )
    print(f"Encoded feature indices (2 <= nunique < 10): {encoded_features}")

    # Tree-based models (no feature scaling)
    models = {
        "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=42, n_jobs=-1),
        "LightGBM": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
    }

    fitted = {}
    for name, model in models.items():
        print(f"Training {name}...")
        fitted[name] = train_classifier(model, X_train, y_train)

    # Save each trained model as pickle (paper: rf_model.pkl, xgb_model.pkl, lgb_model.pkl)
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_filenames = {
        "RandomForest": "rf_model.pkl",
        "XGBoost": "xgb_model.pkl",
        "LightGBM": "lgb_model.pkl",
    }
    for name, model in fitted.items():
        path = Path(models_dir) / model_filenames[name]
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved {name} to {path}")

    # Testing: test_classifier for each model (paper 6.1–6.4), then plot ROC curves (6.5)
    rf_fpr, rf_tpr, _, _ = test_classifier(fitted["RandomForest"], X_test, y_test)
    xgb_fpr, xgb_tpr, _, _ = test_classifier(fitted["XGBoost"], X_test, y_test)
    lgb_fpr, lgb_tpr, _, _ = test_classifier(fitted["LightGBM"], X_test, y_test)

    fpr_list = [rf_fpr, xgb_fpr, lgb_fpr]
    tpr_list = [rf_tpr, xgb_tpr, lgb_tpr]
    label_list = ["RandomForest", "XGBoost", "LightGBM"]
    plot_roc_curves(fpr_list, tpr_list, label_list)


if __name__ == "__main__":
    main()
