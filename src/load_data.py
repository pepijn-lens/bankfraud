import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.constants import RESULTS_DIR, TARGET_COL

def download_data(output_dir: Path):
    # Download latest version
    output_dir.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022")
    shutil.move(path, output_dir)
    print(f"Dataset downloaded and moved to {output_dir}")

def get_data(data_dir: Path) -> pd.DataFrame:
    file = data_dir / "Base.csv"
    if not file.is_file():
        print("File not found, downloading...")
        download_data(data_dir)
        if not file.is_file():
            raise FileNotFoundError("File not found after downloading.")
    return pd.read_csv(file)

def get_constant_columns(df: pd.DataFrame) -> list[str]:
    """
    Identifies columns that contain only a single distinct value (including NaN).
    """
    return [col for col in df.columns if df[col].nunique(dropna=False) <= 1]

def preprocess_global(df: pd.DataFrame) -> pd.DataFrame:
    """Global cleaning steps that do not depend on data distribution.

    Performs:
    - Row cleaning (dropping constant rows).
    - Column cleaning (dropping constant columns).
    - One-hot encoding.
    """
    df = df.copy()

    # Drop constant rows (where all columns have the same value)
    df_np = df.to_numpy()
    same_value_mask = np.all(df_np == df_np[:, [0]], axis=1)
    if same_value_mask.sum() > 0:
        print(f"Dropping {same_value_mask.sum()} constant rows.")
        df = df[~same_value_mask]

    # Drop constant columns
    constant_cols = get_constant_columns(df)
    if constant_cols:
        print(f"Dropping constant columns: {constant_cols}")
        df.drop(columns=constant_cols, inplace=True)

    # One-hot encoding
    categorical_types = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_types, drop_first=True)

    return df

def is_binary_int_col(s: pd.Series) -> bool:
    unique_vals = pd.Series(s.unique())
    if len(unique_vals) > 2:
        return False
    return set(unique_vals).issubset({0, 1})

def preprocess_fold(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Applies stateful transformations (Imputation, Scaling) fitting ONLY on X_train."""

    # Make copies to avoid side effects
    X_train_proc = X_train.copy()
    X_val_proc = X_val.copy()
    X_test_proc = X_test.copy()

    # Impute columns
    impute_cols = [
        "credit_risk_score",
        "device_distinct_emails_8w",
        "session_length_in_minutes",
        "current_address_months_count"
    ]
    impute_cols = [c for c in impute_cols if c in X_train.columns]

    if impute_cols:
        imputer = SimpleImputer(missing_values=-1, strategy='median')
        imputer.fit(X_train_proc[impute_cols])
        X_train_proc[impute_cols] = imputer.transform(X_train_proc[impute_cols])
        X_val_proc[impute_cols] = imputer.transform(X_val_proc[impute_cols])
        X_test_proc[impute_cols] = imputer.transform(X_test_proc[impute_cols])

    # Scale columns
    numeric_cols = X_train_proc.select_dtypes(include=[np.number]).columns
    continuous_cols = [col for col in numeric_cols if not is_binary_int_col(X_train_proc[col])]

    if continuous_cols:
        scaler = StandardScaler()
        scaler.fit(X_train_proc[continuous_cols])
        X_train_proc[continuous_cols] = scaler.transform(X_train_proc[continuous_cols])
        X_val_proc[continuous_cols] = scaler.transform(X_val_proc[continuous_cols])
        X_test_proc[continuous_cols] = scaler.transform(X_test_proc[continuous_cols])

    return X_train_proc, X_val_proc, X_test_proc

def prepare_data(
    data_path: Path | None = None,
    train_month_max: int = 5,
    val_month: int = 6,
    test_month: int = 7,
    save_test_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data, preprocess globally, and split chronologically by month.
    
    Args:
        data_path: Path to data file. If None, uses _get_data_path() logic.
        train_month_max: Maximum month for training (inclusive, so month <= train_month_max).
                         Default 5 means train on months <= 5.
        val_month: Month to use for validation. Default 6.
        test_month: Month to use for test. Default 7.
        save_test_path: If provided, save test set to this path. If None, saves to DATA_DIR / "test.csv".
    
    Returns:
        (df_train, df_val, df_test): DataFrames after preprocess_global, split by month.
                                     Month column is still present in returned DataFrames.
    """
    from src.constants import DATA_DIR
    
    # Resolve data path
    if data_path is None:
        p = DATA_DIR / "2" / "Base.csv"
        if p.exists():
            data_path = p
        else:
            data_path = DATA_DIR / "Base.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        df = get_data(DATA_DIR)
    
    # Preprocess globally
    df = preprocess_global(df)
    
    # Chronological split
    df_train = df[df['month'] <= train_month_max].copy()
    df_val = df[df['month'] == val_month].copy()
    df_test = df[df['month'] == test_month].copy()
    
    # Save test set if requested
    if save_test_path is None:
        save_test_path = DATA_DIR / "test.csv"
    if save_test_path and len(df_test) > 0:
        df_test.to_csv(save_test_path, index=False)
        print(f"Saved test set (month {test_month}) to {save_test_path}")
    
    print(f"Train: {len(df_train)} samples (month <= {train_month_max})")
    print(f"Val: {len(df_val)} samples (month == {val_month})")
    print(f"Test: {len(df_test)} samples (month == {test_month})")
    
    return df_train, df_val, df_test


def plot_graphs(df: pd.DataFrame) -> None:
    # Numeric features to plot (exclude target)
    numeric_features = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if df[c].nunique() >= 10
    ]
    n_features = len(numeric_features)
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols

    # KDE plots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows))
    fig.suptitle('Distribution of Numeric Features by Fraud Status')
    if nrows == 1:
        axes = axes.reshape(1, -1)
    for i, feature in enumerate(numeric_features):
        ax = axes[i // ncols, i % ncols]
        sns.kdeplot(data=df[df['fraud_bool'] == 0][feature], fill=True, ax=ax, label='Not Fraud', warn_singular=False)
        sns.kdeplot(data=df[df['fraud_bool'] == 1][feature], fill=True, ax=ax, label='Fraud', warn_singular=False)
        ax.set_xlabel(feature)
        ax.legend()
    for j in range(n_features, nrows * ncols):
        axes.flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "distribution_of_numeric_features.png")

    # Box plots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows))
    fig.suptitle('Box Plot of Numeric Features by Fraud Status')
    if nrows == 1:
        axes = axes.reshape(1, -1)
    for i, feature in enumerate(numeric_features):
        ax = axes[i // ncols, i % ncols]
        sns.boxplot(data=df, x='fraud_bool', y=feature, ax=ax, boxprops=dict(alpha=.6))
        ax.set_xlabel('')
        ax.set_ylabel(feature)
        ax.set_xticklabels(['Not Fraud', 'Fraud'])
    for j in range(n_features, nrows * ncols):
        axes.flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "boxplots_of_numeric_features.png")
