import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import pandas as pd
import numpy as np

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
