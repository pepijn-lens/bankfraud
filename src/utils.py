import shutil
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

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

def pop_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.copy()
    y = X.pop(target_col)
    return X, y

def shuffle_data(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    return df.sample(frac=1, random_state=seed)

def split_data(df: pd.DataFrame, ratios: list[float]) -> list[pd.DataFrame]:
    """
    Splits a DataFrame into multiple subsets based on a list of ratios.

    Args:
        df: The source DataFrame to split.
        ratios: A list of floats (0 to 1) defining the size of the first n splits.
                The sum of these must be <= 1.

    Returns:
        A list of len(ratios) + 1 DataFrames. The last DataFrame contains
        the remaining data corresponding to (1 - sum(ratios)).
    """
    # Validate inputs
    if any(r < 0 for r in ratios):
        raise ValueError("All ratios must be greater than or equal to 0.")
    if sum(ratios) > 1:
        raise ValueError("The sum of ratios must be less than or equal to 1.")

    # Create the first N splits based on the provided ratios
    total_rows = len(df)
    splits = []
    current_idx = 0
    for ratio in ratios:
        n_rows = round(total_rows * ratio)
        end_idx = current_idx + n_rows

        split = df.iloc[current_idx:end_idx].copy()
        splits.append(split)

        current_idx = end_idx

    # Create the final (N+1)th split with the remaining data
    remaining_split = df.iloc[current_idx:].copy()
    splits.append(remaining_split)

    return splits

def onehot_encode(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Number of columns before one-hot encoding: {len(df.columns)}")
    categorical_types = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_types, drop_first=True)
    print(f"Number of columns after one-hot encoding: {len(df.columns)}")
    return df

def analyze_data(df: pd.DataFrame) -> None:
    print("Starting analysis...")

    print(df.describe())
    print(df.info())

    print("\nDistinct values for each column:")
    for col in df.columns:
        distinct_values = df[col].unique()
        n_unique_values = len(distinct_values)
        max_values = 64
        if n_unique_values <= max_values:
            distinct_values.sort()
            print(f"- {col} ({n_unique_values}): {distinct_values}")
        else:
            print(f"- {col} ({n_unique_values}): > {max_values} unique values")

    print("\nNumber of -1 values per column (hidden if zero):")
    for col in df.columns:
        n_missing = (df[col] == -1).sum()
        if n_missing > 0:
            print(f"- {col}: {n_missing}")

    df_np = df.to_numpy()
    same_value_mask = np.all(df_np == df_np[:, [0]], axis=1)
    n_corrupted = same_value_mask.sum()
    print(f"\nNumber of rows where all values are equal: {n_corrupted}")

    print("\nCorrelations:")
    if "credit_risk_score" in df.columns and "fraud_bool" in df.columns:
        corr = df["credit_risk_score"].corr(df["fraud_bool"])
        print(f"Pearson correlation (credit_risk_score vs fraud_bool): {corr:.4f}")

def one_distinct_value(xs: pd.Series) -> bool:
    """Returns True if the series only contains a single distinct value.

    Considers NaN as a distinct value.
    """
    return xs.nunique(dropna=False) <= 1

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning data based on analysis...")

    df = df.copy()

    df_np = df.to_numpy()
    same_value_mask = np.all(df_np == df_np[:, [0]], axis=1)
    n_corrupted = same_value_mask.sum()
    if n_corrupted > 0:
        print(f"- Removing {n_corrupted} corrupted rows (all values equal)")
        df = df[~same_value_mask]

    for col in df.columns:
        if one_distinct_value(df[col]):
            df.drop(columns=[col], inplace=True)
            print(f"- Dropped constant column: {col}")

    return df

def filter_cols(cols: list[str] | None, df: pd.DataFrame) -> list[str]:
    if cols is None:
        return list(df.columns)
    return [col for col in cols if col in df.columns]

def impute_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cols: list[str] | None = None,
) -> None:
    cols = filter_cols(cols, X_train)
    if not cols:
        return
    print(f"Imputing features: {cols}")

    imputer = SimpleImputer(missing_values=-1, strategy='median')
    imputer.fit(X_train[cols])
    X_train[cols] = imputer.transform(X_train[cols])
    X_val[cols] = imputer.transform(X_val[cols])
    X_test[cols] = imputer.transform(X_test[cols])

def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cols: list[str] | None = None,
) -> None:
    cols = filter_cols(cols, X_train)
    if not cols:
        return
    print(f"Scaling features: {cols}")

    scaler = StandardScaler()
    scaler.fit(X_train[cols])
    X_train[cols] = scaler.transform(X_train[cols])
    X_val[cols] = scaler.transform(X_val[cols])
    X_test[cols] = scaler.transform(X_test[cols])

def train_model(X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float = 99.0):
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    model.fit(X_train, y_train)
    return model

def plot_categorical_fraud_rates(df: pd.DataFrame, target_col: str, categorical_cols: list[str], output_dir: Path):
    """
    Plots the mean fraud rate for categorical features.
    """
    print("Plotting categorical fraud rates...")
    mean_fraud_rate = df[target_col].mean()

    output_dir.mkdir(parents=True, exist_ok=True)

    for col in categorical_cols:
        plt.figure(figsize=(10, 6))

        # Calculate stats
        fraud_rate = df.groupby(col)[target_col].mean()
        counts = df[col].value_counts().sort_index()

        # Plot
        sns.barplot(x=fraud_rate.index, y=fraud_rate.values, color='skyblue')
        plt.axhline(mean_fraud_rate, color='red', linestyle='--', label=f'Mean fraud rate ({mean_fraud_rate:.4f})')

        # Annotate
        for i, (label, count) in enumerate(counts.items()):
            if label in fraud_rate.index:  # Safety check
                bar_height = fraud_rate[label]
                y_pos = bar_height / 2
                color = "black" if bar_height < 0.5 else "white"
                plt.text(i, y_pos, f"n={count}", ha='center', va='center', fontsize=9, color=color)

        plt.title(f'Fraud Ratio and Counts for {col}')
        plt.ylabel('Mean Fraud Rate')
        plt.xlabel(col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"fraud_rate_per_{col}.png")

def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series):
    print("Evaluating model...")
    y_pred = model.predict(X_val)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))