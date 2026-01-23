"""This is all the data analysis logic from Robert's main file (now removed)"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from constants import DATA_DIR, TARGET_COL, RESULTS_DIR
from load_data import get_data


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

if __name__ == "__main__":
    df = get_data(DATA_DIR)

    analyze_data(df)

    plot_categorical_fraud_rates(
        df,
        target_col=TARGET_COL,
        categorical_cols=[
            *df.select_dtypes(include=['object']).columns,
            "income",
            "customer_age",
        ],
        output_dir = RESULTS_DIR,
    )


