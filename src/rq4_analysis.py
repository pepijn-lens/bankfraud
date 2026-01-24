import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.training import prepare_data
from src.evaluation import ValueAwareEvaluator
from src.constants import RESULTS_DIR

def plot_error_concentration(caught_df, missed_df, avg_caught, avg_missed):
    """
    Plots the density of credit limits for detected vs. missed fraud cases.
    Saves the figure to the results directory.
    """
    plt.figure(figsize=(10, 6))

    # Plot Missed Fraud (False Negatives)
    sns.kdeplot(data=missed_df, x='proposed_credit_limit', fill=True,
                color='#e74c3c', alpha=0.5, label='Missed Fraud (False Negatives)', linewidth=0)

    # Plot Detected Fraud (True Positives)
    sns.kdeplot(data=caught_df, x='proposed_credit_limit', fill=True,
                color='#2ecc71', alpha=0.5, label='Detected Fraud (True Positives)', linewidth=0)

    # Reference lines for averages
    plt.axvline(avg_missed, color='#c0392b', linestyle='--', alpha=0.8, label=f'Avg Missed: ${avg_missed:,.0f}')
    plt.axvline(avg_caught, color='#27ae60', linestyle='--', alpha=0.8, label=f'Avg Detected: ${avg_caught:,.0f}')

    plt.title('RQ4: Systematic Concentration of Missed Fraud', fontsize=14)
    plt.xlabel('Proposed Credit Limit ($)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 3000)  # Focus on low-value range

    # Save to results folder
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "rq4_error_concentration.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() # Prevents popup
    print(f"Graph saved to: {output_path}")

def run_rq4_analysis(model_path="models/rf_model.pkl"):
    """
    Executes RQ4 analysis: Checks if minimizing expected loss leads to
    systematic concentration of errors among low-value cases.
    """
    print(f"\n{'='*60}\nRQ4: Long-Term Consequences Analysis\n{'='*60}")

    # Load test data to ensure consistency with training pipeline
    _, X_test, _, y_test, _ = prepare_data(test_size=0.25, random_state=42)

    path = Path(model_path)
    if not path.exists():
        print(f"Model not found at {path}. Skipping RQ4.")
        return

    print(f"Loading model from {path}...")
    with open(path, "rb") as f:
        model = pickle.load(f)

    # Generate probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Find Loss-Minimizing Threshold
    print("Calculating optimal decision threshold...")
    evaluator = ValueAwareEvaluator()
    thresholds = np.linspace(0, 1, 101)
    
    best_loss = float('inf')
    best_thresh = 0.5

    for thresh in thresholds:
        res = evaluator.evaluate(
            y_true=y_test,
            y_pred_prob=y_prob,
            X_features=X_test,
            threshold_method="static",
            static_threshold=thresh
        )
        if res['Total_Bank_Loss_($)'] < best_loss:
            best_loss = res['Total_Bank_Loss_($)']
            best_thresh = thresh

    print(f"Optimal Threshold: {best_thresh:.4f} (Total Loss: ${best_loss:,.2f})")

    # Analyze Error Distribution
    y_pred = (y_prob >= best_thresh).astype(int)
    
    df_analysis = X_test.copy()
    df_analysis['is_fraud'] = y_test
    df_analysis['predicted_fraud'] = y_pred

    # Filter for actual fraud cases
    fraud_cases = df_analysis[df_analysis['is_fraud'] == 1]
    caught = fraud_cases[fraud_cases['predicted_fraud'] == 1]
    missed = fraud_cases[fraud_cases['predicted_fraud'] == 0]

    avg_caught = caught['proposed_credit_limit'].mean()
    avg_missed = missed['proposed_credit_limit'].mean()

    print(f"\n{'-'*60}\nAnalysis Results\n{'-'*60}")
    print(f"Total Fraud Cases: {len(fraud_cases)}")
    print(f"Detected (High-Value): {len(caught)} (Avg: ${avg_caught:,.0f})")
    print(f"Missed (Low-Value):    {len(missed)} (Avg: ${avg_missed:,.0f})")
    print(f"{'-'*60}")

    if avg_caught > avg_missed:
        print("Conclusion: Model systematically prioritizes high-value fraud.")
    else:
        print("Conclusion: No systematic value bias detected.")

    plot_error_concentration(caught, missed, avg_caught, avg_missed)

    