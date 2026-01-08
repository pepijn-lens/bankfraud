import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from src.load_data import preprocess_data, split_data, normalize
from src.models import get_base_model
from src.evaluation import ValueAwareEvaluator

# to print whole tables
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

CONFIG = {
    "data_path": "data/2/Base.csv",
    "naive_threshold": 0.5,
    "high_exposure_quantile": 0.90,
    "alpha": 1.0,
}

def find_best_f1_threshold(y_true, y_prob):
    """Finds the static threshold that maximizes F1 Score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    # Argmax finds the index of the highest F1 score
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

def run_experiment():
    print("1. Loading and Preprocessing Data...")
    df = pd.read_csv(CONFIG["data_path"])

    df = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_scaled, X_val_scaled, X_test_scaled, _ = normalize(X_train, X_val, X_test)

    print(f"Data ready. Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    print("\n2. Training Logistic Regression...")
    model = get_base_model()
    model.fit(X_train_scaled, y_train)

    y_prob_val = model.predict_proba(X_val_scaled)[:, 1]
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]

    best_static_thresh = find_best_f1_threshold(y_val, y_prob_val)
    print(f"Optimal Static Threshold (Based on Val F1): {best_static_thresh:.4f}")

    print("\n3. Fitting Value-Aware Evaluator...")
    evaluator = ValueAwareEvaluator()
    evaluator.fit(X_train, y_train)

    # -------------------------
    # RQ1: Overall test set
    # -------------------------
    print("\n=== RQ1: Total Cost Analysis (Lower is Better) ===")

    res_naive = evaluator.calculate_savings(
        y_true=y_test,
        y_pred_prob=y_prob_test,
        X_features=X_test,
        threshold_method="static",
        static_threshold=CONFIG["naive_threshold"],
        alpha=CONFIG["alpha"],
    )
    res_f1 = evaluator.calculate_savings(
        y_true=y_test,
        y_pred_prob=y_prob_test,
        X_features=X_test,
        threshold_method="static",
        static_threshold=best_static_thresh,
        alpha=CONFIG["alpha"],
    )
    res_dynamic = evaluator.calculate_savings(
        y_true=y_test,
        y_pred_prob=y_prob_test,
        X_features=X_test,
        threshold_method="dynamic",
        alpha=CONFIG["alpha"],
    )

    results_df = pd.DataFrame(
        [res_naive, res_f1, res_dynamic],
        index=["Naive (0.5)", "Best F1", "Value-Aware"],
    )

    money_cols = ["Total_Bank_Loss_($)", "Fraud_Loss_($)", "False_Alarm_Cost_($)", "Fraud_Caught_($)"]
    print(results_df[money_cols].round(0))
    print(results_df[["recall", "accuracy", "f1"]].round(3))

    # --- Sensitivity analysis over alpha ---
    print("\n=== Sensitivity Analysis: Varying alpha ===")

    alphas = [0.01, 0.05, 0.10]
    rows = []

    for a in alphas:
        res_naive = evaluator.calculate_savings(
            y_true=y_test,
            y_pred_prob=y_prob_test,
            X_features=X_test,
            threshold_method="static",
            static_threshold=CONFIG["naive_threshold"],
            alpha=a,
        )
        res_f1 = evaluator.calculate_savings(
            y_true=y_test,
            y_pred_prob=y_prob_test,
            X_features=X_test,
            threshold_method="static",
            static_threshold=best_static_thresh,
            alpha=a,
        )
        res_dynamic = evaluator.calculate_savings(
            y_true=y_test,
            y_pred_prob=y_prob_test,
            X_features=X_test,
            threshold_method="dynamic",
            alpha=a,
        )

        rows.extend([
            {"alpha": a, "method": "Naive (0.5)", **res_naive},
            {"alpha": a, "method": "Best F1", **res_f1},
            {"alpha": a, "method": "Value-Aware", **res_dynamic},
        ])

    alpha_results = pd.DataFrame(rows)

    print(
        alpha_results[
            ["alpha", "method",
             "Total_Bank_Loss_($)",
             "Fraud_Loss_($)",
             "False_Alarm_Cost_($)",
             "recall"]
        ].round(3)
    )


    # -------------------------
    # RQ2: High-exposure segment
    # -------------------------
    print("\n=== RQ2: High-exposure cases (top quantile by proposed credit limit) ===")

    q = CONFIG["high_exposure_quantile"]
    cutoff = X_test["proposed_credit_limit"].quantile(q)
    high_exposure_mask = X_test["proposed_credit_limit"] >= cutoff

    print(
        f"High-exposure definition: proposed_credit_limit >= {cutoff:.2f} "
        f"(top {(1 - q) * 100:.0f}% of applications)"
    )

    high_res_naive = evaluator.calculate_savings(
        y_true=y_test[high_exposure_mask],
        y_pred_prob=y_prob_test[high_exposure_mask],
        X_features=X_test[high_exposure_mask],
        threshold_method="static",
        static_threshold=CONFIG["naive_threshold"],
        alpha=CONFIG["alpha"],
    )
    high_res_f1 = evaluator.calculate_savings(
        y_true=y_test[high_exposure_mask],
        y_pred_prob=y_prob_test[high_exposure_mask],
        X_features=X_test[high_exposure_mask],
        threshold_method="static",
        static_threshold=best_static_thresh,
        alpha=CONFIG["alpha"],
    )
    high_res_dynamic = evaluator.calculate_savings(
        y_true=y_test[high_exposure_mask],
        y_pred_prob=y_prob_test[high_exposure_mask],
        X_features=X_test[high_exposure_mask],
        threshold_method="dynamic",
        alpha=CONFIG["alpha"],
    )

    print(f"Recall (Naive 0.5):   {high_res_naive['recall']:.2%}")
    print(f"Recall (Best F1):     {high_res_f1['recall']:.2%}")
    print(f"Recall (Value-Aware): {high_res_dynamic['recall']:.2%}")

if __name__ == "__main__":
    run_experiment()