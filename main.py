import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from src.load_data import preprocess_data, normalize
from src.models import get_base_model
from src.evaluation import ValueAwareEvaluator
import matplotlib.pyplot as plt


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

def run_cross_validation(df, label_col, alpha=1.0, n_splits=5, seed=42):
    """
    Runs stratified K-fold cross-validation and return per-fold results for each method.
    """
    X = df.drop(columns=[label_col])
    y = df[label_col].to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows = []
    print("Running stratified K-fold CV...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Fold {fold}")
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y[val_idx]

        X_train_scaled, X_val_scaled, _, _ = normalize(X_train, X_val, X_val)

        model = get_base_model()
        model.fit(X_train_scaled, y_train)

        p_val = model.predict_proba(X_val_scaled)[:, 1]
        best_f1_thresh = find_best_f1_threshold(y_val, p_val)

        evaluator = ValueAwareEvaluator()
        evaluator.fit(X_train, y_train)

        print("Static evaluator with 0.5 threshold.")
        res_naive = evaluator.calculate_savings(
            y_true=y_val,
            y_pred_prob=p_val,
            X_features=X_val,
            threshold_method="static",
            static_threshold=0.5,
            alpha=alpha,
        )
        print("Static evaluator with fitted threshold.")
        res_f1 = evaluator.calculate_savings(
            y_true=y_val,
            y_pred_prob=p_val,
            X_features=X_val,
            threshold_method="static",
            static_threshold=best_f1_thresh,
            alpha=alpha,
        )
        print("Value-aware evaluator.")
        res_dynamic = evaluator.calculate_savings(
            y_true=y_val,
            y_pred_prob=p_val,
            X_features=X_val,
            threshold_method="dynamic",
            alpha=alpha,
        )

        rows.append({"fold": fold, "method": "Naive (0.5)", "best_f1_thresh": best_f1_thresh, **res_naive})
        rows.append({"fold": fold, "method": "Best F1", "best_f1_thresh": best_f1_thresh, **res_f1})
        rows.append({"fold": fold, "method": "Value-Aware", "best_f1_thresh": best_f1_thresh, **res_dynamic})

    results = pd.DataFrame(rows)

    summary = (
        results.groupby("method")[["Total_Bank_Loss_($)", "Fraud_Loss_($)", "False_Alarm_Cost_($)", "Fraud_Caught_($)", "recall", "accuracy", "f1"]]
        .mean()
        .sort_values("Total_Bank_Loss_($)")
    )

    return results, summary

def run_experiment():
    print("1. Loading and Preprocessing Data...")
    df = pd.read_csv(CONFIG["data_path"])

    df = preprocess_data(df)
    label_col = "fraud_bool"
    alphas = [CONFIG["alpha"], 0.01, 0.05, 0.10]
    alphas = sorted(set(alphas))

    all_alpha_summaries = []

    for a in alphas:
        print(f"Running stratified K-fold CV for alpha {a}")
        cv_results, cv_summary = run_cross_validation(
            df=df,
            label_col=label_col,
            alpha=a
        )

        cv_summary = cv_summary.copy()
        cv_summary.insert(0, "alpha", a)
        cv_summary.insert(1, "method", cv_summary.index)
        cv_summary = cv_summary.reset_index(drop=True)

        all_alpha_summaries.append(cv_summary)

        # -------------------------
        # RQ1: Overall test set
        # -------------------------
        if a == CONFIG["alpha"]:
            print(f"\n=== RQ1 (CV): Average results across folds (alpha={a}) ===")
            with pd.option_context("display.max_columns", None, "display.width", None):
                money_cols = ["Total_Bank_Loss_($)", "Fraud_Loss_($)", "False_Alarm_Cost_($)", "Fraud_Caught_($)"]
                print(cv_summary[["method"] + money_cols].round(0).set_index("method"))
                print(cv_summary[["method", "recall", "accuracy", "f1"]].round(3).set_index("method"))

    # Sensitivity analysis table across alpha
    print("\n=== Sensitivity analysis (CV): alpha in {0.01, 0.05, 0.10} ===")
    alpha_summary_df = pd.concat(all_alpha_summaries, ignore_index=True)

    # keep output compact and comparable
    cols = [
        "alpha", "method",
        "Total_Bank_Loss_($)",
        "Fraud_Loss_($)",
        "False_Alarm_Cost_($)",
        "recall",
        "accuracy",
        "f1",
    ]
    print(alpha_summary_df[cols].round(3))
    df_plot = alpha_summary_df.sort_values("alpha")

    plt.figure(figsize=(8.5, 5.2))

    for method, g in df_plot.groupby("method"):
        plt.plot(
            g["alpha"],
            g["Total_Bank_Loss_($)"],
            marker="o",
            linewidth=2,
            label=method,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"False-alarm cost scaling $\alpha$ (log scale)")
    plt.ylabel("Total loss (log scale)")
    plt.title("Total Loss vs False-Alarm Cost Scaling")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # # -------------------------
    # # RQ1: Overall test set
    # # -------------------------
    # print("\n=== RQ1: Total Cost Analysis (Lower is Better) ===")
    #
    # res_naive = evaluator.calculate_savings(
    #     y_true=y_test,
    #     y_pred_prob=y_prob_test,
    #     X_features=X_test,
    #     threshold_method="static",
    #     static_threshold=CONFIG["naive_threshold"],
    #     alpha=CONFIG["alpha"],
    # )
    # res_f1 = evaluator.calculate_savings(
    #     y_true=y_test,
    #     y_pred_prob=y_prob_test,
    #     X_features=X_test,
    #     threshold_method="static",
    #     static_threshold=best_static_thresh,
    #     alpha=CONFIG["alpha"],
    # )
    # res_dynamic = evaluator.calculate_savings(
    #     y_true=y_test,
    #     y_pred_prob=y_prob_test,
    #     X_features=X_test,
    #     threshold_method="dynamic",
    #     alpha=CONFIG["alpha"],
    # )
    #
    # results_df = pd.DataFrame(
    #     [res_naive, res_f1, res_dynamic],
    #     index=["Naive (0.5)", "Best F1", "Value-Aware"],
    # )
    #
    # money_cols = ["Total_Bank_Loss_($)", "Fraud_Loss_($)", "False_Alarm_Cost_($)", "Fraud_Caught_($)"]
    # print(results_df[money_cols].round(0))
    # print(results_df[["recall", "accuracy", "f1"]].round(3))
    #
    # # --- Sensitivity analysis over alpha ---
    # print("\n=== Sensitivity Analysis: Varying alpha ===")
    #
    # alphas = [0.01, 0.05, 0.10]
    # rows = []
    #
    # for a in alphas:
    #     res_naive = evaluator.calculate_savings(
    #         y_true=y_test,
    #         y_pred_prob=y_prob_test,
    #         X_features=X_test,
    #         threshold_method="static",
    #         static_threshold=CONFIG["naive_threshold"],
    #         alpha=a,
    #     )
    #     res_f1 = evaluator.calculate_savings(
    #         y_true=y_test,
    #         y_pred_prob=y_prob_test,
    #         X_features=X_test,
    #         threshold_method="static",
    #         static_threshold=best_static_thresh,
    #         alpha=a,
    #     )
    #     res_dynamic = evaluator.calculate_savings(
    #         y_true=y_test,
    #         y_pred_prob=y_prob_test,
    #         X_features=X_test,
    #         threshold_method="dynamic",
    #         alpha=a,
    #     )
    #
    #     rows.extend([
    #         {"alpha": a, "method": "Naive (0.5)", **res_naive},
    #         {"alpha": a, "method": "Best F1", **res_f1},
    #         {"alpha": a, "method": "Value-Aware", **res_dynamic},
    #     ])
    #
    # alpha_results = pd.DataFrame(rows)
    #
    # print(
    #     alpha_results[
    #         ["alpha", "method",
    #          "Total_Bank_Loss_($)",
    #          "Fraud_Loss_($)",
    #          "False_Alarm_Cost_($)",
    #          "recall"]
    #     ].round(3)
    # )
    #
    #
    # # -------------------------
    # # RQ2: High-exposure segment
    # # -------------------------
    # print("\n=== RQ2: High-exposure cases (top quantile by proposed credit limit) ===")
    #
    # q = CONFIG["high_exposure_quantile"]
    # cutoff = X_test["proposed_credit_limit"].quantile(q)
    # high_exposure_mask = X_test["proposed_credit_limit"] >= cutoff
    #
    # print(
    #     f"High-exposure definition: proposed_credit_limit >= {cutoff:.2f} "
    #     f"(top {(1 - q) * 100:.0f}% of applications)"
    # )
    #
    # high_res_naive = evaluator.calculate_savings(
    #     y_true=y_test[high_exposure_mask],
    #     y_pred_prob=y_prob_test[high_exposure_mask],
    #     X_features=X_test[high_exposure_mask],
    #     threshold_method="static",
    #     static_threshold=CONFIG["naive_threshold"],
    #     alpha=CONFIG["alpha"],
    # )
    # high_res_f1 = evaluator.calculate_savings(
    #     y_true=y_test[high_exposure_mask],
    #     y_pred_prob=y_prob_test[high_exposure_mask],
    #     X_features=X_test[high_exposure_mask],
    #     threshold_method="static",
    #     static_threshold=best_static_thresh,
    #     alpha=CONFIG["alpha"],
    # )
    # high_res_dynamic = evaluator.calculate_savings(
    #     y_true=y_test[high_exposure_mask],
    #     y_pred_prob=y_prob_test[high_exposure_mask],
    #     X_features=X_test[high_exposure_mask],
    #     threshold_method="dynamic",
    #     alpha=CONFIG["alpha"],
    # )
    #
    # print(f"Recall (Naive 0.5):   {high_res_naive['recall']:.2%}")
    # print(f"Recall (Best F1):     {high_res_f1['recall']:.2%}")
    # print(f"Recall (Value-Aware): {high_res_dynamic['recall']:.2%}")

if __name__ == "__main__":
    run_experiment()