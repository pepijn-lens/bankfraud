import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.load_data import preprocess_data, normalize
from src.models import get_base_model, get_random_forest
from src.evaluation import ValueAwareEvaluator
import matplotlib.pyplot as plt


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

CONFIG = {
    "data_path": "data/2/Base.csv",
    "naive_threshold": 0.5,
    "n_splits": 5,
    "seed": 42,
}

MODEL_FACTORIES = {
    "Logistic Regression": get_base_model,
    "Random Forest": get_random_forest,
}

DECISION_RULES = [
    {"decision_rule": "Static (0.5)", "threshold_method": "static", "static_threshold": 0.5},
    {"decision_rule": "Value-Aware", "threshold_method": "dynamic"},
]


def run_cross_validation(df, label_col, n_splits=5, seed=42):
    """
    Stratified K-fold CV. For each fold:
      - scale using training fold only
      - fit each model
      - evaluate two decision rules on the same predicted probabilities

    Returns:
      results: per-fold results for each (model, decision_rule)
      summary: mean metrics across folds grouped by (model, decision_rule)
    """
    X = df.drop(columns=[label_col])
    y = df[label_col].to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows = []
    evaluator = ValueAwareEvaluator()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print("Fold", fold)
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        X_train_scaled, X_test_scaled, _, *_ = normalize(X_train, X_test, X_test)

        for model_name, factory in MODEL_FACTORIES.items():
            print("Model", model_name)
            model = factory()
            model.fit(X_train_scaled, y_train)

            p_test = model.predict_proba(X_test_scaled)[:, 1]

            for rule in DECISION_RULES:
                print("Rule", rule["decision_rule"])
                kwargs = dict(
                    y_true=y_test,
                    y_pred_prob=p_test,
                    X_features=X_test,
                    threshold_method=rule["threshold_method"],
                )
                if rule["threshold_method"] == "static":
                    kwargs["static_threshold"] = rule["static_threshold"]

                res = evaluator.evaluate(**kwargs)

                rows.append({
                    "fold": fold,
                    "model": model_name,
                    "decision_rule": rule["decision_rule"],
                    **res
                })

    results = pd.DataFrame(rows)

    metrics_cols = [
        "Total_Bank_Loss_($)",
        "Fraud_Loss_($)",
        "False_Alarm_Cost_($)",
        "Fraud_Caught_($)",
        "recall",
        "accuracy",
        "f1",
    ]
    metrics_cols = [c for c in metrics_cols if c in results.columns]

    summary = (
        results
        .groupby(["model", "decision_rule"])[metrics_cols]
        .mean(numeric_only=True)
        .sort_values("Total_Bank_Loss_($)")
    )

    return results, summary

def plot_total_loss_by_model_and_rule(cv_summary):
    summary = cv_summary.reset_index()

    pivot = summary.pivot(index="model", columns="decision_rule", values="Total_Bank_Loss_($)")

    models = pivot.index.tolist()
    rules = pivot.columns.tolist()

    x = np.arange(len(models))
    width = 0.35 if len(rules) == 2 else 0.8 / max(len(rules), 1)

    plt.figure(figsize=(8, 4))

    for j, rule in enumerate(rules):
        offset = (j - (len(rules) - 1) / 2) * width
        plt.bar(x + offset, pivot[rule].values, width, label=rule)

    plt.xticks(x, models, rotation=15, ha="right")
    plt.ylabel("Total Bank Loss ($)")
    plt.title("Total financial loss by model and decision rule (CV mean)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_experiment():
    print("1. Loading and Preprocessing Data...")
    df = pd.read_csv(CONFIG["data_path"])
    df = preprocess_data(df)

    cv_results, cv_summary = run_cross_validation(
        df=df,
        label_col="fraud_bool",
        n_splits=CONFIG["n_splits"],
        seed=CONFIG["seed"],
    )

    print("\n=== CV Summary (mean across folds) ===")
    money_cols = ["Total_Bank_Loss_($)", "Fraud_Loss_($)", "False_Alarm_Cost_($)", "Fraud_Caught_($)"]
    print(cv_summary[money_cols].round(0))
    print(cv_summary[["recall", "accuracy", "f1"]].round(3))

    plot_total_loss_by_model_and_rule(cv_summary)


if __name__ == "__main__":
    run_experiment()
