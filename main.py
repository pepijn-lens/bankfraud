import pandas as pd
import numpy as np
from typing import Dict
from sklearn.model_selection import StratifiedKFold

from src.binning import BinningStrat, BinningTransformer
from src.constants import DATA_DIR, TARGET_COL
from src.load_data import preprocess_global, preprocess_fold, prepare_data
from src.models import get_base_model, get_random_forest, get_xgb_model
from src.evaluation import ValueAwareEvaluator
from src.training import train_and_save_classifiers, evaluate_on_test_set, train_logistic_regression
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import os

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

CONFIG = {
    "data_path": DATA_DIR / "2" / "Base.csv",
    "naive_threshold": 0.5,
    "n_splits": 5,
    "seed": 42,
}

MODEL_FACTORIES = {
    "Logistic Regression": get_base_model,
    "Random Forest": get_random_forest,
    "XGBoost": get_xgb_model,
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
        X_train_scaled, X_test_scaled, _ = preprocess_fold(X_train, X_test, X_test)

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
    df = preprocess_global(df)

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


def run_age_fairness_analysis(
    df: pd.DataFrame,
    label_col: str = "fraud_bool",
    age_col: str = "customer_age",
    n_splits: int = 5,
    seed: int = 42,
    exclude_age_from_features: bool = False,
    n_bins: int = 4,
    binning_method: str = "quantile"
):
    """
    Run age fairness analysis (RQ5) across CV folds.
    
    Evaluates:
    - Flag rate (selection rate) per age group
    - TPR/FPR per age group (equal opportunity/equalized odds)
    - Loss per age group (FN loss and FP cost)
    - Statistical significance via bootstrap CIs
    """
    from src.age_fairness import GroupFairnessAnalyzer
    from sklearn.model_selection import StratifiedKFold
    
    X = df.drop(columns=[label_col])
    y = df[label_col].to_numpy()
    
    # Store age separately if we're excluding it from training
    age_data = None
    if exclude_age_from_features:
        if age_col in X.columns:
            age_data = X[age_col].copy()  # Store age data
            X = X.drop(columns=[age_col])
            print(f"Ablation: Training without '{age_col}' feature")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if binning_method == "quantile":
        strat = BinningStrat.QUANTILE
    elif binning_method == "uniform":
        strat = BinningStrat.UNIFORM
    else:
        raise ValueError(f"Unexpected binning_method: {binning_method}")
    binner = BinningTransformer(n_bins=n_bins, strategy=strat)

    analyzer = GroupFairnessAnalyzer(
        group_col=age_col,
        binning_transformer=binner,
        random_state=seed
    )
    evaluator = ValueAwareEvaluator()
    
    all_cv_results = {
        "static": [],
        "value_aware": []
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Fold {fold}/{n_splits}")
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]
        
        # Normalize BEFORE adding age back
        X_train_scaled, X_test_scaled, _ = preprocess_fold(X_train, X_test, X_test)
        
        # Now add age back to X_test for grouping (but NOT to X_test_scaled)
        if exclude_age_from_features and age_data is not None:
            X_test = X_test.copy()
            X_test[age_col] = age_data.iloc[test_idx].values
        
        # Train model
        model = get_base_model()
        model.fit(X_train_scaled, y_train)
        p_test = model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate static rule
        y_pred_static = evaluator.predict_static(p_test, static_threshold=0.5)
        metrics_static = analyzer.evaluate_cv_fold(
            y_true=y_test,
            y_pred=y_pred_static,
            y_pred_prob=p_test,
            X_test=X_test,
        )
        metrics_static["fold"] = fold
        metrics_static["decision_rule"] = "Static (0.5)"
        all_cv_results["static"].append(metrics_static)
        
        # Evaluate value-aware rule
        y_pred_value_aware = evaluator.predict_value_aware(p_test, X_test)
        metrics_va = analyzer.evaluate_cv_fold(
            y_true=y_test,
            y_pred=y_pred_value_aware,
            y_pred_prob=p_test,
            X_test=X_test,
        )
        metrics_va["fold"] = fold
        metrics_va["decision_rule"] = "Value-Aware"
        all_cv_results["value_aware"].append(metrics_va)
    
    # Aggregate results
    results = {}
    for rule_name, fold_results in all_cv_results.items():
        aggregated = analyzer.aggregate_cv_results(fold_results, compute_cis=True)
        results[rule_name] = aggregated
    
    return results


def run_income_fairness_analysis(
        df: pd.DataFrame,
        label_col: str = "fraud_bool",
        income_col: str = "income",
        n_splits: int = 5,
        seed: int = 42,
        n_bins: int = 4,
        binning_method: str = "quantile"
):
    """
    Run income fairness analysis across CV folds.
    """
    from src.age_fairness import GroupFairnessAnalyzer
    from sklearn.model_selection import StratifiedKFold

    X = df.drop(columns=[label_col])
    y = df[label_col].to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if binning_method == "quantile":
        strat = BinningStrat.QUANTILE
    elif binning_method == "uniform":
        strat = BinningStrat.UNIFORM
    else:
        raise ValueError(f"Unexpected binning_method: {binning_method}")
    binner = BinningTransformer(n_bins=n_bins, strategy=strat)

    analyzer = GroupFairnessAnalyzer(
        group_col=income_col,
        binning_transformer=binner,
        random_state=seed
    )
    evaluator = ValueAwareEvaluator()

    all_cv_results = {
        "static": [],
        "value_aware": []
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Fold {fold}/{n_splits}")
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        # Normalize
        X_train_scaled, X_test_scaled, _ = preprocess_fold(X_train, X_test, X_test)

        # Train model
        model = get_base_model()
        model.fit(X_train_scaled, y_train)
        p_test = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate static rule
        y_pred_static = evaluator.predict_static(p_test, static_threshold=0.5)
        metrics_static = analyzer.evaluate_cv_fold(
            y_true=y_test,
            y_pred=y_pred_static,
            y_pred_prob=p_test,
            X_test=X_test,
        )
        metrics_static["fold"] = fold
        metrics_static["decision_rule"] = "Static (0.5)"
        all_cv_results["static"].append(metrics_static)

        # Evaluate value-aware rule
        y_pred_value_aware = evaluator.predict_value_aware(p_test, X_test)
        metrics_va = analyzer.evaluate_cv_fold(
            y_true=y_test,
            y_pred=y_pred_value_aware,
            y_pred_prob=p_test,
            X_test=X_test,
        )
        metrics_va["fold"] = fold
        metrics_va["decision_rule"] = "Value-Aware"
        all_cv_results["value_aware"].append(metrics_va)

    # Aggregate results
    results = {}
    for rule_name, fold_results in all_cv_results.items():
        aggregated = analyzer.aggregate_cv_results(fold_results, compute_cis=True)
        results[rule_name] = aggregated

    return results


def print_age_fairness_summary(results: Dict):
    """
    Print a formatted summary of age fairness analysis results.
    
    """
    for rule_name, rule_results in results.items():
        print(f"\n{'='*60}")
        print(f"Decision Rule: {rule_name.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        
        mean_metrics = rule_results["mean_metrics"]
        std_metrics = rule_results["std_metrics"]
        ci_metrics = rule_results.get("ci_metrics", pd.DataFrame())
        
        print("\nMean Metrics per Age Group:")
        print(mean_metrics.round(4))
        
        if not std_metrics.empty:
            print("\nStandard Deviation per Age Group:")
            print(std_metrics.round(4))
        
        if not ci_metrics.empty:
            print("\n95% Bootstrap Confidence Intervals:")
            print(ci_metrics.round(4))
        
        # Compute disparities if multiple groups
        if len(mean_metrics) > 1:
            from src.age_fairness import compute_disparity_metrics
            combined = mean_metrics.reset_index()
            disparities = compute_disparity_metrics(combined)
            if not disparities.empty:
                print("\nDisparity Metrics (relative to largest group):")
                print(disparities.round(4))

if __name__ == "__main__":
    # # Run old experiment with underfitted classifiers
    # run_experiment()

    # Models are already saved in the models/ directory and will be automatically
    # loaded by evaluate_on_test_set() below.
    #
    # Only uncomment the line below if you need to retrain models with different
    # hyperparameters or if models are missing.
    # fitted_models, X_test, y_test = train_and_save_classifiers()
    
    # Evaluate on test set (automatically loads existing models from models/ directory)
    evaluate_on_test_set()
    
    # # Run age fairness analysis (RQ5)
    # print("\n" + "="*60)
    # print("RQ5: Age Fairness Analysis")
    # print("="*60)
    
    # df = pd.read_csv(CONFIG["data_path"])
    # df = preprocess_global(df)
    
    # # With age included
    # print("\n--- Analysis WITH age as feature ---")
    # results_with_age = run_age_fairness_analysis(
    #     df=df,
    #     label_col="fraud_bool",
    #     exclude_age_from_features=False,
    #     n_splits=CONFIG["n_splits"],
    #     seed=CONFIG["seed"]
    # )
    # print_age_fairness_summary(results_with_age)
    
    # # Ablation: without age
    # print("\n--- Ablation: WITHOUT age as feature ---")
    # results_without_age = run_age_fairness_analysis(
    #     df=df,
    #     label_col="fraud_bool",
    #     exclude_age_from_features=True,
    #     n_splits=CONFIG["n_splits"],
    #     seed=CONFIG["seed"]
    # )
    # print_age_fairness_summary(results_without_age)

    # # Run income fairness analysis
    # print("\n" + "=" * 60)
    # print("Income Fairness Analysis")
    # print("=" * 60)

    # results_income = run_income_fairness_analysis(
    #     df=df,
    #     label_col="fraud_bool",
    #     income_col="income",  # Ensure this matches your dataset column name
    #     n_splits=CONFIG["n_splits"],
    #     seed=CONFIG["seed"]
    # )
    # # Re-using the summary printer as the structure is identical
    # print_age_fairness_summary(results_income)
