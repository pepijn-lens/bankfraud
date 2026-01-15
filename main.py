import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.load_data import normalize
from src.models import get_base_model, get_random_forest, get_lgbm_model
from src.evaluation import ValueAwareEvaluator
import matplotlib.pyplot as plt


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

CONFIG = {
    "data_path": "data/2/train.csv",
    "naive_threshold": 0.5,
    "n_splits": 5,
    "seed": 42,
}

MODEL_FACTORIES = {
    # "Logistic Regression": get_base_model,
    # "Random Forest": get_random_forest,
    "LightGBM": get_lgbm_model,
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
      - compute metrics on both training and validation sets

    Returns:
      results: per-fold results for each (model, decision_rule, split)
      summary: mean metrics across folds grouped by (model, decision_rule, split)
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

        # In CV, we only have train and validation (held-out fold)
        # normalize expects 3 sets, so we pass validation fold as both val and test
        X_train_scaled, X_val_scaled, _, _ = normalize(X_train, X_test, X_test)
        X_test_scaled = X_val_scaled  # Use the validation scaled data

        for model_name, factory in MODEL_FACTORIES.items():
            print("Model", model_name)
            model = factory()
            model.fit(X_train_scaled, y_train)

            # Get predictions for both train and test
            p_train = model.predict_proba(X_train_scaled)[:, 1]
            p_test = model.predict_proba(X_test_scaled)[:, 1]

            for rule in DECISION_RULES:
                print("Rule", rule["decision_rule"])
                
                # Evaluate on training set
                train_kwargs = dict(
                    y_true=y_train,
                    y_pred_prob=p_train,
                    X_features=X_train,
                    threshold_method=rule["threshold_method"],
                )
                if rule["threshold_method"] == "static":
                    train_kwargs["static_threshold"] = rule["static_threshold"]
                
                train_res = evaluator.evaluate(**train_kwargs)
                
                # Evaluate on validation set
                test_kwargs = dict(
                    y_true=y_test,
                    y_pred_prob=p_test,
                    X_features=X_test,
                    threshold_method=rule["threshold_method"],
                )
                if rule["threshold_method"] == "static":
                    test_kwargs["static_threshold"] = rule["static_threshold"]
                
                test_res = evaluator.evaluate(**test_kwargs)

                # Store training results
                rows.append({
                    "fold": fold,
                    "model": model_name,
                    "decision_rule": rule["decision_rule"],
                    "split": "train",
                    **train_res
                })
                
                # Store validation results
                rows.append({
                    "fold": fold,
                    "model": model_name,
                    "decision_rule": rule["decision_rule"],
                    "split": "validation",
                    **test_res
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
        .groupby(["model", "decision_rule", "split"])[metrics_cols]
        .mean(numeric_only=True)
        .sort_values("Total_Bank_Loss_($)")
    )

    return results, summary

def plot_total_loss_by_model_and_rule(cv_summary):
    summary = cv_summary.reset_index()
    
    # Filter to only validation data for this plot
    summary_val = summary[summary["split"] == "validation"].copy()

    pivot = summary_val.pivot(index="model", columns="decision_rule", values="Total_Bank_Loss_($)")

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
    plt.title("Total financial loss by model and decision rule (CV mean - Validation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("total_loss_by_model_and_rule.png")

def plot_train_vs_val_metrics(cv_summary, metrics=['recall', 'accuracy', 'f1']):
    """
    Plot training vs validation metrics to check for overfitting.
    
    Args:
        cv_summary: DataFrame with columns model, decision_rule, split, and metric columns
        metrics: list of metric names to plot
    """
    summary = cv_summary.reset_index()
    
    # Filter to only include metrics that exist
    available_metrics = [m for m in metrics if m in summary.columns]
    
    if not available_metrics:
        print("No valid metrics found to plot")
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        # Pivot to get train and validation side by side
        pivot = summary.pivot_table(
            index=["model", "decision_rule"],
            columns="split",
            values=metric,
            aggfunc='mean'
        )
        
        # Create grouped bar chart
        x = np.arange(len(pivot))
        width = 0.35
        
        train_vals = pivot.get("train", [])
        val_vals = pivot.get("validation", [])
        
        if len(train_vals) > 0 and len(val_vals) > 0:
            ax.bar(x - width/2, train_vals, width, label='Train', alpha=0.8)
            ax.bar(x + width/2, val_vals, width, label='Validation', alpha=0.8)
            
            # Set x-axis labels
            labels = [f"{row[0]}\n{row[1]}" for row in pivot.index]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
            
            ax.set_ylabel(metric)
            ax.set_title(f'{metric.capitalize()}: Train vs Validation')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data for {metric}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig("train_vs_val_metrics.png")


def run_experiment():
    print("1. Loading and Preprocessing Data...")
    try:
        df = pd.read_csv("data/2/train.csv")
    except FileNotFoundError:
        print("Hi there, I see you have not run the data splitting script yet. Please run `python src/load_data.py` to create the training and test sets. Then you're good to go :)")
        exit(1)

    label_col = "fraud_bool"

    cv_results, cv_summary = run_cross_validation(
        df=df,
        label_col=label_col,
        n_splits=CONFIG["n_splits"],
        seed=CONFIG["seed"],
    )

    print("\n=== CV Summary (mean across folds) ===")
    money_cols = ["Total_Bank_Loss_($)", "Fraud_Loss_($)", "False_Alarm_Cost_($)", "Fraud_Caught_($)"]
    print(cv_summary[money_cols].round(0))
    print(cv_summary[["recall", "accuracy", "f1"]].round(3))

    plot_total_loss_by_model_and_rule(cv_summary)
    
    # Plot training vs validation metrics
    plot_train_vs_val_metrics(cv_summary, metrics=['recall', 'accuracy', 'f1'])


if __name__ == "__main__":
    run_experiment()