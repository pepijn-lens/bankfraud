import pandas as pd

from constants import DATA_DIR, TARGET_COL, RESULTS_DIR
from fairness import FairnessEvaluator, FairnessCriterion, FairnessMitigator
from utils import (
    get_data,
    clean_data,
    analyze_data,
    onehot_encode,
    pop_target,
    plot_categorical_fraud_rates,
    train_model,
    evaluate_model, impute_features, scale_features, shuffle_data, split_data, get_bin_edges, bin_column
)


def main():
    # 1. Load Data
    print("--- 1. Loading Data ---")
    df = get_data(DATA_DIR)

    # 2. Initial Analysis & Cleaning
    print("\n--- 2. Analysis & Cleaning ---")
    analyze_data(df)
    df = clean_data(df)

    # 3. Exploratory Data Analysis (EDA)
    # We plot before encoding to keep categorical labels readable
    print("\n--- 3. EDA Plots ---")
    plot_categorical_fraud_rates(
        df,
        output_dir=RESULTS_DIR,
        target_col=TARGET_COL,
        categorical_cols=[
            *df.select_dtypes(include=['object']).columns,
            "income",
            "customer_age",
        ]
    )

    # 4. Encoding
    # Note: We encode before splitting for simplicity in this specific dataset (fixed categories).
    # In strict production with unknown categories, this should be part of a pipeline.
    print("\n--- 4. Feature Encoding ---")
    df = onehot_encode(df)

    # 5. Split Data
    print("\n--- 5. Splitting Data ---")
    # Split into Train (75%), Val (12.5%), Test (12.5%)
    shuffled_df = shuffle_data(df, seed=42)
    train_df, val_df, test_df = split_data(shuffled_df, ratios=[0.75, 0.125])

    # Save sensitive info for fairness analysis
    sensitive_cols = ["income", "customer_age"]
    val_sensitive_raw = val_df[sensitive_cols].copy()

    # Create bin edges on training data to ensure consistency
    bin_edges = {}
    for col in sensitive_cols:
        bin_edges[col] = get_bin_edges(train_df[col], n_bins=3)

    # Separate Features and Target
    X_train, y_train = pop_target(train_df, TARGET_COL)
    X_val, y_val = pop_target(val_df, TARGET_COL)
    X_test, y_test = pop_target(test_df, TARGET_COL)

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # 6. Preprocessing (Impute & Scale)
    # Fits on Train ONLY, transforms Val/Test
    print("\n--- 6. Preprocessing (Impute & Scale) ---")
    impute_features(X_train, X_val, X_test, cols=[
        "credit_risk_score",
        "device_distinct_emails_8w",
        "session_length_in_minutes",
        "current_address_months_count"
    ])
    scale_features(X_train, X_val, X_test)

    # 7. Training
    print("\n--- 7. Training ---")
    model = train_model(X_train, y_train, scale_pos_weight=99.0)
    evaluate_model(model, X_train, y_train)

    # 8. Evaluation
    print("\n--- 8. Evaluation on Validation Set ---")
    evaluate_model(model, X_val, y_val)

    # 9. Fairness Evaluation
    print("\n--- 9. Fairness Evaluation ---")

    # Get Probability predictions (needed for threshold adjustment)
    val_probs = model.predict_proba(X_val)[:, 1]

    # Prepare DataFrame for Fairness Analysis
    # We map raw sensitive values to bins (Low, Medium, High)
    fairness_df_base = pd.DataFrame({
        TARGET_COL: y_val.values
    })

    labels = ["Low", "Medium", "High"]

    for col in sensitive_cols:
        print(f"\n{'=' * 60}")
        print(f"SENSITIVE ATTRIBUTE: {col}")
        print(f"{'=' * 60}")

        # Get the binned sensitive values for this column
        sensitive_vals = bin_column(
            val_sensitive_raw[col].values,
            bin_edges[col],
            labels=labels
        )
        fairness_df_base["sensitive_current"] = sensitive_vals

        for criterion in FairnessCriterion:
            print(f"\n>>> MITIGATING FOR: {criterion.upper()} <<<")

            # Fit Mitigator on Validation Data
            mitigator = FairnessMitigator(
                y_true=y_val,
                y_proba=val_probs,
                sensitive_vals=sensitive_vals
            )
            mitigator.fit(criterion)

            # Apply optimized thresholds
            new_preds = mitigator.predict(val_probs, sensitive_vals)

            eval_df = fairness_df_base.copy()
            eval_df["prediction"] = new_preds

            evaluator = FairnessEvaluator(
                df=eval_df,
                target_col=TARGET_COL,
                pred_col="prediction",
                sensitive_col="sensitive_current"
            )

            evaluator.summarize_disparities()

if __name__ == "__main__":
    main()