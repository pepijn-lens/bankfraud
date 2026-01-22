import pandas as pd

from constants import DATA_DIR, TARGET_COL, RESULTS_DIR
from fairness import FairnessCriterion, FairnessMitigator, summarize_disparities
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
    print("--- 1. Loading ---")
    df = get_data(DATA_DIR)

    print("\n--- 2. Analyze & Clean ---")
    analyze_data(df)
    df = clean_data(df)

    print("\n--- 3. EDA ---")
    # We plot before encoding to keep categorical labels readable
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

    print("\n--- 4. Encoding ---")
    # Note: We encode before splitting for simplicity in this specific dataset (fixed categories).
    # In strict production with unknown categories, this should be part of a pipeline.
    df = onehot_encode(df)

    print("\n--- 5. Splitting ---")
    # 70% Train, 15% Val, 15% Test
    shuffled = shuffle_data(df, seed=42)
    train_df, val_df, test_df = split_data(shuffled, ratios=[0.70, 0.15])

    # Save raw sensitive cols for fairness analysis BEFORE they get transformed
    sensitive_cols = ["income", "customer_age"]
    val_sensitive_raw = val_df[sensitive_cols].copy()
    test_sensitive_raw = test_df[sensitive_cols].copy()

    # Calculate Bin Edges on Train (Fit) to apply to Val/Test
    bin_edges = {col: get_bin_edges(train_df[col], n_bins=3) for col in sensitive_cols}
    bin_labels = ["Low", "Medium", "High"]

    X_train, y_train = pop_target(train_df, TARGET_COL)
    X_val, y_val = pop_target(val_df, TARGET_COL)
    X_test, y_test = pop_target(test_df, TARGET_COL)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("\n--- 6. Preprocessing (Impute & Scale) ---")
    # Fits on Train ONLY, transforms Val/Test
    impute_features(X_train, X_val, X_test, cols=[
        "credit_risk_score",
        "device_distinct_emails_8w",
        "session_length_in_minutes",
        "current_address_months_count"
    ])
    # Scale all other features
    scale_features(X_train, X_val, X_test)

    print("\n--- 7. Training ---")
    model = train_model(X_train, y_train)
    evaluate_model(model, X_train, y_train, "Train")
    evaluate_model(model, X_val, y_val, "Validation")
    evaluate_model(model, X_test, y_test, "Test")

    print("\n--- 8. Fairness Evaluation ---")

    # We use Validation set to FIND thresholds, and Test set to EVALUATE them.
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # Create Base DataFrame for Test Evaluation
    eval_df_base = pd.DataFrame({TARGET_COL: y_test.values})

    for col in sensitive_cols:
        print(f"\n{'='*60}\nSENSITIVE ATTRIBUTE: {col}\n{'='*60}")

        # Bin the sensitive data for Val (Fitting) and Test (Evaluating)
        val_groups = bin_column(val_sensitive_raw[col].values, bin_edges[col], bin_labels)
        test_groups = bin_column(test_sensitive_raw[col].values, bin_edges[col], bin_labels)

        eval_df_base[col] = test_groups

        print("\n>>> BASELINE (No Mitigation, Threshold=0.5) <<<")
        default_preds = (test_probs >= 0.5).astype(int)
        eval_df_base["prediction"] = default_preds
        summarize_disparities(eval_df_base, TARGET_COL, "prediction", col)

        for criterion in FairnessCriterion:
            print(f"\n>>> MITIGATING FOR: {criterion.upper()} <<<")

            # 1. Fit Mitigator on VALIDATION data
            mitigator = FairnessMitigator(
                y_true=y_val,
                y_probs=val_probs,
                sensitive_vals=val_groups
            )
            mitigator.fit(criterion)

            # 2. Apply Thresholds to TEST data
            new_preds = mitigator.predict(test_probs, test_groups)

            # 3. Evaluate on TEST data
            eval_df_mitigated = eval_df_base.copy()
            eval_df_mitigated["prediction"] = new_preds
            summarize_disparities(eval_df_mitigated, TARGET_COL, "prediction", col)

if __name__ == "__main__":
    # Run main experiment
    main()
    

