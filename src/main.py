from constants import DATA_DIR, TARGET_COL, RESULTS_DIR
from utils import (
    get_data,
    clean_data,
    analyze_data,
    onehot_encode,
    pop_target,
    plot_categorical_fraud_rates,
    train_model,
    evaluate_model, impute_features, scale_features, shuffle_data, split_data
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
    # Using scale_pos_weight=99 as per the notebook analysis
    model = train_model(X_train, y_train, scale_pos_weight=99.0)
    evaluate_model(model, X_train, y_train)

    # 8. Evaluation
    print("\n--- 8. Evaluation on Validation Set ---")
    evaluate_model(model, X_val, y_val)


if __name__ == "__main__":
    main()