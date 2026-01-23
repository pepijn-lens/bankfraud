import pandas as pd

from analyze_data import get_data
# Use package-style imports so this script can be executed from project root
from src.constants import DATA_DIR, TARGET_COL, RESULTS_DIR
from src.fairness import FairnessCriterion, FairnessMitigator, summarize_disparities



def main():
    ... # originally there was code here to load and split the data, and train/evaluate the model.

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
    

