import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


class FairnessEvaluator:
    def __init__(
            self,
            df: pd.DataFrame,
            target_col: str,
            pred_col: str,
            sensitive_col: str
    ):
        """
        df: Pandas DataFrame containing the data
        target_col: Name of the column with true labels (e.g., 'fraud_bool')
        pred_col: Name of the column with binary model predictions
        sensitive_col: Name of the sensitive attribute (e.g., 'age_group')
        """
        self.df = df
        self.target_col = target_col
        self.pred_col = pred_col
        self.sensitive_col = sensitive_col
        self.groups = sorted(df[sensitive_col].unique())

    def get_independence_metrics(self):
        """
        Checks Independence: P(R=1 | A=a)
        Returns the 'Flag Rate' (Selection Rate) for each group.
        """
        results = {}
        for group in self.groups:
            sub_df = self.df[self.df[self.sensitive_col] == group]
            # Probability of predicting fraud (R=1)
            selection_rate = sub_df[self.pred_col].mean()
            results[group] = selection_rate
        return results

    def get_separation_metrics(self):
        """
        Checks Separation: TPR and FPR parity.
        P(R=1 | Y=1, A=a) -> True Positive Rate
        P(R=1 | Y=0, A=a) -> False Positive Rate
        """
        results = {}
        for group in self.groups:
            sub_df = self.df[self.df[self.sensitive_col] == group]

            # Check if group has both classes to avoid division by zero
            if len(sub_df[self.pred_col].unique()) < 1:
                results[group] = {'TPR': np.nan, 'FPR': np.nan}
                continue

            y_true = sub_df[self.target_col]
            y_pred = sub_df[self.pred_col]

            # Calculate TPR and FPR
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            results[group] = {'TPR': tpr, 'FPR': fpr}
        return results

    def get_sufficiency_metrics(self):
        """
        Checks Sufficiency: Positive Predictive Value (Precision).
        P(Y=1 | R=1, A=a)
        """
        results = {}
        for group in self.groups:
            sub_df = self.df[self.df[self.sensitive_col] == group]

            # Filter to only those predicted as positive (R=1)
            predicted_positive = sub_df[sub_df[self.pred_col] == 1]

            if len(predicted_positive) == 0:
                ppv = 0.0
            else:
                # Calculate precision (how many predicted frauds were actual frauds)
                ppv = predicted_positive[self.target_col].mean()

            results[group] = {'PPV': ppv}
        return results

    def summarize_disparities(self):
        """Prints a summary of disparities."""
        print("Independence (Selection Rate)")
        ind_metrics = self.get_independence_metrics()
        for g, val in ind_metrics.items():
            print(f"- Group {g}: {val:.4f}")

        print("\nSeparation (Error Rates)")
        sep_metrics = self.get_separation_metrics()
        for g, val in sep_metrics.items():
            print(f"- Group {g}: TPR={val['TPR']:.4f}, FPR={val['FPR']:.4f}")

        print("\nSufficiency (Precision/PPV)")
        suf_metrics = self.get_sufficiency_metrics()
        for g, val in suf_metrics.items():
            print(f"- Group {g}: PPV={val['PPV']:.4f}")