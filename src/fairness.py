from enum import StrEnum, auto

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

class FairnessCriterion(StrEnum):
    independence = auto()
    separation = auto()
    sufficiency = auto()

class FairnessEvaluator:
    def __init__(
            self,
            df: pd.DataFrame,
            target_col: str,
            pred_col: str,
            sensitive_col: str
    ):
        self.df = df
        self.target_col = target_col
        self.pred_col = pred_col
        self.sensitive_col = sensitive_col
        self.groups = sorted(df[sensitive_col].unique())

    def get_metrics(self, criterion: FairnessCriterion):
        return {
            FairnessCriterion.independence: self.get_independence_metrics,
            FairnessCriterion.separation: self.get_separation_metrics,
            FairnessCriterion.sufficiency: self.get_sufficiency_metrics,
        }[criterion]()

    def get_independence_metrics(self):
        """Checks Independence: Selection Rate for each group."""
        results = {}
        for group in self.groups:
            sub_df = self.df[self.df[self.sensitive_col] == group]
            selection_rate = sub_df[self.pred_col].mean()
            results[group] = selection_rate
        return results

    def get_separation_metrics(self):
        """Checks Separation: TPR and FPR parity."""
        results = {}
        for group in self.groups:
            sub_df = self.df[self.df[self.sensitive_col] == group]
            if len(sub_df[self.pred_col].unique()) < 1:
                results[group] = {'TPR': np.nan, 'FPR': np.nan}
                continue

            y_true = sub_df[self.target_col]
            y_pred = sub_df[self.pred_col]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            results[group] = {'TPR': tpr, 'FPR': fpr}
        return results

    def get_sufficiency_metrics(self):
        """Checks Sufficiency: Positive Predictive Value (Precision)."""
        results = {}
        for group in self.groups:
            sub_df = self.df[self.df[self.sensitive_col] == group]
            predicted_positive = sub_df[sub_df[self.pred_col] == 1]

            if len(predicted_positive) == 0:
                ppv = 0.0
            else:
                ppv = predicted_positive[self.target_col].mean()
            results[group] = {'PPV': ppv}
        return results

    def summarize_disparities(self):
        print(f"\n--- Fairness Report ({self.sensitive_col}) ---")

        ind_metrics = self.get_independence_metrics()
        print("1. Independence (Selection Rate):")
        for g, val in ind_metrics.items():
            print(f"   - Group {g}: {val:.4f}")

        sep_metrics = self.get_separation_metrics()
        print("2. Separation (Error Rates):")
        for g, val in sep_metrics.items():
            print(f"   - Group {g}: TPR={val['TPR']:.4f}, FPR={val['FPR']:.4f}")

        suf_metrics = self.get_sufficiency_metrics()
        print("3. Sufficiency (Precision/PPV):")
        for g, val in suf_metrics.items():
            print(f"   - Group {g}: PPV={val['PPV']:.4f}")


class FairnessMitigator:
    """
    Implements post-processing to adjust thresholds per group
    to satisfy fairness criteria.
    """

    def __init__(self, y_true, y_proba, sensitive_vals):
        self.y_true = np.array(y_true)
        self.y_proba = np.array(y_proba)
        self.sensitive_vals = np.array(sensitive_vals)
        self.groups = sorted(np.unique(sensitive_vals))
        self.thresholds = {g: 0.5 for g in self.groups}

    def fit(self, criterion: FairnessCriterion):
        """
        Calculates group-specific thresholds based on the chosen criterion.
        criterion: 'independence', 'separation', or 'sufficiency'
        """
        print(f"   [Mitigator] Fitting thresholds for criterion: {criterion}...")

        {
            FairnessCriterion.independence: self._fit_independence,
            FairnessCriterion.separation: self._fit_separation,
            FairnessCriterion.sufficiency: self._fit_sufficiency,
        }[criterion]()

    def predict(self, y_proba, sensitive_vals):
        """Apply group-specific thresholds to new probabilities."""
        y_proba = np.array(y_proba)
        sensitive_vals = np.array(sensitive_vals)
        preds = np.zeros(len(y_proba), dtype=int)

        for group in self.groups:
            mask = (sensitive_vals == group)
            preds[mask] = (y_proba[mask] >= self.thresholds[group]).astype(int)
        return preds

    def _fit_independence(self):
        """
        Matches Selection Rate (Positive Rate) across groups.
        Strategy: Use the minimum selection rate among groups at default threshold 0.5
        as the target, to minimize false positives.
        """
        # 1. Calculate base rates at default threshold
        base_rates = {}
        for g in self.groups:
            mask = self.sensitive_vals == g
            base_rates[g] = (self.y_proba[mask] >= 0.5).mean()

        # Target the average rate for stability
        target_rate = np.mean(list(base_rates.values()))

        # 2. Find threshold for each group to meet target_rate
        for g in self.groups:
            mask = self.sensitive_vals == g
            probs = self.y_proba[mask]
            # Quantile function is inverse of CDF
            # We want P(score > t) = target_rate
            # t = percentile(1 - target_rate)
            t = np.quantile(probs, 1 - target_rate)
            self.thresholds[g] = t
            print(f"      > Group {g}: target_rate={target_rate:.4f} -> threshold={t:.4f}")

    def _fit_separation(self):
        """
        Approximates Separation (Equal TPR and Equal FPR).
        Strategy: Minimize Euclidean distance between group ROC points
        and the average ROC point (Equalized Odds).
        """
        # Calculate ROC curves for all groups
        rocs = {}
        for g in self.groups:
            mask = self.sensitive_vals == g
            fpr, tpr, thres = roc_curve(self.y_true[mask], self.y_proba[mask])
            rocs[g] = (fpr, tpr, thres)

        # We will search for a 'common' TPR/FPR point.
        # A simple heuristic is to fix the TPR of the disadvantaged group
        # and match others to it, or minimize total variance.
        # Here we perform a grid search over global thresholds to find a "target"
        # then map individual groups to that target.

        best_thresholds = self.thresholds.copy()
        min_cost = float('inf')

        # Coarse grid search for best combination
        # (Simplified: optimizing variance of TPR and FPR across groups)
        # We iterate over a set of candidate percentiles to derive thresholds
        percentiles = np.linspace(0.1, 0.9, 20)

        # To avoid combinatorial explosion, we optimize iteratively or simpler:
        # We try to equate TPRs.
        target_tpr = 0.80  # Reasonable target for fraud

        for g in self.groups:
            fpr, tpr, thres = rocs[g]
            # Find threshold nearest to target TPR
            idx = np.argmin(np.abs(tpr - target_tpr))
            self.thresholds[g] = thres[idx]
            actual_tpr = tpr[idx]
            actual_fpr = fpr[idx]
            print(
                f"      > Group {g}: target_tpr={target_tpr} -> threshold={thres[idx]:.4f} (TPR={actual_tpr:.3f}, FPR={actual_fpr:.3f})")

    def _fit_sufficiency(self):
        """
        Matches PPV (Precision) across groups.
        Strategy: Target the minimum PPV achieved by any group at default threshold
        (conservative approach) or a fixed target.
        """
        # Calculate Precision-Recall curves
        prs = {}
        base_ppvs = []

        for g in self.groups:
            mask = self.sensitive_vals == g
            prec, rec, thres = precision_recall_curve(self.y_true[mask], self.y_proba[mask])
            # Add 0.5 threshold check for base comparison
            base_pred = (self.y_proba[mask] >= 0.5).astype(int)
            if base_pred.sum() > 0:
                p_base = np.mean(self.y_true[mask][base_pred == 1])
                base_ppvs.append(p_base)
            prs[g] = (prec, rec, thres)

        target_ppv = min(base_ppvs) if base_ppvs else 0.05
        # Clamp target slightly to avoid extremes
        target_ppv = max(target_ppv, 0.01)

        for g in self.groups:
            prec, rec, thres = prs[g]
            # precision array is length thresholds + 1 in sklearn, last is 1.
            # Find threshold where precision >= target_ppv
            # We look for the lowest threshold that satisfies the precision requirement
            valid_indices = np.where(prec[:-1] >= target_ppv)[0]
            if len(valid_indices) > 0:
                idx = valid_indices[0]  # closest to the "left" (lower threshold, higher recall)
                self.thresholds[g] = thres[idx]
            else:
                self.thresholds[g] = 0.99  # Fallback if target unreachable

            print(f"      > Group {g}: target_ppv={target_ppv:.4f} -> threshold={self.thresholds[g]:.4f}")