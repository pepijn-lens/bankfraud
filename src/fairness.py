from enum import StrEnum, auto

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

class FairnessCriterion(StrEnum):
    independence = auto()
    separation = auto()
    sufficiency = auto()

def summarize_disparities(df: pd.DataFrame, true_col: str, pred_col: str, sensitive_col: str):
    """Prints the three fairness metrics for a specific dataframe."""
    print(f"\n--- Fairness Report ({sensitive_col}) ---")

    # Partition the data based on the sensitive column
    groups = sorted(df[sensitive_col].unique())
    partitions = [(g, df[df[sensitive_col] == g]) for g in groups]

    # 1. Independence
    print("1. Independence (Selection Rate):")
    for g, sub in partitions:
        sel_rate = sub[pred_col].mean()
        print(f"   - {g}: {sel_rate:.4f}")

    # 2. Separation
    print("2. Separation (TPR/FPR):")
    for g, sub in partitions:
        y_t = sub[true_col]
        y_p = sub[pred_col]
        # Handle edge case where a group has no positives or negatives
        if len(y_p) == 0:
            print(f"   - {g}: No Data")
            continue

        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"   - {g}: TPR={tpr:.4f}, FPR={fpr:.4f}")

    # 3. Sufficiency
    print("3. Sufficiency (Precision):")
    for g, sub in partitions:
        pred_pos = sub[sub[pred_col] == 1]
        ppv = pred_pos[true_col].mean() if len(pred_pos) > 0 else 0.0
        print(f"   - {g}: PPV={ppv:.4f}")

class FairnessMitigator:

    def __init__(self, y_true, y_probs, sensitive_vals):
        # Standardize input
        self.y_true = np.array(y_true)
        self.y_probs = np.array(y_probs)
        self.sensitive_vals = np.array(sensitive_vals)
        if self.y_true.ndim != 1:
            raise ValueError(f"Unexpected ndim for y_true: {self.y_true.ndim}")
        if self.y_probs.ndim != 1:
            raise ValueError(f"Unexpected ndim for y_probs: {self.y_true.ndim}")
        if self.sensitive_vals.ndim != 1:
            raise ValueError(f"Unexpected ndim for sensitive_vals: {self.y_true.ndim}")
        if self.y_true.shape != self.y_probs.shape or self.y_probs.shape != self.sensitive_vals.shape:
            raise ValueError(f"Mismatching shapes: {self.y_true.shape}, {self.y_probs.shape}, {self.sensitive_vals.shape}")

        # Get the groups and initial thresholds
        self.groups = sorted(np.unique(sensitive_vals))
        self.thresholds = {g: 0.5 for g in self.groups}

    def predict(self, y_probs, sensitive_vals):
        """Apply group-specific thresholds to new probabilities."""
        # Standardize input
        y_probs = np.array(y_probs)
        sensitive_vals = np.array(sensitive_vals)
        if y_probs.ndim != 1:
            raise ValueError(f"Unexpected ndim for y_probs: {self.y_probs.ndim}")
        if sensitive_vals.ndim != 1:
            raise ValueError(f"Unexpected ndim for sensitive_vals: {self.y_probs.ndim}")
        if y_probs.shape != sensitive_vals.shape:
            raise ValueError(f"Mismatching shapes: {self.y_probs.shape}, {self.sensitive_vals.shape}")

        # Generate predictions using independent thresholds
        preds = np.zeros(len(y_probs), dtype=int)
        for g in self.groups:
            mask = (sensitive_vals == g)
            preds[mask] = (y_probs[mask] >= self.thresholds[g]).astype(int)
        return preds

    def fit(self, criterion: FairnessCriterion):
        print(f"   [Mitigator] Fitting for {criterion}...")
        if criterion == FairnessCriterion.independence:
            self._fit_independence()
        elif criterion == FairnessCriterion.separation:
            self._fit_separation()
        elif criterion == FairnessCriterion.sufficiency:
            self._fit_sufficiency()
        else:
            raise ValueError(f"Unexpected criterion: {criterion}")

    def _fit_independence(self):
        # Target: Global average selection rate at 0.5
        global_sel_rate = (self.y_probs >= 0.5).mean()

        for g in self.groups:
            mask = self.sensitive_vals == g
            probs = self.y_probs[mask]
            # Find threshold t such that P(prob > t) = global_sel_rate
            # t is the (1 - target) quantile
            t = np.quantile(probs, max(0, min(1, 1 - global_sel_rate)))
            self.thresholds[g] = t
            print(f"      > {g}: target_rate={global_sel_rate:.3f} -> th={t:.4f}")

    def _fit_separation(self):
        # Target: Global TPR at default threshold 0.5
        # This ensures we don't aim for an impossible TPR
        global_preds = (self.y_probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_true, global_preds).ravel()
        target_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.5

        for g in self.groups:
            mask = self.sensitive_vals == g
            fpr, tpr, thres = roc_curve(self.y_true[mask], self.y_probs[mask])

            # Find threshold resulting in TPR closest to target
            idx = np.argmin(np.abs(tpr - target_tpr))
            self.thresholds[g] = thres[idx]
            print(f"      > {g}: target_tpr={target_tpr:.3f} -> th={thres[idx]:.4f} (Actual TPR={tpr[idx]:.3f})")

    def _fit_sufficiency(self):
        # Target: Min precision (PPV) observed across groups at 0.5
        # This prevents raising the bar too high for disadvantaged groups
        base_ppvs = []
        for g in self.groups:
            mask = self.sensitive_vals == g
            preds = (self.y_probs[mask] >= 0.5).astype(int)
            if preds.sum() > 0:
                ppv = self.y_true[mask][preds == 1].mean()
                base_ppvs.append(ppv)

        target_ppv = min(base_ppvs) if base_ppvs else 0.1

        for g in self.groups:
            mask = self.sensitive_vals == g
            prec, rec, thres = precision_recall_curve(self.y_true[mask], self.y_probs[mask])

            # Find lowest threshold where precision >= target
            valid = np.where(prec[:-1] >= target_ppv)[0]
            if len(valid) > 0:
                # Use the lowest threshold (highest recall) that satisfies precision
                idx = valid[0]
                self.thresholds[g] = thres[idx]
            else:
                self.thresholds[g] = 0.95  # Fallback

            print(f"      > {g}: target_ppv={target_ppv:.3f} -> th={self.thresholds[g]:.4f}")