import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.metrics import confusion_matrix
from scipy import stats
import warnings

from binning import BinningTransformer


def compute_disparity_metrics(
    group_metrics: pd.DataFrame,
    reference_group: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute disparity measures relative to a reference group.

    If reference_group is None, uses the group with largest n as reference.

    """
    if len(group_metrics) == 0:
        return pd.DataFrame()

    if reference_group is None:
        reference_group = group_metrics.loc[group_metrics["n"].idxmax(), "group"]

    ref_metrics = group_metrics[group_metrics["group"] == reference_group]
    if len(ref_metrics) == 0:
        warnings.warn(f"Reference group '{reference_group}' not found. Using first group.")
        reference_group = group_metrics.iloc[0]["group"]
        ref_metrics = group_metrics[group_metrics["group"] == reference_group]

    ref_metrics = ref_metrics.iloc[0]

    disparities = []
    for _, row in group_metrics.iterrows():
        if row["group"] == reference_group:
            continue

        disparities.append({
            "group": row["group"],
            "reference_group": reference_group,
            "flag_rate_ratio": row["flag_rate"] / ref_metrics["flag_rate"] if ref_metrics["flag_rate"] > 0 else np.nan,
            "flag_rate_diff": row["flag_rate"] - ref_metrics["flag_rate"],
            "tpr_ratio": row["tpr"] / ref_metrics["tpr"] if ref_metrics["tpr"] > 0 else np.nan,
            "tpr_diff": row["tpr"] - ref_metrics["tpr"],
            "fpr_ratio": row["fpr"] / ref_metrics["fpr"] if ref_metrics["fpr"] > 0 else np.nan,
            "fpr_diff": row["fpr"] - ref_metrics["fpr"],
            "total_loss_ratio": row["total_loss"] / ref_metrics["total_loss"] if ref_metrics["total_loss"] > 0 else np.nan,
            "total_loss_diff": row["total_loss"] - ref_metrics["total_loss"],
        })

    return pd.DataFrame(disparities)


class GroupFairnessAnalyzer:
    """
    Analyzer for evaluating fairness across groups (e.g., Age, Gender).

    Evaluates:
    - Flag rate (selection rate) per group
    - TPR/FPR per group (equal opportunity/equalized odds)
    - Loss per group (FN loss and FP cost)
    - Statistical significance via bootstrap CIs or paired tests
    """

    def __init__(
        self,
        group_col: str,
        binning_transformer: Optional[BinningTransformer] = None,
        random_state: int = 42
    ):
        """
        Args:
            group_col: Name of the column to group by (e.g., "customer_age").
            binning_transformer: Instance of BinningTransformer.
                                 If None, assumes group_col is already categorical.
                                 If provided, it will be used to discretize the group_col.
            random_state: Random seed for bootstrap.
        """
        self.group_col = group_col
        self.binning_transformer = binning_transformer
        self.random_state = random_state

    def _get_group_labels(self, X: pd.DataFrame) -> pd.Series:
        """
        Extracts and optionally bins the grouping column from the data.
        """
        if self.group_col not in X.columns:
            raise ValueError(f"Group column '{self.group_col}' not found.")

        raw_values = X[self.group_col]
        if self.binning_transformer is None:
            return raw_values.astype(str)
        return self.binning_transformer.fit_transform(raw_values)

    def compute_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_prob: Optional[np.ndarray] = None,
        groups: Optional[pd.Series] = None,
        X_features: Optional[pd.DataFrame] = None,
        credit_col: str = "proposed_credit_limit",
        ops_cost: float = 100.0,
        margin_rate: float = 0.05
    ) -> pd.DataFrame:
        """
        Compute fairness metrics per age group.
        
        Metrics:
        - Flag rate (selection rate): P(y_pred=1 | group)
        - TPR (True Positive Rate): P(y_pred=1 | y_true=1, group) - Equal Opportunity
        - FPR (False Positive Rate): P(y_pred=1 | y_true=0, group) - Equalized Odds component
        - FN loss: Total fraud loss from false negatives
        - FP cost: Total cost from false positives
        - Total loss: FN loss + FP cost
        
        Args:
            y_true: True labels (0=legit, 1=fraud)
            y_pred: Binary predictions
            y_pred_prob: Predicted probabilities (optional, for future use)
            groups: Series with age group assignments
            X_features: Feature dataframe containing credit limit
            credit_col: Name of credit limit column
            ops_cost: Operational cost per false positive
            margin_rate: Margin rate for FP cost calculation
            
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if groups is None:
            # If no groups provided, compute overall metrics
            groups = pd.Series(["Overall"] * len(y_true))
        
        results = []
        
        for group in groups.unique():
            mask = groups == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            if len(y_true_group) == 0:
                continue
            
            # Confusion matrix
            try:
                tn, fp, fn, tp = confusion_matrix(
                    y_true_group, 
                    y_pred_group, 
                    labels=[0, 1]
                ).ravel()
            except ValueError:
                # Handle edge case where only one class is present
                unique_pred = np.unique(y_pred_group)
                unique_true = np.unique(y_true_group)
                if len(unique_pred) == 1 and len(unique_true) == 1:
                    if unique_pred[0] == 0 and unique_true[0] == 0:
                        tn, fp, fn, tp = len(y_true_group), 0, 0, 0
                    elif unique_pred[0] == 1 and unique_true[0] == 1:
                        tn, fp, fn, tp = 0, 0, 0, len(y_true_group)
                    else:
                        tn, fp, fn, tp = 0, 0, 0, 0
                else:
                    raise
            
            # Flag rate (selection rate)
            flag_rate = (y_pred_group == 1).mean()
            
            # TPR and FPR
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            # Loss metrics (if X_features provided)
            fn_loss = 0.0
            fp_cost = 0.0
            total_loss = 0.0
            
            if X_features is not None:
                if hasattr(X_features, 'loc'):
                    X_group = X_features.loc[mask]
                else:
                    X_group = X_features[mask]
                
                if credit_col in X_group.columns:
                    credit = X_group[credit_col].to_numpy(dtype=float)
                    
                    # FN loss: fraud that was accepted
                    fn_mask = (y_true_group == 1) & (y_pred_group == 0)
                    fn_loss = float(np.sum(credit[fn_mask]))
                    
                    # FP cost: legitimate flagged
                    fp_mask = (y_true_group == 0) & (y_pred_group == 1)
                    if fp_mask.sum() > 0:
                        fp_per_case = ops_cost + margin_rate * credit[fp_mask]
                        fp_cost = float(np.sum(fp_per_case))
                    
                    total_loss = fn_loss + fp_cost
            
            results.append({
                "group": group,
                "n": len(y_true_group),
                "n_fraud": int(y_true_group.sum()),
                "fraud_rate": float(y_true_group.mean()),
                "flag_rate": flag_rate,
                "tpr": tpr,
                "fpr": fpr,
                "fn_loss": fn_loss,
                "fp_cost": fp_cost,
                "total_loss": total_loss,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            })
        
        return pd.DataFrame(results)

    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic_func: Callable = np.mean,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        random_state: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for a statistic.
        
        Args:
            data: Array of values to bootstrap
            statistic_func: Function to compute statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (default: 0.95 for 95% CI)
            random_state: Random seed

        """
        if random_state is not None:
            np.random.seed(random_state)
        elif self.random_state is not None:
            np.random.seed(self.random_state)
        
        n = len(data)
        if n == 0:
            return (np.nan, np.nan)
        
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            bootstrap_sample = data[indices]
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return (lower, upper)
    
    def evaluate_cv_fold(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_prob: Optional[np.ndarray] = None,
        X_test: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Evaluate a single CV fold and return group-level metrics.
        
        Args:
            y_true: True labels for test set
            y_pred: Binary predictions
            y_pred_prob: Predicted probabilities (optional)
            X_test: Test feature dataframe
        """
        if X_test is None:
            raise ValueError("X_test must be provided")

        groups = self._get_group_labels(X_test)

        metrics = self.compute_group_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_prob=y_pred_prob,
            groups=groups,
            X_features=X_test
        )
        
        return metrics
    
    def aggregate_cv_results(
        self,
        cv_results: List[pd.DataFrame],
        compute_cis: bool = True,
        n_bootstrap: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """
        Aggregate results across CV folds.
        
        Args:
            cv_results: List of DataFrames, one per CV fold
            compute_cis: Whether to compute bootstrap CIs
            n_bootstrap: Number of bootstrap samples for CI
            

        """
        if len(cv_results) == 0:
            return {
                "mean_metrics": pd.DataFrame(),
                "std_metrics": pd.DataFrame(),
                "ci_metrics": pd.DataFrame()
            }
        
        # Concatenate all fold results
        all_results = pd.concat(cv_results, ignore_index=True)
        
        # Group by group and compute mean/std
        numeric_cols = all_results.select_dtypes(include=[np.number]).columns.tolist()
        if "group" in numeric_cols:
            numeric_cols.remove("group")
        
        mean_metrics = all_results.groupby("group")[numeric_cols].mean()
        std_metrics = all_results.groupby("group")[numeric_cols].std()
        
        result_dict = {
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
        }
        
        if compute_cis:
            # Compute bootstrap CIs for each metric and group
            ci_data = {}
            for group in all_results["group"].unique():
                group_data = all_results[all_results["group"] == group]
                ci_data[group] = {}
                
                for metric in ["flag_rate", "tpr", "fpr", "total_loss", "fn_loss", "fp_cost"]:
                    if metric in group_data.columns:
                        values = group_data[metric].values
                        if len(values) > 0 and not np.all(np.isnan(values)):
                            lower, upper = self.bootstrap_ci(
                                values[~np.isnan(values)],
                                statistic_func=np.mean,
                                n_bootstrap=n_bootstrap,
                                random_state=self.random_state
                            )
                            ci_data[group][f"{metric}_ci_lower"] = lower
                            ci_data[group][f"{metric}_ci_upper"] = upper
            
            if ci_data:
                result_dict["ci_metrics"] = pd.DataFrame(ci_data).T
            else:
                result_dict["ci_metrics"] = pd.DataFrame()
        
        return result_dict
    
    def paired_test_across_folds(
        self,
        cv_results_group1: List[pd.DataFrame],
        cv_results_group2: List[pd.DataFrame],
        metric: str = "flag_rate",
        test_type: str = "t-test"
    ) -> pd.DataFrame:
        """
        Perform paired statistical test across CV folds for a given metric.
        
        Compares metrics between two groups (e.g., two age groups) across folds.
        """
        if len(cv_results_group1) != len(cv_results_group2):
            raise ValueError("Both groups must have same number of CV folds")
        
        # Extract metric values for each fold
        values1 = []
        values2 = []
        
        for df1, df2 in zip(cv_results_group1, cv_results_group2):
            if metric in df1.columns and metric in df2.columns:
                # Take first row (assuming single group per fold)
                val1 = df1[metric].iloc[0] if len(df1) > 0 else np.nan
                val2 = df2[metric].iloc[0] if len(df2) > 0 else np.nan
                values1.append(val1)
                values2.append(val2)
        
        values1 = np.array(values1)
        values2 = np.array(values2)
        
        # Remove NaN pairs
        valid_mask = ~(np.isnan(values1) | np.isnan(values2))
        values1 = values1[valid_mask]
        values2 = values2[valid_mask]
        
        if len(values1) < 2:
            return pd.DataFrame({
                "metric": [metric],
                "test_type": [test_type],
                "statistic": [np.nan],
                "p_value": [np.nan],
                "significant": [False]
            })
        
        # Perform test
        if test_type == "t-test":
            statistic, p_value = stats.ttest_rel(values1, values2)
        elif test_type == "wilcoxon":
            statistic, p_value = stats.wilcoxon(values1, values2)
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
        
        return pd.DataFrame({
            "metric": [metric],
            "test_type": [test_type],
            "statistic": [statistic],
            "p_value": [p_value],
            "significant": [p_value < 0.05]
        })
