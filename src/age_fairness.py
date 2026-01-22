import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.metrics import confusion_matrix
from scipy import stats
import warnings


class AgeFairnessAnalyzer:
    """
    Analyzer for evaluating fairness across age groups (RQ5).
    
    Evaluates:
    - Flag rate (selection rate) per age group
    - TPR/FPR per age group (equal opportunity/equalized odds)
    - Loss per age group (FN loss and FP cost)
    - Statistical significance via bootstrap CIs or paired tests
    """
    
    def __init__(
        self,
        age_col: str = "customer_age",
        n_bins: int = 4,
        binning_method: str = "quantile",
        predefined_bins: Optional[List[float]] = None,
        random_state: int = 42
    ):
        """
        Args:
            age_col: Name of the age column
            n_bins: Number of age bins (if using quantile or uniform binning)
            binning_method: "quantile", "uniform", or "predefined"
            predefined_bins: List of bin edges for predefined binning (e.g., [18, 30, 45, 60, 100])
            random_state: Random seed for bootstrap
        """
        self.age_col = age_col
        self.n_bins = n_bins
        self.binning_method = binning_method
        self.predefined_bins = predefined_bins
        self.random_state = random_state
        self.bin_edges_ = None
        self.bin_labels_ = None
        
    def create_age_bins(
        self,
        ages: pd.Series,
        return_labels: bool = True
    ) -> Tuple[pd.Series, Optional[List[str]]]:
        """
        Discretize age into K bins to ensure stable estimation of group-conditional rates.
        
        Args:
            ages: Series of age values
            return_labels: Whether to return readable bin labels
            
        """
        ages = pd.Series(ages)
        
        if self.binning_method == "predefined":
            if self.predefined_bins is None:
                raise ValueError("predefined_bins must be provided for predefined binning")
            self.bin_edges_ = self.predefined_bins
            age_bins = pd.cut(
                ages, 
                bins=self.bin_edges_, 
                include_lowest=True, 
                duplicates='drop'
            )
        elif self.binning_method == "quantile":
            age_bins, self.bin_edges_ = pd.qcut(
                ages, 
                q=self.n_bins, 
                retbins=True, 
                duplicates='drop'
            )
        elif self.binning_method == "uniform":
            age_bins, self.bin_edges_ = pd.cut(
                ages, 
                bins=self.n_bins, 
                retbins=True, 
                include_lowest=True, 
                duplicates='drop'
            )
        else:
            raise ValueError(f"Unknown binning_method: {self.binning_method}. Must be 'quantile', 'uniform', or 'predefined'")
        
        if return_labels:
            # Create readable labels
            self.bin_labels_ = []
            for i in range(len(self.bin_edges_) - 1):
                low = int(self.bin_edges_[i])
                high = int(self.bin_edges_[i + 1])
                if i == len(self.bin_edges_) - 2:
                    # Last bin: include upper bound
                    self.bin_labels_.append(f"{low}-{high}")
                else:
                    self.bin_labels_.append(f"{low}-{high-1}")
            
            # Map intervals to labels
            label_map = {
                interval: label 
                for interval, label in zip(age_bins.cat.categories, self.bin_labels_)
            }
            age_bins = age_bins.map(label_map)
        
        return age_bins, self.bin_labels_
    
    def compute_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_prob: Optional[np.ndarray] = None,
        age_groups: Optional[pd.Series] = None,
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
            age_groups: Series with age group assignments
            X_features: Feature dataframe containing credit limit
            credit_col: Name of credit limit column
            ops_cost: Operational cost per false positive
            margin_rate: Margin rate for FP cost calculation
            
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if age_groups is None:
            # If no groups provided, compute overall metrics
            age_groups = pd.Series(["Overall"] * len(y_true))
        
        results = []
        
        for group in age_groups.unique():
            mask = age_groups == group
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
                if hasattr(X_features, 'iloc'):
                    X_group = X_features.iloc[mask]
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
                "age_group": group,
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
    
    def compute_disparity_metrics(
        self,
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
            reference_group = group_metrics.loc[group_metrics["n"].idxmax(), "age_group"]
        
        ref_metrics = group_metrics[group_metrics["age_group"] == reference_group]
        if len(ref_metrics) == 0:
            warnings.warn(f"Reference group '{reference_group}' not found. Using first group.")
            reference_group = group_metrics.iloc[0]["age_group"]
            ref_metrics = group_metrics[group_metrics["age_group"] == reference_group]
        
        ref_metrics = ref_metrics.iloc[0]
        
        disparities = []
        for _, row in group_metrics.iterrows():
            if row["age_group"] == reference_group:
                continue
            
            disparities.append({
                "age_group": row["age_group"],
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
        age_col: str = "customer_age"
    ) -> pd.DataFrame:
        """
        Evaluate a single CV fold and return group-level metrics.
        
        Args:
            y_true: True labels for test set
            y_pred: Binary predictions
            y_pred_prob: Predicted probabilities (optional)
            X_test: Test feature dataframe
            age_col: Name of age column
            
        """
        if X_test is None:
            raise ValueError("X_test must be provided")
        
        if age_col not in X_test.columns:
            raise ValueError(f"Age column '{age_col}' not found in X_test. Available columns: {list(X_test.columns)}")
        
        ages = X_test[age_col]
        age_groups, _ = self.create_age_bins(ages, return_labels=True)
        
        metrics = self.compute_group_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_prob=y_pred_prob,
            age_groups=age_groups,
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
        
        # Group by age_group and compute mean/std
        numeric_cols = all_results.select_dtypes(include=[np.number]).columns.tolist()
        if "age_group" in numeric_cols:
            numeric_cols.remove("age_group")
        
        mean_metrics = all_results.groupby("age_group")[numeric_cols].mean()
        std_metrics = all_results.groupby("age_group")[numeric_cols].std()
        
        result_dict = {
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
        }
        
        if compute_cis:
            # Compute bootstrap CIs for each metric and group
            ci_data = {}
            for group in all_results["age_group"].unique():
                group_data = all_results[all_results["age_group"] == group]
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
