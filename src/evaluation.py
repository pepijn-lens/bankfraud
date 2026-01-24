"""
Evaluation methods for fraud detection models.

Contains:
- MLMetricsEvaluator: Standard ML evaluation metrics (ROC, classification reports, confusion matrices)
- ValueAwareEvaluator: Cost-sensitive evaluation with financial loss metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
    auc,
    roc_curve,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from matplotlib import pyplot as plt


class MLMetricsEvaluator:
    """
    Standard machine learning evaluation metrics and visualizations.
    """

    def print_cls_report(self, y_test, y_pred, title):
        """Calculate and print classification report."""
        default_report = classification_report(y_test, y_pred, target_names=['No Fraud', 'Fraud'])
        
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred)

        print(title)
        print('*****' * 10)
        print(default_report)

        return recall

    def plot_con_matrix(self, ax, y_test, y_pred, title, total_loss=None):
        """Plot confusion matrix on given axes."""
        classes = ['No Fraud', 'Fraud']

        con_matrix = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = con_matrix.ravel()
        fpr = fp / (fp + tn)

        ax.imshow(con_matrix, interpolation='nearest', cmap=plt.cm.Blues)

        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

        fmt = 'd'
        threshold = con_matrix.max() / 2.
        for i, j in np.ndindex(con_matrix.shape):
            ax.text(j, i, format(con_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if con_matrix[i, j] > threshold else "black")

        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        if total_loss is not None:
            ax.set_title(f'{title}\nTotal Bank Loss: ${total_loss:,.2f}')
        else:
            ax.set_title(f'{title} with {fpr*100:.2f}% FPR')

    def print_cv_results(self, model):
        """Print cross-validation results from a search object."""
        means = model.cv_results_['mean_test_score']
        params = model.cv_results_['params']

        combined_results = zip(means, params)
        sorted_results = sorted(combined_results, key=lambda x: x[0], reverse=True)

        for mean, param in sorted_results:
            print("mean_test_score: %f, params: %r" % (mean, param))

    def plot_roc_curves(self, fpr_list, tpr_list, label_list, va_points=None, static_05_points=None, 
                        va_thresholds=None):
        """
        Plot ROC curves for multiple models with value-aware threshold points and static 0.5 threshold marked.
        
        Args:
            fpr_list: List of FPR arrays for each model
            tpr_list: List of TPR arrays for each model
            label_list: List of model names
            va_points: Optional list of (fpr, tpr) tuples for value-aware threshold points to mark
            static_05_points: Optional list of (fpr, tpr) tuples for static 0.5 threshold points to mark
            va_thresholds: Optional list of threshold values for value-aware points
        """
        plt.figure(figsize=(8, 8))
        
        # Define distinct colors for each model
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i in range(len(fpr_list)):
            color = colors[i % len(colors)]
            roc_auc_score = auc(fpr_list[i], tpr_list[i])
            plt.plot(fpr_list[i], tpr_list[i], color=color, label=f'{label_list[i]} (AUC={roc_auc_score:.2f})')
            
            # Mark static 0.5 threshold point if provided (use dot/circle)
            if static_05_points and i < len(static_05_points) and static_05_points[i] is not None:
                static_fpr, static_tpr = static_05_points[i]
                plt.plot(static_fpr, static_tpr, color=color, marker='o', markersize=10, 
                        markeredgewidth=2, markeredgecolor='black', linestyle='None',
                        label=f'{label_list[i]} - Static 0.5')
                # Annotate with threshold value
                plt.annotate(f'0.50', xy=(static_fpr, static_tpr), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9, color=color,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color))
            
            # Mark value-aware threshold point if provided (use triangle)
            if va_points and i < len(va_points) and va_points[i] is not None:
                va_fpr, va_tpr = va_points[i]
                plt.plot(va_fpr, va_tpr, color=color, marker='^', markersize=10, 
                        markeredgewidth=2, markeredgecolor='black', linestyle='None',
                        label=f'{label_list[i]} - Optimal Threshold')
                # Annotate with threshold value
                if va_thresholds and i < len(va_thresholds) and va_thresholds[i] is not None:
                    threshold_val = va_thresholds[i]
                    plt.annotate(f'{threshold_val:.4f}', xy=(va_fpr, va_tpr), xytext=(5, -15), 
                               textcoords='offset points', fontsize=9, color=color,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color))
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def test_classifier(self, classifier, X_test, y_test, value_evaluator=None):
        """
        Evaluates a binary classifier by generating ROC curves, classification reports, and confusion matrices.
        Uses value-aware threshold (minimizing total bank loss) if value_evaluator and proposed_credit_limit are available.
        
        Parameters:
        -----------
        classifier : classifier object
            Binary classifier object to be evaluated.
        X_test : numpy.ndarray or pandas.DataFrame
            Test feature data. Must contain 'proposed_credit_limit' if using value-aware threshold.
        y_test : numpy.ndarray or pandas.Series
            Test target labels.
        value_evaluator : ValueAwareEvaluator, optional
            If provided and X_test has 'proposed_credit_limit', finds threshold that minimizes total bank loss.
            
        Returns:
        --------
        tuple : A tuple containing (fpr, tpr, default_recall, target_recall, va_threshold, va_fpr_tpr)
                va_threshold: The optimal threshold found (value-aware or FPR-based)
                va_fpr_tpr: (fpr, tpr) point for the optimal threshold, or None
        """
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        # Determine optimal threshold: value-aware if possible, otherwise FPR-based
        va_threshold = None
        va_fpr_tpr = None
        
        if value_evaluator is not None and "proposed_credit_limit" in X_test.columns:
            # Find threshold that minimizes total bank loss
            # Try a range of thresholds (from ROC curve thresholds, sampled for efficiency)
            # Use every Nth threshold to speed up search
            step = max(1, len(thresholds) // 100)  # Sample up to 100 thresholds
            sampled_indices = range(0, len(thresholds), step)
            
            best_loss = float('inf')
            best_threshold = 0.5
            best_idx = len(thresholds) // 2
            
            for idx in sampled_indices:
                thresh = thresholds[idx]
                results = value_evaluator.evaluate(
                    y_true=y_test,
                    y_pred_prob=y_prob,
                    X_features=X_test,
                    threshold_method="static",
                    static_threshold=thresh,
                )
                total_loss = results['Total_Bank_Loss_($)']
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_threshold = thresh
                    best_idx = idx
            
            va_threshold = best_threshold
            va_fpr_tpr = (fpr[best_idx], tpr[best_idx])
            threshold_label = f'Value-Aware Threshold @ {va_threshold:.4f} (Loss=${best_loss:,.2f})'
        else:
            # Fallback to FPR-based threshold
            target_fpr = 0.05
            best_idx = np.argmin(np.abs(fpr - target_fpr))
            va_threshold = thresholds[best_idx]
            va_fpr_tpr = (fpr[best_idx], tpr[best_idx])
            threshold_label = f'FPR-based Threshold @ {va_threshold:.4f}'
        
        y_pred_threshold = (y_prob >= va_threshold).astype(int)
        
        default_recall = self.print_cls_report(y_test, y_pred, title="Default Threshold @ 0.50")
        target_recall = self.print_cls_report(y_test, y_pred_threshold, title=threshold_label)

        # Compute total bank loss for both thresholds if value_evaluator is available
        default_total_loss = None
        optimal_total_loss = None
        if value_evaluator is not None and "proposed_credit_limit" in X_test.columns:
            # Default threshold (0.5)
            default_results = value_evaluator.evaluate(
                y_true=y_test,
                y_pred_prob=y_prob,
                X_features=X_test,
                threshold_method="static",
                static_threshold=0.5,
            )
            default_total_loss = default_results['Total_Bank_Loss_($)']
            
            # Optimal threshold
            optimal_results = value_evaluator.evaluate(
                y_true=y_test,
                y_pred_prob=y_prob,
                X_features=X_test,
                threshold_method="static",
                static_threshold=va_threshold,
            )
            optimal_total_loss = optimal_results['Total_Bank_Loss_($)']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        self.plot_con_matrix(ax1, y_test, y_pred, title='Default Threshold @ 0.50', total_loss=default_total_loss)
        self.plot_con_matrix(ax2, y_test, y_pred_threshold, title=threshold_label, total_loss=optimal_total_loss)

        plt.tight_layout()
        plt.show()

        # Find static 0.5 threshold point on ROC curve
        static_05_idx = np.argmin(np.abs(thresholds - 0.5))
        static_05_fpr_tpr = (fpr[static_05_idx], tpr[static_05_idx])
        
        return fpr, tpr, thresholds, default_recall, target_recall, va_threshold, va_fpr_tpr, static_05_fpr_tpr


class ValueAwareEvaluator:
    """
    Evaluator for a cost-sensitive fraud detection.

    The evaluator learns how credit limits are distributed among legitimate applicants in the training data.
    This information is later used to map normalized income scores to proxies for customer value.
    """

    def predict_static(self, y_pred_prob, static_threshold: float = 0.5) -> np.ndarray:
        """RQ1 Binary decisions from probabilities using a fixed threshold."""
        p = np.asarray(y_pred_prob, dtype=float)
        return (p >= float(static_threshold)).astype(int)

    def predict_value_aware(
            self,
            y_pred_prob,
            X,
            ops_cost=100.0,
            impact_col="proposed_credit_limit",
            margin_rate=0.05,
    ):
        """
        RQ2: Value-aware decision rule

        Flag if:
            p_i >= C_FP,i / (C_FP,i + C_FN,i)
        """
        p = np.asarray(y_pred_prob, dtype=float)
        credit = X[impact_col].to_numpy(dtype=float)

        C_fn = credit
        C_fp = ops_cost + margin_rate * credit

        threshold = C_fp / (C_fp + C_fn)

        return (p >= threshold).astype(int)

    def fp_loss_proxy(
            self,
            X,
            credit_col="proposed_credit_limit",
            ops_cost=100.0,
            margin_rate=0.05
        ):
        """
        Returns False Positive loss = operational costs + margin * credit
        """
        credit = X[credit_col].to_numpy(dtype=float)
        return ops_cost + margin_rate * credit

    def compute_losses(self, y_true, y_pred, X_features, credit_col="proposed_credit_limit", income_col="income") -> dict:
        """
        Compute financial losses and error counts given model decisions.

        The method evaluates the outcomes of binary decisions by aggregating:
        - fraud loss from fraudulent applications that were accepted, and
        - false alarm cost from legitimate applications that were incorrectly flagged.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        credit = X_features[credit_col].to_numpy(dtype=float)
        fp_per_case = self.fp_loss_proxy(
            X_features,
            credit_col=credit_col
        )

        tp = (y_true == 1) & (y_pred == 1)
        fn = (y_true == 1) & (y_pred == 0)
        fp = (y_true == 0) & (y_pred == 1)

        fraud_caught = float(np.sum(credit[tp]))
        fraud_loss = float(np.sum(credit[fn]))
        false_alarm_cost = float(np.sum(fp_per_case[fp]))
        total_loss = fraud_loss + false_alarm_cost

        return {
            "fraud_caught": fraud_caught,
            "fraud_loss": fraud_loss,
            "false_alarm_cost": false_alarm_cost,
            "total_loss": total_loss,
            "tp": int(tp.sum()),
            "fn": int(fn.sum()),
            "fp": int(fp.sum()),
        }

    def evaluate(self, y_true, y_pred_prob, X_features, threshold_method='static', static_threshold=0.5):
        """
        Evaluate a fraud detection model using a cost-based loss formulation.

        This method applies a chosen decision rule (static or dynamic) to convert
        predicted fraud probabilities into binary decisions, and then computes the
        resulting financial losses and standard classification metrics.

        Financial losses are defined as:
        - Fraud loss: total exposure from fraud applications that were accepted.
        - False alarm cost: total cost incurred by incorrectly flagging legitimate applications

        Args:
            y_true: true labels (0=legit, 1=fraud)
            y_pred_prob: predicted probabilities of fraud
            X_features: feature data containing `proposed_credit_limit` and `income`
            threshold_method: decision rule used to produce binary predictions
                - "static": flag if probability >= static_threshold
                - "dynamic": flag using a cost-based threshold derived from exposure
            static_threshold: probability threshold used when threshold_method="static"
        """
        if threshold_method=="static":
            y_pred = self.predict_static(y_pred_prob, static_threshold=static_threshold)
        else:
            y_pred = self.predict_value_aware(
                y_pred_prob, X_features
            )

        losses = self.compute_losses(
            y_true, y_pred, X_features
        )

        return {
            "threshold_type": f'Static ({static_threshold:.2f})'  if threshold_method == 'static' else f'Dynamic',
            "Total_Bank_Loss_($)": float(losses["total_loss"]),
            "Fraud_Loss_($)": float(losses["fraud_loss"]),
            "False_Alarm_Cost_($)": float(losses["false_alarm_cost"]),
            "Fraud_Caught_($)": float(losses["fraud_caught"]),
            "recall": recall_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "tp": losses["tp"],
            "fn": losses["fn"],
            "fp": losses["fp"],
        }

    def sweep_thresholds(
        self,
        y_true,
        y_pred_prob,
        X_features,
        threshold_range=None,
        n_thresholds=200,
    ):
        """
        Sweep through multiple thresholds to find the one that minimizes total bank loss.
        
        Args:
            y_true: True labels (0=legit, 1=fraud)
            y_pred_prob: Predicted probabilities of fraud
            X_features: Feature data containing proposed_credit_limit
            threshold_range: Tuple (min, max) for threshold range. If None, uses (0, 1).
            n_thresholds: Number of thresholds to test (default 200)
        
        Returns:
            DataFrame with columns: threshold, Total_Bank_Loss_($), Fraud_Loss_($),
            False_Alarm_Cost_($), recall, accuracy, f1
        """
        if threshold_range is None:
            threshold_range = (0.0, 1.0)
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        results = []
        
        for thresh in thresholds:
            res = self.evaluate(
                y_true=y_true,
                y_pred_prob=y_pred_prob,
                X_features=X_features,
                threshold_method="static",
                static_threshold=thresh,
            )
            results.append({
                "threshold": thresh,
                "Total_Bank_Loss_($)": res["Total_Bank_Loss_($)"],
                "Fraud_Loss_($)": res["Fraud_Loss_($)"],
                "False_Alarm_Cost_($)": res["False_Alarm_Cost_($)"],
                "Fraud_Caught_($)": res["Fraud_Caught_($)"],
                "recall": res["recall"],
                "accuracy": res["accuracy"],
                "f1": res["f1"],
                "tp": res["tp"],
                "fn": res["fn"],
                "fp": res["fp"],
            })
        
        df = pd.DataFrame(results)
        return df

    def plot_threshold_sweep(
        self,
        y_true,
        y_pred_prob,
        X_features,
        threshold_range=None,
        n_thresholds=200,
        save_path=None,
    ):
        """
        Plot total bank loss vs threshold to visualize the optimal threshold.
        
        Args:
            y_true: True labels (0=legit, 1=fraud)
            y_pred_prob: Predicted probabilities of fraud
            X_features: Feature data containing proposed_credit_limit
            threshold_range: Tuple (min, max) for threshold range. If None, uses (0, 1).
            n_thresholds: Number of thresholds to test (default 200)
            save_path: Optional path to save the plot
        """
        df = self.sweep_thresholds(
            y_true, y_pred_prob, X_features, threshold_range, n_thresholds
        )
        
        # Find optimal threshold
        optimal_idx = df["Total_Bank_Loss_($)"].idxmin()
        optimal_thresh = df.loc[optimal_idx, "threshold"]
        optimal_loss = df.loc[optimal_idx, "Total_Bank_Loss_($)"]
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot 1: Total Bank Loss
        axes[0].plot(df["threshold"], df["Total_Bank_Loss_($)"], linewidth=2, label="Total Bank Loss")
        axes[0].axvline(optimal_thresh, color='r', linestyle='--', linewidth=2, 
                       label=f'Optimal @ {optimal_thresh:.4f} (${optimal_loss:,.2f})')
        axes[0].axvline(0.5, color='g', linestyle='--', linewidth=1, alpha=0.7, label='Default @ 0.5')
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('Total Bank Loss ($)')
        axes[0].set_title('Total Bank Loss vs Threshold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Components (Fraud Loss + False Alarm Cost)
        axes[1].plot(df["threshold"], df["Fraud_Loss_($)"], linewidth=2, label="Fraud Loss", color='orange')
        axes[1].plot(df["threshold"], df["False_Alarm_Cost_($)"], linewidth=2, label="False Alarm Cost", color='purple')
        axes[1].axvline(optimal_thresh, color='r', linestyle='--', linewidth=2, 
                       label=f'Optimal @ {optimal_thresh:.4f}')
        axes[1].axvline(0.5, color='g', linestyle='--', linewidth=1, alpha=0.7, label='Default @ 0.5')
        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel('Loss ($)')
        axes[1].set_title('Loss Components vs Threshold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved threshold sweep plot to {save_path}")
        
        plt.show()
        
        print(f"\nOptimal Threshold: {optimal_thresh:.4f}")
        print(f"Optimal Total Bank Loss: ${optimal_loss:,.2f}")
        print(f"\nAt optimal threshold:")
        print(f"  Fraud Loss: ${df.loc[optimal_idx, 'Fraud_Loss_($)']:,.2f}")
        print(f"  False Alarm Cost: ${df.loc[optimal_idx, 'False_Alarm_Cost_($)']:,.2f}")
        print(f"  Recall: {df.loc[optimal_idx, 'recall']:.4f}")
        print(f"  F1: {df.loc[optimal_idx, 'f1']:.4f}")
        
        return df, optimal_thresh
