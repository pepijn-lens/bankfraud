import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

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
