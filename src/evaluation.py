import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class ValueAwareEvaluator:
    """
    Evaluator for a cost-sensitive fraud detection.

    The evaluator learns how credit limits are distributed among legitimate applicants in the training data.
    This information is later used to map normalized income scores to proxies for customer value.
    """
    def __init__(self):
        self.limit_quantiles = None

    def fit(self, X_train, y_train):
        """
        Learn credit-limit quantiles from legitimate training applications.

        The method looks only at training samples labeled as legitimate (y = 0)
        and computes credit-limit values at different percentiles. These values
        are used later to translate income ranks into proxy amounts.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        required_col = "proposed_credit_limit"
        if required_col not in X_train.columns:
            raise ValueError(f"X_train must contain column '{required_col}'.")

        y_train = pd.Series(y_train, index=X_train.index)
            
        legit_mask = (y_train == 0)

        legit_limits = X_train.loc[legit_mask, required_col]

        self.limit_quantiles = legit_limits.quantile(np.linspace(0, 1, 101))
        print("Evaluator Fitted: Computed credit-limit quantiles from legitimate training data.")

    def get_proxy_value(self, income_scores):
        """
        Map normalized income scores to proxy values using credit-limit quantiles.

        Each income score [0, 1] is interpreted as a percentile rank and mapped
        to the corresponding credit-limit percentile learned from legitimate
        training data. The returned values are proxy values on the same scale as
        `proposed_credit_limit`.
        """
        if self.limit_quantiles is None:
            raise RuntimeError(
                "Evaluator must be fitted before calling map_income_to_value()."
            )

        clipped_scores = np.clip(income_scores, 0, 1)
        indices = (clipped_scores * 100).astype(int)
        indices = np.clip(indices, 0, 100)
        quantile_values = self.limit_quantiles.values

        return quantile_values[indices]

    def get_decision_thresholds(self, fraud_loss, false_alarm_cost):
        """
        Compute cost-based decision thresholds for rejecting an application.

        For each application, the threshold represents the minimum predicted
        fraud probability at which rejection is cheaper than acceptance.
        It is defined as: threshold = value / (risk + value)
            where risk is the loss incurred if fraud is accepted and
            value is the cost of rejecting a legitimate application.
        """
        thresholds = false_alarm_cost / (fraud_loss + false_alarm_cost + 1e-9)
        return thresholds

    def predict(self, y_pred_prob, X_features, threshold_method='static', static_threshold=0.5, alpha=1.0) -> np.ndarray:
        """
        Converts predicted fraud probabilities into decisions.

        This method applies either a static probability threshold or a cost-based
        decision rule to determine whether each application should be flagged.
        """
        p = np.asarray(y_pred_prob)

        if threshold_method == "static":
            return (p >= static_threshold).astype(int)

        credit_exposure = X_features["proposed_credit_limit"].to_numpy()
        proxy_value = self.get_proxy_value(X_features["income"].to_numpy())
        false_alarm_cost = alpha * proxy_value

        thresholds = self.get_decision_thresholds(
            fraud_loss=credit_exposure,
            false_alarm_cost=false_alarm_cost,
        )
        return (p >= thresholds).astype(int)

    def compute_costs(self, y_true, y_pred, X_features, alpha=1.0) -> dict:
        """
        Compute financial losses and error counts given model decisions.

        The method evaluates the outcomes of binary decisions by aggregating:
        - fraud loss from fraudulent applications that were accepted, and
        - false alarm cost from legitimate applications that were incorrectly flagged.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        credit_exposure = X_features["proposed_credit_limit"].to_numpy()
        proxy_value = self.get_proxy_value(X_features["income"].to_numpy())
        false_alarm_cost_per_case = alpha * proxy_value

        tp = (y_true == 1) & (y_pred == 1)
        fn = (y_true == 1) & (y_pred == 0)
        fp = (y_true == 0) & (y_pred == 1)

        return {
            "fraud_caught": float(np.sum(credit_exposure[tp])),
            "fraud_loss": float(np.sum(credit_exposure[fn])),
            "false_alarm_cost": float(np.sum(false_alarm_cost_per_case[fp])),
            "tp": int(tp.sum()),
            "fn": int(fn.sum()),
            "fp": int(fp.sum()),
        }

    def calculate_savings(self, y_true, y_pred_prob, X_features, threshold_method='static', static_threshold=0.5, alpha=1.0):
        """
        Evaluate a fraud detection model using a cost-based loss formulation.

        This method applies a chosen decision rule (static or cost-based) to convert
        predicted fraud probabilities into binary decisions, and then computes the
        resulting financial losses and standard classification metrics.

        Financial losses are defined as:
        - Fraud loss: total exposure from fraud applications that were accepted.
        - False alarm cost: total cost incurred by incorrectly flagging legitimate applications,
            scaled by alpha - because usually falsly flaging a legit application doesn't cost us the whole credit limit (it's much smaller).

        Args:
            y_true: true labels (0=legit, 1=fraud)
            y_pred_prob: predicted probabilities of fraud
            X_features: feature data containing `proposed_credit_limit` and `income`
            threshold_method: decision rule used to produce binary predictions
                - "static": flag if probability >= static_threshold
                - "dynamic": flag using a cost-based threshold derived from exposure and proxy values
            static_threshold: probability threshold used when threshold_method="static".
            alpha: scaling factor applied to proxy values used in the false alarm cost.
        """
        y_pred = self.predict(
            y_pred_prob=y_pred_prob,
            X_features=X_features,
            threshold_method=threshold_method,
            static_threshold=static_threshold,
            alpha=alpha
        )

        costs = self.compute_costs(
            y_true=y_true,
            y_pred=y_pred,
            X_features=X_features,
            alpha=alpha
        )

        total_loss = costs["fraud_loss"] + costs["false_alarm_cost"]

        return {
            'threshold_type': 'Dynamic' if threshold_method == 'dynamic' else f'Static ({static_threshold:.2f})',
            'Total_Bank_Loss_($)': total_loss,
            'Fraud_Loss_($)': costs["fraud_loss"],
            'False_Alarm_Cost_($)': costs["false_alarm_cost"],
            'Fraud_Caught_($)': costs["fraud_caught"],
            'recall': recall_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }