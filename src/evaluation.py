import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class ValueAwareEvaluator:
    def __init__(self):
        self.limit_quantiles = None

    def fit(self, X_train, y_train):
        """
        Learns the 'Ladder of Value' from legitimate customers in the training set.
        """
        # Ensure we are working with standard Series/Arrays
        if isinstance(y_train, np.ndarray):
            y_train = pd.Series(y_train)
            
        # 1. Filter for legitimate customers only (Label = 0)
        legit_mask = (y_train == 0)
        
        # We need the original indices to align with X_train if it's a dataframe
        if hasattr(legit_mask, 'values'):
            legit_mask = legit_mask.values
            
        legit_limits = X_train.iloc[legit_mask]['proposed_credit_limit']

        # 2. Build the Ladder of Value (Quantiles of Limits)
        # This defines what a "Normal" limit looks like for every percentile
        self.limit_quantiles = legit_limits.quantile(np.linspace(0, 1, 101))
        print("Evaluator Fitted: 'Ladder of Value' created from legitimate training data.")

    def get_proxy_value(self, income_scores):
        """
        Maps Income Scores to 'Proxy Limits' (Euros) using the fitted Ladder.
        """
        # Clip scores to be safe
        clipped_scores = np.clip(income_scores, 0, 1)
        # Convert score (0.9) to index (90)
        indices = (clipped_scores * 100).astype(int)
        
        # specific fix: ensure indices are within bounds 0-100
        indices = np.clip(indices, 0, 100)
        
        quantile_values = self.limit_quantiles.values
        return quantile_values[indices]

    def get_dynamic_thresholds(self, risk_amount, value_amount):
        """
        Calculates the specific threshold T_i = Value / (Risk + Value)
        """
        # Added small epsilon (1e-9) to avoid division by zero
        thresholds = value_amount / (risk_amount + value_amount + 1e-9)
        return thresholds

    def calculate_savings(self, y_true, y_pred_prob, X_features, threshold_method='static', static_threshold=0.5):
        """
        Computes financial metrics.
        """
        # --- CRITICAL FIX: Convert inputs to Numpy Arrays ---
        # This drops the Pandas Index, preventing mismatch bugs.
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(y_pred_prob, 'values'):
            y_pred_prob = y_pred_prob.values
            
        # 1. Get Financial Values (Risk & Value)
        risk_amount = X_features['proposed_credit_limit'].values
        customer_value = self.get_proxy_value(X_features['income'].values)
        
        # 2. Determine Decisions
        if threshold_method == 'dynamic':
            # Calculate Dynamic Threshold per person
            thresholds = self.get_dynamic_thresholds(risk_amount, customer_value)
            y_pred = (y_pred_prob >= thresholds).astype(int)
        else:
            # Standard Static Threshold
            y_pred = (y_pred_prob >= static_threshold).astype(int)

        # --- CALCULATE METRICS ---
        
        # TP: Fraud Caught (Good) -> Saved the Risk Amount
        tp_mask = (y_true == 1) & (y_pred == 1)
        fraud_caught_val = np.sum(risk_amount[tp_mask])
        
        # FN: Fraud Missed (Bad) -> Lost the Risk Amount
        fn_mask = (y_true == 1) & (y_pred == 0)
        fraud_loss_val = np.sum(risk_amount[fn_mask])
        
        # FP: False Alarm (Bad) -> Lost the Customer Value
        fp_mask = (y_true == 0) & (y_pred == 1)
        false_alarm_cost = np.sum(customer_value[fp_mask])
        
        # Total Badness (Lower is Better)
        total_loss = fraud_loss_val + false_alarm_cost
        
        return {
            'threshold_type': 'Dynamic' if threshold_method == 'dynamic' else f'Static ({static_threshold:.2f})',
            'Total_Bank_Loss_($)': total_loss,
            'Fraud_Loss_($)': fraud_loss_val,
            'False_Alarm_Cost_($)': false_alarm_cost,
            'Fraud_Caught_($)': fraud_caught_val,
            'recall': recall_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }