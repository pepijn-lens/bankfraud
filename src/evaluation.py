import pandas as pd
import numpy as np

def calculate_expected_loss(y_true, y_prob, credit_limits, incomes, alpha=0.05):
    """
    Calculates Expected Loss using SELF-REFERENTIAL VALUE MAPPING.
    
    METHODOLOGY:
    1. C_FN (Risk) = Raw Credit Limit (Euros).
    2. C_FP (Value) = Alpha * Quantile(Limit_Distribution, Income_Score)
    
    Args:
        y_true: True labels (0 or 1)
        y_prob: Predicted probability of fraud
        credit_limits: RAW credit limit values (Euros)
        incomes: Income Quantiles (0 to 1)
        alpha: Profit margin (default 0.05)
    """
    total_loss = 0
    decisions = []
    
    # 1. Setup the "Dictionary"
    limit_dist = np.array(credit_limits) 
    
    # 2. Get Inputs
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    incomes = np.array(incomes) 
    
    # 3. Calculate Value (C_FP) using the Translation
    # Map income quantile -> Credit Limit Value
    mapped_values = np.quantile(limit_dist, incomes)
    
    # Apply the profit margin (alpha)
    # NO fixed_cost added here. Purely proportional.
    c_fp_values = alpha * mapped_values
    
    # 4. Calculate Risk (C_FN)
    c_fn_values = limit_dist
    
    for i in range(len(y_true)):
        c_fn = c_fn_values[i]  # Risk (Real Currency)
        c_fp = c_fp_values[i]  # Value (Mapped Currency)
        
        # Dynamic Threshold Calculation
        threshold = c_fp / (c_fn + c_fp)
        
        # Decision Rule
        pred_fraud = 1 if y_prob[i] > threshold else 0
        decisions.append(pred_fraud)
        
        # Calculate Actual Financial Loss
        if pred_fraud == 1 and y_true[i] == 0:
            total_loss += c_fp  # We rejected a valid customer
        elif pred_fraud == 0 and y_true[i] == 1:
            total_loss += c_fn  # We accepted a fraud
            
    return total_loss, decisions