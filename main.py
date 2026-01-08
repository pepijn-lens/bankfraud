import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from src.load_data import preprocess_data, split_data, normalize
from src.models import get_base_model
from src.evaluation import ValueAwareEvaluator

def find_best_f1_threshold(y_true, y_prob):
    """Finds the static threshold that maximizes F1 Score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    # Argmax finds the index of the highest F1 score
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

def run_experiment():
    print("1. Loading and Preprocessing Data...")
    # Update path if necessary
    df = pd.read_csv("data/2/Base.csv") 
    
    df = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize(X_train, X_val, X_test)
    
    print(f"Data ready. Train shape: {X_train.shape}")
    
    # --- MODEL TRAINING ---
    print("\n2. Training Logistic Regression...")
    model = get_base_model()
    model.fit(X_train_scaled, y_train)
    
    # Get Probabilities
    y_prob_val = model.predict_proba(X_val_scaled)[:, 1]
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
    
    # --- FIND OPTIMAL STATIC THRESHOLD (Baseline B) ---
    best_static_thresh = find_best_f1_threshold(y_val, y_prob_val)
    print(f"Optimal Static Threshold (Based on Val F1): {best_static_thresh:.4f}")
    
    # --- VALUE-AWARE EVALUATION ---
    print("\n3. Fitting Value-Aware Evaluator...")
    evaluator = ValueAwareEvaluator()
    evaluator.fit(X_train, y_train)
    
    # --- RQ1: Comparison ---
    print("\n=== RQ1: Total Cost Analysis (Lower is Better) ===")
    
    res_naive = evaluator.calculate_savings(y_test, y_prob_test, X_test, threshold_method='static', static_threshold=0.5)
    res_f1 = evaluator.calculate_savings(y_test, y_prob_test, X_test, threshold_method='static', static_threshold=best_static_thresh)
    res_dynamic = evaluator.calculate_savings(y_test, y_prob_test, X_test, threshold_method='dynamic')
    
    # Create comparison table
    results_df = pd.DataFrame([res_naive, res_f1, res_dynamic], 
                              index=['Naive (0.5)', 'Best F1', 'Value-Aware'])
    
    # Select columns
    money_cols = ['Total_Bank_Loss_($)', 'Fraud_Loss_($)', 'False_Alarm_Cost_($)', 'Fraud_Caught_($)']
    
    # Print Money Columns (No Decimals)
    print(results_df[money_cols].round(0))
    
    # Print Metrics Columns (4 Decimals) - Added 'accuracy' here
    print(results_df[['recall', 'accuracy']].round(3))

    # --- RQ2: High Impact Analysis ---
    print("\n=== RQ2: High Impact Fraud Cases Only ===")
    whale_mask = X_test['proposed_credit_limit'] >= X_test['proposed_credit_limit'].quantile(0.90)
    
    # 1. Naive Baseline (0.5) on Whales
    whale_res_naive = evaluator.calculate_savings(y_test[whale_mask], y_prob_test[whale_mask], X_test[whale_mask], 
                                                  threshold_method='static', static_threshold=0.5)
    
    # 2. Best F1 Baseline on Whales
    whale_res_f1 = evaluator.calculate_savings(y_test[whale_mask], y_prob_test[whale_mask], X_test[whale_mask], 
                                               threshold_method='static', static_threshold=best_static_thresh)
    
    # 3. Value-Aware on Whales
    whale_res_dynamic = evaluator.calculate_savings(y_test[whale_mask], y_prob_test[whale_mask], X_test[whale_mask], 
                                                    threshold_method='dynamic')
    
    print(f" Recall (Naive 0.5):   {whale_res_naive['recall']:.2%}")
    print(f" Recall (Best F1):     {whale_res_f1['recall']:.2%}")
    print(f" Recall (Dynamic):     {whale_res_dynamic['recall']:.2%}")

if __name__ == "__main__":
    run_experiment()