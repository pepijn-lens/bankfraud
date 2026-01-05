import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from src import load_data, models, evaluation

def main():
    # ---------------------------------------------------------
    # 1. LOAD & PREPROCESS
    # ---------------------------------------------------------
    print("Loading and Preprocessing...")
    # Ensure this path matches your folder structure
    df = pd.read_csv("data/2/Base.csv") 
    df = load_data.preprocess_data(df)

    # ---------------------------------------------------------
    # 2. SPLIT DATA
    # ---------------------------------------------------------
    print("Splitting data...")
    # X_train, X_test, etc. here are UNSCALED (Original Units)
    X_train, X_val, X_test, y_train, y_val, y_test = load_data.split_data(df)

    # ---------------------------------------------------------
    # 3. NORMALIZE (For Model Training Only)
    # ---------------------------------------------------------
    print("Scaling features...")
    # We create scaled versions specifically for the Logistic Regression model
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = load_data.normalize(
        X_train, X_val, X_test
    )

    # ---------------------------------------------------------
    # 4. TRAIN MODEL
    # ---------------------------------------------------------
    print("Training Base Classifier...")
    model = models.get_base_model()
    model.fit(X_train_scaled, y_train)
    
    # Get probabilities (Class 1 = Fraud)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # ---------------------------------------------------------
    # 5. FINANCIAL PERFORMANCE COMPARISON
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("FINANCIAL PERFORMANCE REPORT")
    print("="*80)

    # --- [FIX] Convert y_test to Numpy Array ---
    # This prevents the KeyError by removing the shuffled pandas index
    y_test_np = y_test.to_numpy()

    # --- A. SETUP COSTS (Self-Referential Logic) ---
    # Risk (C_FN) = The Raw Credit Limit
    risks = X_test['proposed_credit_limit'].values
    
    c_alpha=2.0
    # Value (C_FP) = Alpha * (Credit Limit at Customer's Income Quantile)
    incomes = X_test['income'].values
    limit_dist = df['proposed_credit_limit'].values 
    
    mapped_values = np.quantile(limit_dist, incomes)
    values = c_alpha * mapped_values 

    # --- B. STANDARD MODEL (Static Threshold 0.5) ---
    std_preds = [1 if p > 0.5 else 0 for p in y_prob]
    
    std_fraud_loss = 0
    std_cust_loss = 0
    
    for i in range(len(y_test_np)):
        # We use y_test_np[i] instead of y_test[i]
        if std_preds[i] == 0 and y_test_np[i] == 1: # False Negative
            std_fraud_loss += risks[i]
        elif std_preds[i] == 1 and y_test_np[i] == 0: # False Positive
            std_cust_loss += values[i]
            
    std_total = std_fraud_loss + std_cust_loss

    # --- C. VALUE-AWARE MODEL (Dynamic Threshold) ---
    _, dyn_decisions = evaluation.calculate_expected_loss(
        y_test_np, y_prob,  # Pass numpy version here too
        X_test['proposed_credit_limit'], 
        X_test['income'],
        alpha=c_alpha
    )
    
    dyn_fraud_loss = 0
    dyn_cust_loss = 0
    
    for i in range(len(y_test_np)):
        if dyn_decisions[i] == 0 and y_test_np[i] == 1: # FN
            dyn_fraud_loss += risks[i]
        elif dyn_decisions[i] == 1 and y_test_np[i] == 0: # FP
            dyn_cust_loss += values[i]
            
    dyn_total = dyn_fraud_loss + dyn_cust_loss

    # --- D. PRINT RESULTS TABLE ---
    print(f"{'Metric':<25} | {'Standard Model':<20} | {'Value-Aware Model':<20} | {'Improvement':<15}")
    print("-" * 90)
    print(f"{'Total Financial Loss':<25} | €{std_total:,.2f}          | €{dyn_total:,.2f}          | €{std_total - dyn_total:,.2f}")
    print(f"{'  - Lost to Fraud':<25} | €{std_fraud_loss:,.2f}          | €{dyn_fraud_loss:,.2f}          |")
    print(f"{'  - Lost Opportunity':<25} | €{std_cust_loss:,.2f}          | €{dyn_cust_loss:,.2f}          |")
    print("-" * 90)
    
    # Calculate Value-Weighted Recall
    total_fraud_val = sum(risks[i] for i in range(len(y_test_np)) if y_test_np[i] == 1)
    std_recall_val = 1 - (std_fraud_loss / total_fraud_val)
    dyn_recall_val = 1 - (dyn_fraud_loss / total_fraud_val)
    
    print(f"{'Value-Weighted Recall':<25} | {std_recall_val:.2%}               | {dyn_recall_val:.2%}               | {(dyn_recall_val - std_recall_val)*100:.2f} pts")

    # ---------------------------------------------------------
    # 6. SENSITIVITY ANALYSIS (Hypothesis Testing)
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS (Impact of Profit Margin)")
    print("="*80)
    
    alphas_to_test = [1.0, 2.0, 5.0]
    
    print(f"{'Alpha':<10} | {'Fraud Detected':<15} | {'Loss Reduction':<15}")
    print("-" * 50)

    for a in alphas_to_test:
        loss, decisions = evaluation.calculate_expected_loss(
            y_test_np, y_prob, 
            X_test['proposed_credit_limit'], 
            X_test['income'],
            alpha=a
        )
        
        current_values = a * mapped_values
        std_loss_current = 0
        for i in range(len(y_test_np)):
            if std_preds[i] == 0 and y_test_np[i] == 1: std_loss_current += risks[i]
            elif std_preds[i] == 1 and y_test_np[i] == 0: std_loss_current += current_values[i]
            
        improvement = std_loss_current - loss
        tn, fp, fn, tp = confusion_matrix(y_test_np, decisions).ravel()
        
        print(f"{a:<10} | {tp}/{fn+tp}           | €{improvement:,.2f}")

    print("-" * 50)

if __name__ == "__main__":
    main()