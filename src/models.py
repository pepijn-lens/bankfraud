from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

def get_base_model():
    """Returns the baseline Logistic Regression model."""
    # class_weight='balanced' helps with the 1.1% fraud rate
    return LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)

def get_random_forest():
    """Returns a Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

def get_lgbm_model():
    """Returns the LightGBM model with regularization to prevent overfitting."""
    return lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=20,  
        min_child_samples=10,
        learning_rate=0.1,  
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

def get_xgboost_model():
    """Returns the XGBoost model."""
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,  
        learning_rate=0.05, 
        scale_pos_weight=90,  
        reg_alpha=1.0,
        reg_lambda=1.0, 
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1
    )