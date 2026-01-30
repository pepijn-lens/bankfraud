from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_base_model():
    """Returns the baseline Logistic Regression model."""
    # class_weight='balanced' helps with the 1.1% fraud rate
    return LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)

def get_random_forest():
    """Returns a Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=100,
        max_features='sqrt',
        max_depth=10,
        criterion='gini',
        random_state=42,
        n_jobs=-1
    )


def get_xgb_model():
    """Returns the XGBoost model."""
    return XGBClassifier(
        random_state=42,
        n_jobs=-1,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        min_child_weight=4,
        subsample=0.8,
        colsample_bytree=0.6
    )