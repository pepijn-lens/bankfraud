from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

def get_base_model():
    """Returns the baseline Logistic Regression model."""
    # class_weight='balanced' helps with the 1.1% fraud rate
    return LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)

def get_lgbm_model():
    """Returns the LightGBM model."""
    return lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)