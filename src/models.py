from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

def get_base_model():
    """Returns the baseline Logistic Regression model."""
    # class_weight='balanced' helps with the 1.1% fraud rate
    return LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)

def get_random_forest():
    """Returns a Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

def get_lgbm_model():
    """Returns the LightGBM model."""
    return lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)