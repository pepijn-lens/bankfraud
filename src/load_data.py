import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_data(df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """
    Preprocess the data by:
    - filling in the missing values.
    - dropping the device_fraud_count column.
    - turning the categorical columns into one hot encoded columns.
    """
    
    # For the features with comparatively little missing values, we just take the median of the feature
    for column in ["credit_risk_score", "device_distinct_emails_8w", "session_length_in_minutes", "current_address_months_count"]:
        median_risk_score = df[column].median()

        mask = df[column] == -1
        df.loc[mask, column] = int(median_risk_score)

    if "device_fraud_count" in df.columns:
        df = df.drop(columns=["device_fraud_count"])

    # Convert DataFrame to numpy array for fast, vectorized operation
    df_np = df.to_numpy()
    # For each row, check if all elements are the same (ignoring NaNs if present)
    same_value_mask = np.all(df_np == df_np[:, [0]], axis=1)
    rows_with_same_value = pd.Series(same_value_mask, index=df.index)
    # Drop those rows where all values are the same
    rows_to_remove = df[rows_with_same_value].index.tolist()
    df = df.drop(index=rows_to_remove)

    # Turn the categorical columns into one hot encoded columns
    categorical_types = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_types, drop_first=True)

    return df


def normalize(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Normalize continuous numerical features using statistics from the training set.

    - Fits a StandardScaler on continuous (non-binary, non-boolean) numeric columns of X_train.
    - Applies the same transformation to X_train, X_val and X_test.
    - Leaves binary and boolean / one-hot encoded features unchanged.

    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler
    """

    # Identify numeric and boolean columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    bool_cols = X_train.select_dtypes(include=["bool"]).columns

    # Detect binary integer columns (only {0,1} in training data, not already bool)
    binary_int_cols: list[str] = []
    for col in numeric_cols:
        if col in bool_cols:
            continue
        unique_vals = pd.Series(X_train[col].dropna().unique())
        if unique_vals.nunique() <= 2 and set(unique_vals.tolist()).issubset({0, 1}):
            binary_int_cols.append(col)

    # Continuous columns are numeric but not boolean and not binary {0,1}
    continuous_cols = [
        col for col in numeric_cols
        if col not in bool_cols and col not in binary_int_cols
    ]

    if not continuous_cols:
        # Nothing to normalize
        return X_train.copy(), X_val.copy(), X_test.copy(), StandardScaler()

    scaler = StandardScaler()
    scaler.fit(X_train[continuous_cols])

    # Work on copies to avoid modifying inputs in-place
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[continuous_cols] = scaler.transform(X_train[continuous_cols])
    X_val_scaled[continuous_cols] = scaler.transform(X_val[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the data into training, validation and testing sets.
    The first 6 months are for training, the 7th month is for validation, and the 8th month is for testing.
    """
    # Split the data based on the month column
    # The first 6 months are for training
    X_train = df[df['month'] < 6]
    # The 7th month is for validation
    X_val = df[df['month'] == 6]
    # The 8th month is for testing
    X_test = df[df['month'] == 7]

    y_train = X_train.pop('fraud_bool')
    y_val = X_val.pop('fraud_bool')
    y_test = X_test.pop('fraud_bool')

    return X_train, X_val, X_test, y_train, y_val, y_test