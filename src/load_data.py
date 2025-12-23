import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by:
    - filling in the missing values.
    - dropping the device_fraud_count column.
    - turning the categorical columns into one hot encoded columns.
    """
    
    # For the features with comparatively little missing values, we just take the median of the feature
    for column in ["credit_risk_score", "device_distinct_emails_8w", "session_length_in_minutes", "current_address_months_count"]:
        median_risk_score = df[column].median()

        df[df[column] == -1] = int(median_risk_score)

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