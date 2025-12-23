## Setup 
Make sure to create a virtual environment and install the requirements. 

## Download dataset
To download the dataset run `python download_data.py` from the project root. 

## Using `src/load_data.py`

Basic usage:

```python
import pandas as pd
from src.load_data import preprocess_data, split_data, normalize

df = pd.read_csv("data/2/Base.csv")

# 1) Preprocess full dataset (imputation, dropping columns, one-hot encoding)
df_prep = preprocess_data(df)

# 2) Split into train/val/test by month and separate labels
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_prep)

# 3) Normalize continuous features using statistics from X_train
X_train_norm, X_val_norm, X_test_norm, scaler = normalize(X_train, X_val, X_test)
```