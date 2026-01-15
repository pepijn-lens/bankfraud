## Setup 
We use uv to manage dependencies. 
- Install uv: https://docs.astral.sh/uv/getting-started/installation/
- Run `uv sync`, to create the .venv, install all dependencies, and ensure the .lock file is up to date.
- To run a script using the .venv, use `uv run <command>`, example: `uv run python src/download_data.py`.
  - Alternatively you can activate the shell you can use `source .venv/bin/activate` on macOS / Linux, or `.venv\Scripts\activate` on Windows.
- You add and remove dependencies with `uv add <dep>` and `uv remove <dep>`, don't add things to the pyproject.toml manually. 

## Download dataset
To download the dataset run `python download_data.py` from the project root. We split the data into training and test sets before doing our experiment. Please run `python src/load_data.py` to make this split.

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