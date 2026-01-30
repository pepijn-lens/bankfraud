## Setup 
We use uv to manage dependencies. 
- Install uv: https://docs.astral.sh/uv/getting-started/installation/
- Run `uv sync`, to create the .venv, install all dependencies, and ensure the .lock file is up to date.
- To run a script using the .venv, use `uv run <command>`, example: `uv run python src/download_data.py`.
  - Alternatively you can activate the shell you can use `source .venv/bin/activate` on macOS / Linux, or `.venv\Scripts\activate` on Windows.
- You add and remove dependencies with `uv add <dep>` and `uv remove <dep>`, don't add things to the pyproject.toml manually. 

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

## Training and Loading Classifiers

### Training

Classifiers are trained in `src/training.py` using a 75/25 stratified train/test split:

- **Tree-based models** (RandomForest, XGBoost): Use `train_and_save_classifiers()` which performs hyperparameter tuning with `RandomizedSearchCV` and cross-validation. SMOTENC is applied inside the CV pipeline to avoid data leakage. ⚠️ **Warning**: This takes 2-6+ hours to complete. Models are saved to `models/rf_model.pkl` and `models/xgb_model.pkl`.

- **Logistic Regression**: Use `train_logistic_regression()` which applies SMOTENC and trains a logistic regression model. Saved to `models/lr_model.pkl`.

### Loading Models

Models are saved as pickle files in the `models/` directory. To load a model:

```python
import pickle
from pathlib import Path

# Load a specific model
with open(Path("models/rf_model.pkl"), "rb") as f:
    model = pickle.load(f)

# Or use evaluate_on_test_set() which automatically loads all available models
from src.training import evaluate_on_test_set
evaluate_on_test_set()  # Loads models and evaluates on test set
```

### Test Set Loading

The test set is prepared using `prepare_data()` in `src/training.py`, which:
1. Loads and preprocesses the data globally
2. Performs a stratified 75/25 train/test split (same `random_state=42`)
3. Returns `X_train, X_test, y_train, y_test, encoded_features`

When calling `evaluate_on_test_set()`, if no test set is provided, it automatically calls `prepare_data()` to generate the test set using the same split as training.