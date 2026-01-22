import kagglehub
import shutil

from constants import DATA_DIR

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download latest version
path = kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022")

shutil.move(path, DATA_DIR / "bank_account_fraud_dataset")

print("Dataset downloaded and moved to the data folder")