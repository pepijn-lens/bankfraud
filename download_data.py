import kagglehub
import os
import shutil

os.makedirs("data", exist_ok=True)

# Download latest version
path = kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022")

shutil.move(path, "data/bank_account_fraud_dataset.zip")

print("Dataset downloaded and moved to the data folder")