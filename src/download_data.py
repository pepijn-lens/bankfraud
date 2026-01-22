from pathlib import Path

import kagglehub
import shutil

from constants import DATA_DIR

def download_data(output_dir: Path):
    # Download latest version
    output_dir.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022")
    shutil.move(path, output_dir)
    print(f"Dataset downloaded and moved to {output_dir}")

if __name__ == "__main__":
    download_data(DATA_DIR / "bank_account_fraud_dataset")