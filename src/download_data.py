from constants import DATA_DIR
from load_data import download_data

if __name__ == "__main__":
    download_data(DATA_DIR / "bank_account_fraud_dataset")