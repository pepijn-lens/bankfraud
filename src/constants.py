# This file contains some constants values to be used across all other files.
# This file should never import another file, it should only be imported.

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

