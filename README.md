## Setup 
We use uv to manage dependencies. 
- Install uv: https://docs.astral.sh/uv/getting-started/installation/
- Run `uv sync`, to create the .venv, install all dependencies, and ensure the .lock file is up to date.
- To run a script using the .venv, use `uv run <command>`, example: `uv run python src/download_data.py`.
  - Alternatively you can activate the shell you can use `source .venv/bin/activate` on macOS / Linux, or `.venv\Scripts\activate` on Windows.
- You add and remove dependencies with `uv add <dep>` and `uv remove <dep>`, don't add things to the pyproject.toml manually. 

## Download dataset
To download the dataset run "python download_data.py" from the project root. 