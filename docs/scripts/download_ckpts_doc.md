# Download Model Checkpoints

This script downloads model checkpoints from the Hugging Face Hub based on configurations specified in a YAML file.

## Functionality
- **Load Configuration**: Reads a YAML configuration file to get model details.
- **Download Model**: Downloads files for specified models from the Hugging Face Hub to a local directory.
  - Checks for a valid `local_dir` in the configuration; skips download if `local_dir` is null.
  - Creates the local directory if it doesn't exist.
  - Supports `allow` and `deny` patterns to filter files:
    - If `allow` patterns are specified, only those files are downloaded.
    - If no `allow` patterns are provided, all files are downloaded except those matching `deny` patterns.
  - Uses `hf_hub_download` from the `huggingface_hub` library with symlinks disabled.

## Command-Line Arguments
- `--config_path`: Path to the YAML configuration file (defaults to `configs/model_ckpts.yaml`).

## Dependencies
- `argparse`: For parsing command-line arguments.
- `os`: For directory creation.
- `yaml`: For reading the configuration file.
- `huggingface_hub`: For downloading files from the Hugging Face Hub.

## Usage
Run the script with:
```bash
python scripts/download_ckpts.py --config_path <path_to_yaml>
```
The script processes each model in the configuration file, printing the model ID and local directory for each.