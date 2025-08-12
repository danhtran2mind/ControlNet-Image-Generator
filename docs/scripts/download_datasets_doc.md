# Download Datasets

This script downloads datasets from Hugging Face using configuration details specified in a YAML file.

## Functionality
- **Load Configuration**: Reads dataset details from a YAML configuration file.
- **Download Dataset**: Downloads datasets from Hugging Face if the platform is specified as 'HuggingFace' in the configuration.
- **Command-Line Argument**: Accepts a path to the configuration file via the `--config_path` argument (defaults to `configs/datasets_info.yaml`).
- **Dataset Information**: Extracts dataset name and local storage directory from the configuration, splits the dataset name into user and model hub components, and saves the dataset to the specified directory.
- **Verification**: Prints dataset details, including user name, model hub name, storage location, and dataset information for confirmation.
- **Platform Check**: Only processes datasets from Hugging Face; unsupported platforms are flagged with a message.

## Usage
Run the script with the command:  
`python script_name.py --config_path path/to/config.yaml`

The configuration file should contain:
- `dataset_name`: Format as `user_name/model_hub_name`.
- `local_dir`: Directory to save the dataset.
- `platform`: Must be set to `HuggingFace` for the script to process.