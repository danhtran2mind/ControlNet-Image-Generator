import argparse
import yaml
from datasets import load_dataset


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def download_huggingface_dataset(config):
    # Get dataset details from config
    dataset_name = config['dataset_name']
    local_dir = config['local_dir']

    # Split dataset name into user_name and model_hub_name
    user_name, model_hub_name = dataset_name.split('/')

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset(dataset_name, cache_dir=local_dir)

    # Print information for verification
    print(f"User Name: {user_name}")
    print(f"Model Hub Name: {model_hub_name}")
    print(f"Dataset saved to: {local_dir}")
    print(f"Dataset info: {ds}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download dataset from Hugging Face")
    parser.add_argument('--config_path', 
                        type=str, 
                        default='configs/datasets_info.yaml', 
                        help='Path to the dataset configuration YAML file')
    
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config_path)

    # Download dataset if platform is HuggingFace
    if config['platform'] == 'HuggingFace':
        download_huggingface_dataset(config)
    else:
        print(f"Unsupported platform: {config['platform']}")