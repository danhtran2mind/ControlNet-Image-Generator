import argparse
import os
import yaml
from huggingface_hub import hf_hub_download

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_model(model_id, file_name, output_dir):
    try:
        print(f"Downloading {file_name} for model {model_id}...")
        hf_hub_download(
            repo_id=model_id,
            filename=file_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded {file_name} to {output_dir}")
    except Exception as e:
        print(f"Error downloading {file_name} for {model_id}: {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download model checkpoints from Hugging Face")
    parser.add_argument(
        '--config_path', 
        type=str, 
        default='configs/model_ckpts.yaml',
        help='Path to the model checkpoints config file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./ckpts',
        help='Output directory for downloaded checkpoints'
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configuration
    config = load_config(args.config_path)

    # Process each model in the config
    for model_config in config:
        model_id = model_config.get('model_id')
        allow_files = model_config.get('allow', [])
        deny_files = model_config.get('deny', [])

        # Download allowed files
        for file_name in allow_files:
            if file_name not in deny_files:
                download_model(model_id, file_name, args.output_dir)
