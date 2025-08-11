import argparse
import os
import yaml
from huggingface_hub import hf_hub_download

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def download_model(model_config):
    model_id = model_config["model_id"]
    local_dir = model_config["local_dir"]
    
    if local_dir is None:
        print(f"Skipping download for {model_id}: local_dir is null")
        return
    
    os.makedirs(local_dir, exist_ok=True)
    
    allow_patterns = model_config.get("allow", [])
    deny_patterns = model_config.get("deny", [])
    
    if allow_patterns:
        for file in allow_patterns:
            hf_hub_download(
                repo_id=model_id,
                filename=file,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
    else:
        print(f"Warning: No allow patterns specified for {model_id}. Skipping download of unspecified files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model checkpoints from Hugging Face Hub")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/model_ckpts.yaml",
        help="Path to the configuration YAML file"
    )

    args = parser.parse_args()

    config = load_config(args.config_path)
    
    for model_config in config:
        print(f"Processing {model_config['model_id']} (local_dir: {model_config['local_dir']})")
        download_model(model_config)