```python
import os
import shutil
import subprocess
import argparse

def setup_diffusers(target_dir):
    # Define paths
    diffusers_dir = os.path.join(target_dir, "diffusers")
    
    # Create third_party directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Check if diffusers already exists in third_party
    if os.path.exists(diffusers_dir):
        print(f"Diffusers already exists in {target_dir}. Skipping clone.")
        return
    
    # Clone diffusers repository
    subprocess.run(["git", "clone", "https://github.com/huggingface/diffusers"], 
                  cwd=target_dir, check=True)
    
    # Change to diffusers directory and install
    original_dir = os.getcwd()
    os.chdir(diffusers_dir)
    try:
        subprocess.run(["pip", "install", "-e", "."], check=True)
    finally:
        os.chdir(original_dir)
    
    print(f"Diffusers successfully cloned and installed to {diffusers_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup diffusers in a specified directory.")
    parser.add_argument("--target-dir", type=str, default="src/third_party",
                       help="Target directory to clone diffusers into (default: src)")
    
    args = parser.parse_args()
    setup_diffusers(args.target_dir)
