import os
import sys
import argparse

# Add diffusers examples/controlnet to sys.path to import train_controlnet
sys.path.append(os.path.join(os.path.dirname(__file__), "..", 
                             "third_party", "diffusers", "examples", "controlnet"))
from train_controlnet import main

def parse_args():
    parser = argparse.ArgumentParser(description="Train a ControlNet model with Stable Diffusion",
                                    allow_abbrev=False)
    # Don't define any specific arguments to mimic sys.argv[1:]
    _, all_args = parser.parse_known_args()
    return all_args

if __name__ == "__main__":
    args = parse_args()
    main(args)