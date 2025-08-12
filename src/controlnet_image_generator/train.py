import argparse
import os
import sys

# Add diffusers examples/controlnet to sys.path to import train_controlnet
sys.path.append(os.path.join(os.path.dirname(__file__), "..", 
                             "third_party", "diffusers", "examples", "controlnet"))
from train_controlnet import main

if __name__ == "__main__":
    main(*sys.argv[1:])