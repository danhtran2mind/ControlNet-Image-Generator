# import os
# import sys

# # Add diffusers examples/controlnet to sys.path to import train_controlnet
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", 
#                              "third_party", "diffusers", "examples", "controlnet"))
# from train_controlnet import main

# if __name__ == "__main__":
#     main(sys.argv[1:])

import argparse
import os
import sys

# Add diffusers examples/controlnet to sys.path to import train_controlnet
sys.path.append(os.path.join(os.path.dirname(__file__), "..", 
                             "third_party", "diffusers", "examples", "controlnet"))
from train_controlnet import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ControlNet model")
    # Define arguments to match train_controlnet.py expectations
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--checkpoints_total_limit", type=int, default=2)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    # Optional: Add other arguments that train_controlnet.py might expect
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="./log")
    args = parser.parse_args()
    main(args)