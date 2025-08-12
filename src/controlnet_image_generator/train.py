import argparse
import os
import sys

# Add diffusers examples/controlnet to sys.path to import train_controlnet
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", 
                             "third_party", "diffusers", "examples", "controlnet"))

from train_controlnet import main

def parse_args():
    parser = argparse.ArgumentParser(description="Train a ControlNet model with Stable Diffusion")
    parser.add_argument("--pretrained_model_name_or_path", type=str, 
                        default="stabilityai/stable-diffusion-2-1",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--resume_from_checkpoint", type=str, required=False,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--output_dir", type=str, 
                        default="./ckpts/Stable-Diffusion-2.1-Openpose-ControlNet",
                        help="Output directory for saving checkpoints")
    parser.add_argument("--dataset_name", type=str, 
                        default="HighCWu/open_pose_controlnet_subset",
                        help="Name of the dataset on Hugging Face or path to dataset")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Image resolution for training")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for training")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Use 8-bit Adam optimizer")
    parser.add_argument("--num_train_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision training mode")
    parser.add_argument("--checkpoints_total_limit", type=int, default=2,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--checkpointing_steps", type=int, default=500,
                        help="Save a checkpoint every X steps")
    parser.add_argument("--validation_steps", type=int, default=100,
                        help="Run validation every X steps")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

