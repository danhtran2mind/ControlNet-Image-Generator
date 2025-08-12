import cv2
import torch
from PIL import Image
import numpy as np
import yaml
import argparse
from controlnet_aux import OpenposeDetector
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    UniPCMultistepScheduler
)

from utils.download import load_image
from utils.plot import image_grid
import os
from tqdm import tqdm
import re
import uuid

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")

def initialize_controlnet(config):
    model_id = config['model_id']
    local_dir = config.get('local_dir', model_id)
    return ControlNetModel.from_pretrained(
        local_dir if local_dir != model_id else model_id,
        torch_dtype=torch.float16
    )

def initialize_pipeline(controlnet, config):
    model_id = config['model_id']
    local_dir = config.get('local_dir', model_id)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        local_dir if local_dir != model_id else model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

def setup_device(pipe):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    pipe.to(device)
    return device

def generate_images(pipe, prompts, pose_images, generators, negative_prompts, num_steps, guidance_scale, controlnet_conditioning_scale, width, height):
    return pipe(
        prompts,
        pose_images,
        negative_prompt=negative_prompts,
        generator=generators,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        width=width,
        height=height
    ).images

def infer(args):
    # Load configuration
    configs = load_config(args.config_path)
    
    # Initialize models
    controlnet_detector = OpenposeDetector.from_pretrained(
        configs[2]['model_id']  # lllyasviel/ControlNet
    )
    controlnet = initialize_controlnet(configs[0])
    pipe = initialize_pipeline(controlnet, configs[1])
    
    # Setup device
    device = setup_device(pipe)
    
    # Load and process image
    try:
        if args.input_image:
            demo_image = Image.open(args.input_image).convert("RGB")
        elif args.image_url:
            demo_image = load_image(args.image_url)
        else:
            raise ValueError("Either --input_image or --image_url must be provided")
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")
    
    poses = [controlnet_detector(demo_image)]
    
    # Generate images
    generators = [torch.Generator(device="cpu").manual_seed(args.seed + i) for i in range(len(poses))]
    
    output_images = generate_images(
        pipe,
        [args.prompt] * len(generators),
        poses,
        generators,
        [args.negative_prompt] * len(generators),
        args.num_steps,
        args.guidance_scale,
        args.controlnet_conditioning_scale,
        args.width,
        args.height
    )
    
    # Save images if save_output is True
    if args.save_output:
        os.makedirs(args.output_dir, exist_ok=True)
        for i, img in enumerate(tqdm(output_images, desc="Saving images")):
            if args.use_prompt_as_output_name:
                # Sanitize prompt for filename (replace spaces and special characters)
                sanitized_prompt = re.sub(r'[^\w\s-]', '', args.prompt).replace(' ', '_').lower()
                filename = f"{sanitized_prompt}_{i}.png"
            else:
                # Use UUID for filename
                filename = f"{uuid.uuid4()}_{i}.png"
            img.save(os.path.join(args.output_dir, filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ControlNet image generation with pose detection")
    # Create mutually exclusive group for input_image and image_url
    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument("--input_image", type=str, default=None,
                             help="Path to local input image (default: tests/test_data/yoga1.jpg)")
    image_group.add_argument("--image_url", type=str, default=None,
                             help="URL of input image (e.g., https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/yoga1.jpeg)")
    
    parser.add_argument("--config_path", type=str, default="configs/model_ckpts.yaml", 
                        help="Path to configuration YAML file")
    parser.add_argument("--prompt", type=str, default="a man is doing yoga",
                        help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, 
                        default="monochrome, lowres, bad anatomy, worst quality, low quality",
                        help="Negative prompt for image generation")
    parser.add_argument("--num_steps", type=int, default=20,
                        help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=2,
                        help="Random seed for generation")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of the generated image")
    parser.add_argument("--height", type=int, default=512,
                        help="Height of the generated image")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for prompt adherence")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0,
                        help="ControlNet conditioning scale")
    parser.add_argument("--output_dir", type=str, default="tests/test_data",
                        help="Directory to save generated images")
    parser.add_argument("--use_prompt_as_output_name", action="store_true",
                        help="Use prompt as part of output image filename")
    parser.add_argument("--save_output", action="store_true artr",
                        help="Save generated images to output directory")
    
    args = parser.parse_args()
    infer(args)

# Using image_url
# python script.py \
#     --config_path configs/model_ckpts.yaml \
#     --image_url https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/yoga1.jpeg \
#     --prompt "a man is doing yoga in a serene park" \
#     --negative_prompt "monochrome, lowres, bad anatomy" \
#     --num_steps 30 \
#     --seed 42 \
#     --width 512 \
#     --height 512 \
#     --guidance_scale 7.5 \
#     --controlnet_conditioning_scale 0.8 \
#     --output_dir "tests/test_data" \
#     --save_output

# Using input_image
# python script.py \
#     --config_path configs/model_ckpts.yaml \
#     --input_image "tests/test_data/yoga1.jpg" \
#     --prompt "a man is doing yoga in a serene park" \
#     --negative_prompt "monochrome, lowres, bad anatomy" \
#     --num_steps 30 \
#     --seed 42 \
#     --width 512 \
#     --height 512 \
#     --guidance_scale 7.5 \
#     --controlnet_conditioning_scale 0.8 \
#     --output_dir "tests/test_data" \ 
#     --save_output