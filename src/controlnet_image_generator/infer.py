import cv2
import torch
from PIL import Image
import numpy as np
import yaml
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from utils.download import load_image
from utils.plot import image_grid

def load_config(config_path="configs/model_ckpts.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

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

def generate_images(pipe, prompts, pose_images, generators, negative_prompts, num_steps=20):
    return pipe(
        prompts,
        pose_images,
        negative_prompt=negative_prompts,
        generator=generators,
        num_inference_steps=num_steps
    ).images

def main():
    # Load configuration
    configs = load_config()
    
    # Initialize models
    controlnet_detector = OpenposeDetector.from_pretrained(
        configs[2]['model_id']  # lllyasviel/ControlNet
    )
    controlnet = initialize_controlnet(configs[0])
    pipe = initialize_pipeline(controlnet, configs[1])
    
    # Setup device
    device = setup_device(pipe)
    
    # Load and process image
    demo_image = load_image("https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/yoga1.jpeg")
    poses = [controlnet_detector(demo_image)]
    
    # Generate images
    generators = [torch.Generator(device="cpu").manual_seed(2) for _ in range(len(poses))]
    prompt = "a man is doing yoga"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    
    output_images = generate_images(
        pipe,
        [prompt] * len(generators),
        poses,
        generators,
        [negative_prompt] * len(generators)
    )
    
    # Display results
    image_grid(output_images, 2, 2)

if __name__ == "__main__":
    main()