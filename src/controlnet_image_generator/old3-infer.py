import torch
import argparse
from inference.config_loader import load_config, find_config_by_model_id
from inference.model_initializer import (
    initialize_controlnet, 
    initialize_pipeline, 
    initialize_controlnet_detector
)
from inference.device_manager import setup_device
from inference.image_processor import load_input_image, detect_poses
from inference.image_generator import generate_images, save_images

def infer(
    config_path,
    input_image,
    image_url,
    prompt,
    negative_prompt,
    num_steps,
    seed,
    width,
    height,
    guidance_scale,
    controlnet_conditioning_scale,
    output_dir,
    use_prompt_as_output_name,
    save_output
):
    # Load configuration
    configs = load_config(config_path)
    
    # Initialize models
    controlnet_detector_config = find_config_by_model_id(configs, "lllyasviel/ControlNet")
    controlnet_config = find_config_by_model_id(configs, 
                                                "danhtran2mind/Stable-Diffusion-2.1-Openpose-ControlNet")
    pipeline_config = find_config_by_model_id(configs, 
                                              "stabilityai/stable-diffusion-2-1")
    
    controlnet_detector = initialize_controlnet_detector(controlnet_detector_config)
    controlnet = initialize_controlnet(controlnet_config)
    pipe = initialize_pipeline(controlnet, pipeline_config)
    
    # Setup device
    device = setup_device(pipe)
    
    # Load and process image
    demo_image = load_input_image(input_image, image_url)
    poses = detect_poses(controlnet_detector, demo_image)
    
    # Generate images
    generators = [torch.Generator(device="cpu").manual_seed(seed + i) for i in range(len(poses))]
    output_images = generate_images(
        pipe,
        [prompt] * len(generators),
        poses,
        generators,
        [negative_prompt] * len(generators),
        num_steps,
        guidance_scale,
        controlnet_conditioning_scale,
        width,
        height
    )
    
    # Save images if required
    if save_output:
        save_images(output_images, output_dir, prompt, use_prompt_as_output_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ControlNet image generation with pose detection")
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
    parser.add_argument("--save_output", action="store_true",
                        help="Save generated images to output directory")
    
    args = parser.parse_args()
    infer(
        config_path=args.config_path,
        input_image=args.input_image,
        image_url=args.image_url,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_steps=args.num_steps,
        seed=args.seed,
        width=args.width,
        height=args.height,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        output_dir=args.output_dir,
        use_prompt_as_output_name=args.use_prompt_as_output_name,
        save_output=args.save_output
    )