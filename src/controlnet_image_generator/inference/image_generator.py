import torch
import os
import re
import uuid
from tqdm import tqdm

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

def save_images(images, output_dir, prompt, use_prompt_as_output_name, index_offset=0):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(tqdm(images, desc="Saving images")):
        if use_prompt_as_output_name:
            sanitized_prompt = re.sub(r'[^\w\s-]', '', prompt).replace(' ', '_').lower()
            filename = f"{sanitized_prompt}_{i + index_offset}.png"
        else:
            filename = f"{uuid.uuid4()}_{i + index_offset}.png"
        img.save(os.path.join(output_dir, filename))