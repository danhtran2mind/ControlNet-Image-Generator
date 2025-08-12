import random
import os
import sys
from src.controlnet_image_generator.infer import infer

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def run_inference(
    input_image,
    prompt,
    negative_prompt,
    num_steps,
    seed,
    width,
    height,
    guidance_scale,
    controlnet_conditioning_scale,
    use_random_seed=False,
):
    config_path = "configs/model_ckpts.yaml"
    
    if use_random_seed:
        seed = random.randint(0, 2 ** 32)
    
    try:
        result = infer(
            config_path=config_path,
            input_image=input_image,
            image_url=None,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=num_steps,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        )
        result = list(result)[0]
        return result, "Inference completed successfully"
    except Exception as e:
        return [], f"Error during inference: {str(e)}"