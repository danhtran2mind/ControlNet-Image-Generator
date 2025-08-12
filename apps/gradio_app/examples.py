import os
import json
from PIL import Image
import gradio as gr

def load_examples(examples_base_path=os.path.join("apps", "gradio_app", 
    "assets", "examples", "Stable-Diffusion-2.1-Openpose-ControlNet")):

    """Load example configurations and input images from the Stable-Diffusion-2.1-Openpose-ControlNet directory."""
    examples = []
    
    # Iterate through example folders (e.g., '1', '2', '3', '4')
    for folder in os.listdir(examples_base_path):
        folder_path = os.path.join(examples_base_path, folder)
        config_path = os.path.join(folder_path, "config.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Extract configuration fields
                input_filename = config["input_image"]
                output_filename = config["output_image"]
                prompt = config.get("prompt", "a man is doing yoga")
                negative_prompt = config.get("negative_prompt", "monochrome, lowres, bad anatomy, worst quality, low quality")
                num_steps = config.get("num_steps", 30)
                seed = config.get("seed", 42)
                width = config.get("width", 512)
                height = config.get("height", 512)
                guidance_scale = config.get("guidance_scale", 7.5)
                controlnet_conditioning_scale = config.get("controlnet_conditioning_scale", 1.0)
                
                # Construct absolute path for input image
                input_image_path = os.path.join(folder_path, input_filename)
                output_image_path = os.path.join(folder_path, output_filename)
                # Check if input image exists
                if os.path.exists(input_image_path):
                    input_image_data = Image.open(input_image_path)
                    output_image_data = Image.open(output_image_path)
                    # Append example data in the order expected by Gradio inputs
                    examples.append([
                        input_image_data,  # Input image
                        prompt,
                        negative_prompt,
                        output_image_data,
                        num_steps,
                        seed,
                        width,
                        height,
                        guidance_scale,
                        controlnet_conditioning_scale,
                        False  # use_random_seed, hardcoded as per original gr.Examples
                    ])
                else:
                    print(f"Input image not found at {input_image_path}")
            
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {config_path}: {str(e)}")
            except Exception as e:
                print(f"Error processing example in {folder_path}: {str(e)}")
    
    return examples

def select_example(evt: gr.SelectData, examples_data):
    """Handle selection of an example to populate Gradio inputs."""
    example_index = evt.index
    # Extract example data
    # input_image_data, prompt, negative_prompt, output_image_data, num_steps, seed, width, height, guidance_scale, controlnet_conditioning_scale, use_random_seed = examples_data[example_index]
    (
        input_image_data,
        prompt,
        negative_prompt,
        output_image_data,
        num_steps,
        seed,
        width,
        height,
        guidance_scale,
        controlnet_conditioning_scale,
        use_random_seed,
    ) = examples_data[example_index]

    
    # Return values to update Gradio interface inputs and output message
    return (
        input_image_data,  # Input image
        prompt,            # Prompt
        negative_prompt,   # Negative prompt
        output_image_data, # Output image
        num_steps,         # Number of inference steps
        seed,              # Random seed
        width,             # Width
        height,            # Height
        guidance_scale,    # Guidance scale
        controlnet_conditioning_scale,  # ControlNet conditioning scale
        use_random_seed,   # Use random seed
        f"Loaded example {example_index + 1} with prompt: {prompt}"  # Output message
    )