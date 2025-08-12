import os
import sys
import subprocess
import gradio as gr
import torch

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.controlnet_image_generator.infer import (
    load_config,
    find_config_by_model_id,
    initialize_controlnet,
    initialize_pipeline,
    initialize_controlnet_detector,
    setup_device,
    load_input_image,
    detect_poses,
    generate_images,
    save_images
)



def run_setup_script():
    setup_script = os.path.join(os.path.dirname(__file__),
                                "gradio_app", "setup_scripts.py")
    try:
        result = subprocess.run(["python", setup_script], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Setup script failed with error: {e.stderr}")
        return f"Setup script failed: {e.stderr}"

def run_inference(
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
    # Validate inputs
    if not input_image and not image_url:
        return None, "Please provide either an input image or an image URL"
    
    # Load configuration
    config_path = "configs/model_ckpts.yaml"
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
        os.makedirs(output_dir, exist_ok=True)
        save_images(output_images, output_dir, prompt, use_prompt_as_output_name)
    
    return output_images, "Image generation completed successfully"

def create_gui():
    
    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# ControlNet Image Generation with Pose Detection")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="Input Image")
                image_url = gr.Textbox(label="Image URL")
                prompt = gr.Textbox(
                    label="Prompt",
                    value="a man is doing yoga"
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="monochrome, lowres, bad anatomy, worst quality, low quality"
                )
                
                with gr.Row():
                    num_steps = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=20,
                        step=1,
                        label="Number of Inference Steps"
                    )
                    seed = gr.Slider(
                        minimum=0,
                        maximum=1000,
                        value=2,
                        step=1,
                        label="Random Seed"
                    )
                
                with gr.Row():
                    width = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Width"
                    )
                    height = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Height"
                    )
                
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=7.5,
                    step=0.1,
                    label="Guidance Scale"
                )
                controlnet_conditioning_scale = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="ControlNet Conditioning Scale"
                )
                
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="tests/test_data"
                )
                use_prompt_as_output_name = gr.Checkbox(
                    label="Use Prompt as Output Name"
                )
                save_output = gr.Checkbox(
                    label="Save Output Images"
                )
                
                submit_button = gr.Button("Generate Images")
            
            with gr.Column():
                output_images = gr.Gallery(label="Generated Images")
                output_message = gr.Textbox(label="Status")
        
        submit_button.click(
            fn=run_inference,
            inputs=[
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
            ],
            outputs=[output_images, output_message]
        )

        return demo

if __name__ == "__main__":
    run_setup_script()
    demo = create_gui()
    demo.launch(share=True)
