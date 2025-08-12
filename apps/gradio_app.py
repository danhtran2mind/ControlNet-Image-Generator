import os
import sys
import subprocess
import gradio as gr
import torch

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.controlnet_image_generator.infer import infer

def run_setup_script():
    setup_script = os.path.join(os.path.dirname(__file__), "gradio_app", "setup_scripts.py")
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
    # Define default config path
    config_path = "configs/model_ckpts.yaml"
    
    # Call the infer function from infer.py
    try:
        result = infer(
            config_path=config_path,
            input_image=input_image,
            image_url=image_url,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=num_steps,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            output_dir=output_dir,
            use_prompt_as_output_name=use_prompt_as_output_name,
            save_output=save_output
        )
        return result, "Inference completed successfully"
    except Exception as e:
        return [], f"Error during inference: {str(e)}"

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