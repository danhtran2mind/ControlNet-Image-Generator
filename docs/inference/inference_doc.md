# ControlNet Image Generation with Pose Detection

This document provides a comprehensive overview of a Python script designed for image generation using ControlNet with pose detection, integrated with the Stable Diffusion model. The script processes an input image to detect human poses and generates new images based on a text prompt, guided by the detected poses.

## Purpose

The script enables users to generate images that adhere to specific poses extracted from an input image, combining the power of ControlNet for pose conditioning with Stable Diffusion for high-quality image synthesis. It is particularly useful for applications requiring pose-guided image generation, such as creating stylized images of people in specific poses (e.g., yoga, dancing) based on a reference image.

## Dependencies

The script relies on the following Python libraries and custom modules:

- **Standard Libraries**:
  - `torch`: For tensor operations and deep learning model handling.
  - `argparse`: For parsing command-line arguments.
  - `os`: For file and directory operations.
  - `sys`: For modifying the Python path to include the project root.

- **Custom Modules** (assumed to be part of the project structure):
  - `inference.config_loader`:
    - `load_config`: Loads model configurations from a YAML file.
    - `find_config_by_model_id`: Retrieves specific model configurations by ID.
  - `inference.model_initializer`:
    - `initialize_controlnet`: Initializes the ControlNet model.
    - `initialize_pipeline`: Initializes the Stable Diffusion pipeline.
    - `initialize_controlnet_detector`: Initializes the pose detection model.
  - `inference.device_manager`:
    - `setup_device`: Configures the computation device (e.g., CPU or GPU).
  - `inference.image_processor`:
    - `load_input_image`: Loads the input image from a local path or URL.
    - `detect_poses`: Detects human poses in the input image.
  - `inference.image_generator`:
    - `generate_images`: Generates images using the pipeline and pose conditions.
    - `save_images`: Saves generated images to the specified directory.

## Script Structure

The script is organized into the following components:

1. **Imports and Path Setup**:
   - Imports necessary libraries and adds the project root directory to the Python path for accessing custom modules.
   - Ensures the script can locate custom modules regardless of the execution context.

2. **Global Variables**:
   - Defines three global variables to cache initialized models:
     - `controlnet_detector`: For pose detection.
     - `controlnet`: For pose-guided conditioning.
     - `pipe`: The Stable Diffusion pipeline.
   - These variables persist across multiple calls to the `infer` function to avoid redundant model initialization.

3. **Main Function: `infer`**:
   - The core function that orchestrates the image generation process.
   - Takes configurable parameters for input, model settings, and output options.

4. **Command-Line Interface**:
   - Uses `argparse` to provide a user-friendly interface for running the script with customizable parameters.

## Main Function: `infer`

The `infer` function handles the end-to-end process of loading models, processing input images, detecting poses, generating images, and optionally saving the results.

### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `config_path` | `str` | Path to the configuration YAML file. | `"configs/model_ckpts.yaml"` |
| `input_image` | `str` | Path to the local input image. Mutually exclusive with `image_url`. | `None` |
| `image_url` | `str` | URL of the input image. Mutually exclusive with `input_image`. | `None` |
| `prompt` | `str` | Text prompt for image generation. | `"a man is doing yoga"` |
| `negative_prompt` | `str` | Negative prompt to avoid undesired features. | `"monochrome, lowres, bad anatomy, worst quality, low quality"` |
| `num_steps` | `int` | Number of inference steps. | `20` |
| `seed` | `int` | Random seed for reproducibility. | `2` |
| `width` | `int` | Width of the generated image (pixels). | `512` |
| `height` | `int` | Height of the generated image (pixels). | `512` |
| `guidance_scale` | `float` | Guidance scale for prompt adherence. | `7.5` |
| `controlnet_conditioning_scale` | `float` | ControlNet conditioning scale for pose influence. | `1.0` |
| `output_dir` | `str` | Directory to save generated images. | `tests/test_data` |
| `use_prompt_as_output_name` | `bool` | Use prompt in output filenames. | `False` |
| `save_output` | `bool` | Save generated images to `output_dir`. | `False` |

### Workflow

1. **Configuration Loading**:
   - Loads model configurations from `config_path` using `load_config`.
   - Retrieves specific configurations for:
     - Pose detection model (`lllyasviel/ControlNet`).
     - ControlNet model (`danhtran2mind/Stable-Diffusion-2.1-Openpose-ControlNet`).
     - Stable Diffusion pipeline (`stabilityai/stable-diffusion-2-1`).

2. **Model Initialization**:
   - Checks if `controlnet_detector`, `controlnet`, or `pipe` are `None`.
   - If `None`, initializes them using the respective configurations to avoid redundant loading.

3. **Device Setup**:
   - Configures the computation device (e.g., CPU or GPU) for the pipeline using `setup_device`.

4. **Image Processing**:
   - Loads the input image from either `input_image` or `image_url` using `load_input_image`.
   - Detects poses in the input image using `detect_poses` with the `controlnet_detector`.

5. **Image Generation**:
   - Creates a list of random number generators seeded with `seed + i` for each detected pose.
   - Generates images using `generate_images`, passing:
     - The pipeline (`pipe`).
     - Repeated prompts and negative prompts for each pose.
     - Detected poses as conditioning inputs.
     - Generators for reproducibility.
     - Parameters like `num_steps`, `guidance_scale`, `controlnet_conditioning_scale`, `width`, and `height`.

6. **Output Handling**:
   - If `save_output` is `True`, saves the generated images to `output_dir` using `save_images`.
   - If `use_prompt_as_output_name` is `True`, incorporates the prompt into the output filenames.
   - Returns the list of generated images.

## Command-Line Interface

The script includes a command-line interface using `argparse` for flexible execution.

### Arguments Table

| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| `--input_image` | `str` | `tests/test_data/yoga1.jpg` | Path to the local input image. Mutually exclusive with `--image_url`. |
| `--image_url` | `str` | `None` | URL of the input image (e.g., `https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/yoga1.jpeg`). Mutually exclusive with `--input_image`. |
| `--config_path` | `str` | `configs/model_ckpts.yaml` | Path to the configuration YAML file for model settings. |
| `--prompt` | `str` | `"a man is doing yoga"` | Text prompt for image generation. |
| `--negative_prompt` | `str` | `"monochrome, lowres, bad anatomy, worst quality, low quality"` | Negative prompt to avoid undesired features in generated images. |
| `--num_steps` | `int` | `20` | Number of inference steps for image generation. |
| `--seed` | `int` | `2` | Random seed for reproducible generation. |
| `--width` | `int` | `512` | Width of the generated image in pixels. |
| `--height` | `int` | `512` | Height of the generated image in pixels. |
| `--guidance_scale` | `float` | `7.5` | Guidance scale for prompt adherence during generation. |
| `--controlnet_conditioning_scale` | `float` | `1.0` | ControlNet conditioning scale to balance pose influence. |
| `--output_dir` | `str` | `tests/test_data` | Directory to save generated images. |
| `--use_prompt_as_output_name` | Flag | `False` | If set, incorporates the prompt into output image filenames. |
| `--save_output` | Flag | `False` | If set, saves generated images to the specified output directory. |

### Example Usage

```bash
python script.py --input_image tests/test_data/yoga1.jpg --prompt "a woman doing yoga in a park" --num_steps 30 --guidance_scale 8.0 --save_output --use_prompt_as_output_name
```

This command:
- Uses the local image `tests/test_data/yoga1.jpg` as input.
- Generates images with the prompt `"a woman doing yoga in a park"`.
- Runs for 30 inference steps with a guidance scale of 8.0.
- Saves the output images to `tests/test_data`, with filenames including the prompt.

Alternatively, using a URL:

```bash
python script.py --image_url https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/yoga1.jpeg --prompt "a person practicing yoga at sunset" --save_output
```

This command uses an online image and saves the generated images without using the prompt in filenames.

## Notes

- **Configuration File**: The script assumes a `configs/model_ckpts.yaml` file exists with configurations for the required models (`lllyasviel/ControlNet`, `danhtran2mind/Stable-Diffusion-2.1-Openpose-ControlNet`, `stabilityai/stable-diffusion-2-1`). Ensure this file is correctly formatted and accessible.
- **Input Requirements**: The input image (local or URL) should contain at least one person for effective pose detection.
- **Model Caching**: Global variables cache the models to improve performance for multiple inferences within the same session.
- **Device Compatibility**: The `setup_device` function determines the computation device. Ensure compatible hardware (e.g., GPU) is available for optimal performance.
- **Output Flexibility**: The script supports generating multiple images if multiple poses are detected, with each image conditioned on one pose.
- **Error Handling**: The script assumes the custom modules handle errors appropriately. Users should verify that input paths, URLs, and model configurations are valid.

## Potential Improvements

- Add error handling for invalid inputs or missing configuration files.
- Support batch processing for multiple input images.
- Allow dynamic model selection via command-line arguments instead of hardcoded model IDs.
- Include options for adjusting pose detection sensitivity or other model-specific parameters.

## Conclusion

This script provides a robust framework for pose-guided image generation using ControlNet and Stable Diffusion. Its modular design and command-line interface make it suitable for both one-off experiments and integration into larger workflows. By leveraging pre-trained models and customizable parameters, it enables users to generate high-quality, pose-conditioned images with minimal setup.