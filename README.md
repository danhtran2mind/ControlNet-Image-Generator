# ControlNet Image Generator 🖌️


## Introduction 🌟


## Key Features ✨


## Notebook

## Dataset
## Demonstration

### Interactive Demo

Explore the interactive demo hosted on HuggingFace:
[![HuggingFace Space Demo](https://img.shields.io/badge/HuggingFace-danhtran2mind%2FAnime--Super--Resolution-yellow?style=flat&logo=huggingface)](https://huggingface.co/spaces/danhtran2mind/Anime-Super-Resolution)

Below is a screenshot of the SlimFace Demo GUI:

<img src="./assets/gradio_app_demo.jpg" alt="SlimFace Demo" height="600">

### Run Locally

To run the Gradio application locally at the default address `localhost:7860`, execute:

```bash
python apps/gradio_app.py
```


## Installation
### Clone GitHub Repository
```bash
git clone https://github.com/danhtran2mind/ControlNet-Image-Generator.git
cd ControlNet-Image-Generator
```
### Install Dependencies
```bash
pip install -r requirements/requirements.txt
```
### EScript-Driven Setup

#### Download Model Checkpoints
```bash
python scripts/download_ckpts.py
```
#### Download Datasets

```bash
python scripts/download_datasets.py
```

#### Setup Third Party (Diffusers for ControlNet Training)
```bash
python scripts/setup_third_party.py
```
## Usage
### Training
#### Training Script
```bash
accelerate launch src/controlnet_image_generator/train.py \
    --pretrained_model_name_or_path="<model_name_or_path>" \
    --output_dir="<output_path>" \
    --dataset_name="<dataset_name>" \
    --resolution=<image_resolution> \
    --learning_rate=<lr_value> \
    --train_batch_size=<batch_size> \
    --gradient_accumulation_steps=<grad_steps> \
    --gradient_checkpointing \
    --use_8bit_adam \
    --num_train_epochs=<num_epochs> \
    --mixed_precision "<precision_type>" \
    --checkpoints_total_limit=<num_limit_ckpts> \
    --checkpointing_steps=<num_checkpoint_step> \
    --validation_steps=<num_step>
```

### Inference
#### Inference Script
```bash
python src/controlnet_image_generator/infer.py
```

```bash
python src/controlnet_image_generator/infer.py \
    --config_path "<config_file_path>" \
    --input_image "<input_image_path>" \
    --prompt "<text_prompt>" \
    --negative_prompt "<negative_text_prompt>" \
    --num_steps <inference_steps> \
    --seed <random_seed> \
    --width <image_width> \
    --height <image_height> \
    --guidance_scale <guidance_value> \
    --controlnet_conditioning_scale <controlnet_scale> \
    --save_output \
    --output_dir "<output_directory>" \
    --use_prompt_as_output_name
```
#### Inference Examples





## Environment

SlimFace requires the following environment:

- **Python**: 3.10 or higher
- **Key Libraries**: Refer to [Requirements Compatible](./requirements/requirements_compatible.txt) for compatible dependencies.


## Project Credits and Resources

- This project leverages code from:

    > The Original Real-ESRGAN by [![GitHub](https://img.shields.io/badge/GitHub-xinntao-blue?style=flat&logo=github)](https://github.com/xinntao) at [![Built on Real-ESRGAN](https://img.shields.io/badge/Built%20on-xinntao%2FReal--ESRGAN-blue?style=flat&logo=github)](https://github.com/xinntao/Real-ESRGAN). Our own bug fixes and enhancements are available at [![Real-ESRGAN Enhancements](https://img.shields.io/badge/GitHub-danhtran2mind%2FReal--ESRGAN-blue?style=flat&logo=github)](https://github.com/danhtran2mind/Real-ESRGAN).

    > The Inference code by 
    [![GitHub](https://img.shields.io/badge/GitHub-ai--forever-blue?style=flat&logo=github)](https://github.com/ai-forever) at [![Built on Real-ESRGAN](https://img.shields.io/badge/Built%20on-ai--forever%2FReal--ESRGAN-blue?style=flat&logo=github)](https://github.com/ai-forever/Real-ESRGAN).
    Our own bug fixes and enhancements are available at [![Real-ESRGAN Enhancements](https://img.shields.io/badge/GitHub-danhtran2mind%2FReal--ESRGAN--inference-blue?style=flat&logo=github)](https://github.com/danhtran2mind/Real-ESRGAN-inference)

- You can explore more Model Hubs at:

    > HuggingFace Model Hub: [![ai-forever Real-ESRGAN Model](https://img.shields.io/badge/HuggingFace-ai--forever%2FReal--ESRGAN-yellow?style=flat&logo=huggingface)](https://huggingface.co/ai-forever/Real-ESRGAN). Real-ESRGAN Model releases: [![Real-ESRGAN releases](https://img.shields.io/badge/GitHub-Real--ESRGAN%2Freleases-blue?style=flat&logo=github)](https://github.com/xinntao/Real-ESRGAN/releases). 
    > You also download `Real-ESRGAN-Anime-finetuning` at [![Real-ESRGAN-Anime-finetuning Model](https://img.shields.io/badge/HuggingFace-danhtran2mind%2FReal--ESRGAN--Anime--finetuning-yellow?style=flat&logo=huggingface)](https://huggingface.co/danhtran2mind/Real-ESRGAN-Anime-finetuning)


<!-- https://github.com/ai-forever/Real-ESRGAN https://github.com/danhtran2mind/Real-ESRGAN-inference
https://huggingface.co/ai-forever/Real-ESRGAN
https://github.com/xinntao/Real-ESRGAN https://github.com/danhtran2mind/Real-ESRGAN 
https://github.com/xinntao/Real-ESRGAN/releases -->