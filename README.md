# ControlNet Image Generator 🖌️

[![GitHub Stars](https://img.shields.io/github/stars/danhtran2mind/ControlNet-Image-Generator?style=social&label=Repo%20Stars)](https://github.com/danhtran2mind/ControlNet-Image-Generator/stargazers)
![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fdanhtran2mind%2FControlNet-Image-Generator&label=Repo+Views&icon=github&color=%236f42c1&message=&style=social&tz=UTC)

[![huggingface-hub](https://img.shields.io/badge/huggingface--hub-blue.svg?logo=huggingface)](https://huggingface.co/docs/hub)
[![accelerate](https://img.shields.io/badge/accelerate-blue.svg?logo=pytorch)](https://huggingface.co/docs/accelerate)
[![bitsandbytes](https://img.shields.io/badge/bitsandbytes-blue.svg)](https://github.com/TimDettmers/bitsandbytes)
[![torch](https://img.shields.io/badge/torch-blue.svg?logo=pytorch)](https://pytorch.org/)
[![Pillow](https://img.shields.io/badge/Pillow-blue.svg)](https://pypi.org/project/pillow/)
[![numpy](https://img.shields.io/badge/numpy-blue.svg?logo=numpy)](https://numpy.org/)
[![transformers](https://img.shields.io/badge/transformers-blue.svg?logo=huggingface)](https://huggingface.co/docs/transformers)
[![torchvision](https://img.shields.io/badge/torchvision-blue.svg?logo=pytorch)](https://pytorch.org/vision/stable/index.html)
[![diffusers](https://img.shields.io/badge/diffusers-blue.svg?logo=huggingface)](https://huggingface.co/docs/diffusers)
[![peft](https://img.shields.io/badge/peft-blue.svg?logo=huggingface)](https://huggingface.co/docs/peft)
[![controlnet-aux](https://img.shields.io/badge/controlnet--aux-blue.svg)](https://github.com/patrickvonplaten/controlnet_aux)
[![gradio](https://img.shields.io/badge/gradio-blue.svg?logo=gradio)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Introduction 🌟

The **ControlNet Image Generator** is an open-source tool 🛠️ leveraging Stable Diffusion 2.1 and ControlNet with OpenPose to create high-quality, pose-guided images 📸. Perfect for researchers and developers 👩‍💻, it offers precise human pose conditioning, interactive demos, and flexible training/inference scripts under the MIT license 📜.

## Key Features 🚀

- **Pose-Guided Generation** 🕺: Uses OpenPose and ControlNet for accurate pose-conditioned images.
- **Stable Diffusion 2.1 Base** 🖼️: Built on Stability AI’s robust model for top-notch results.
- **Interactive Gradio Demo** 🎮: User-friendly interface hosted on Hugging Face Spaces.
- **Customizable Scripts** ⚙️: Supports training and inference with detailed options.
- **Notebook Compatibility** 📒: Works with Colab, SageMaker, Deepnote, JupyterLab, Gradient, Binder, and Kaggle.
- **Optimized Dependencies** 💻: Uses `diffusers`, `transformers`, `torch`, `gradio`, and more for efficiency.
- **Specialized Dataset** 📊: Trained on HighCWu/open_pose_controlnet_subset for pose accuracy.
- **Local & Cloud Support** ☁️: Run locally or explore on cloud platforms.
- **Comprehensive Docs** 📚: Guides for installation, training, and inference.
- **Open-Source** 🌍: MIT-licensed, community-driven via GitHub.

## Notebook

Explore the Openpose-guided ControlNet implementation in the following notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danhtran2mind/ControlNet-Image-Generator/blob/main/notebooks/SD-2.1-Openpose-ControlNet.ipynb)
[![Open in SageMaker](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/danhtran2mind/ControlNet-Image-Generator/blob/main/notebooks/SD-2.1-Openpose-ControlNet.ipynb)
[![Open in Deepnote](https://deepnote.com/buttons/launch-in-deepnote-small.svg)](https://deepnote.com/launch?url=https://github.com/danhtran2mind/ControlNet-Image-Generator/blob/main/notebooks/SD-2.1-Openpose-ControlNet.ipynb)
[![JupyterLab](https://img.shields.io/badge/Launch-JupyterLab-orange?logo=Jupyter)](https://mybinder.org/v2/gh/danhtran2mind/ControlNet-Image-Generator/main?filepath=notebooks/SD-2.1-Openpose-ControlNet.ipynb)
[![Open in Gradient](https://assets.paperspace.com/img/gradient-badge.svg)](https://console.paperspace.com/github/danhtran2mind/ControlNet-Image-Generator/blob/main/notebooks/SD-2.1-Openpose-ControlNet.ipynb)
[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/danhtran2mind/ControlNet-Image-Generator/main)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/notebooks/welcome?src=https%3A%2F%2Fgithub.com%2Fdanhtran2mind%2FControlNet-Image-Generator/blob/main/notebooks/SD-2.1-Openpose-ControlNet.ipynb)
[![View on GitHub](https://img.shields.io/badge/View%20on-GitHub-181717?logo=github)](https://github.com/danhtran2mind/ControlNet-Image-Generator/blob/main/notebooks/SD-2.1-Openpose-ControlNet.ipynb)


## Dataset
The Stable-Diffusion-2.1-Openpose-ControlNet model was trained on the [![HuggingFace Model Hub](https://img.shields.io/badge/HuggingFace-HighCWu%2Fopen_pose_controlnet_subset-yellow?style=flat&logo=huggingface)](https://huggingface.co/datasets/HighCWu/open_pose_controlnet_subset) dataset, available at Hugging Face. This dataset provides specialized data for fine-tuning, enabling precise human pose conditioning using OpenPose for enhanced image generation.

## Base Model
The Stable-Diffusion-2.1-Openpose-ControlNet is built upon the [![HuggingFace Model Hub](https://img.shields.io/badge/HuggingFace-stabilityai%2Fstable--diffusion--2--1-yellow?style=flat&logo=huggingface)](https://huggingface.co/stabilityai/stable-diffusion-2-1) base model. This foundation model, developed by Stability AI, serves as the core architecture, which is further refined with ControlNet to improve control and accuracy in generating pose-conditioned images.



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
### Script-Driven Setup

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

For more information about Scripts, you can see [Download Model Checkpoint Document](docs/scripts/download_ckpts_doc.md) and [Download Dataset Document](docs/scripts/download_datasets_doc.md). ⚙️

## Demonstration

### Interactive Demo

Explore the interactive demo hosted on HuggingFace:
[![HuggingFace Space Demo](https://img.shields.io/badge/HuggingFace-danhtran2mind%2FAnime--Super--Resolution-yellow?style=flat&logo=huggingface)](https://huggingface.co/spaces/danhtran2mind/Anime-Super-Resolution).

Below is a screenshot of the SlimFace Demo GUI:

<img src="./assets/gradio_app_demo.jpg" alt="Image Demo" height="600">

### Run Locally

To run the Gradio application locally at the default address `localhost:7860`, execute:

```bash
python apps/gradio_app.py
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
#### Training Document
For more information about Training, you can see [Training Document](docs/training/training_doc.md). ⚙️

### Inference
#### Inference Script

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
#### Inference Document
Refer to the [Inference Documents](docs/inference/inference_doc.md) for detailed arguments used in Inference. ⚙️

#### Inference Examples
Here are several examples showcasing the capabilities of the`Stable-Diffusion-2.1-Openpose-ControlNet` model
|Prompt|Input Image|Output Image|
|:----:|:----:|:----:|
|A man is doing yoga in a serene park.|<img src="apps/gradio_app/assets/examples/Stable-Diffusion-2.1-Openpose-ControlNet/1/yoga.jpg" alt="Image Demo" height="400">|<img src="apps/gradio_app/assets/examples/Stable-Diffusion-2.1-Openpose-ControlNet/1/a_man_is_doing_yoga_in_a_serene_park_0.png" alt="Image Demo" height="400">|
|A man is galloping on a horse.|<img src="apps/gradio_app/assets/examples/Stable-Diffusion-2.1-Openpose-ControlNet/2/ride_bike.jpg" alt="Image Demo" height="400">|<img src="apps/gradio_app/assets/examples/Stable-Diffusion-2.1-Openpose-ControlNet/2/a_man_is_galloping_on_a_horse_0.png" alt="Image Demo" height="400">|
|A woman is holding a baseball bat in her hand.|<img src="apps/gradio_app/assets/examples/Stable-Diffusion-2.1-Openpose-ControlNet/3/tennis.jpg" alt="Image Demo" height="400">|<img src="apps/gradio_app/assets/examples/Stable-Diffusion-2.1-Openpose-ControlNet/3/a_woman_is_holding_a_baseball_bat_in_her_hand_0.png" alt="Image Demo" height="400">|
|A woman raises a katana.|<img src="apps/gradio_app/assets/examples/Stable-Diffusion-2.1-Openpose-ControlNet/4/man_and_sword.jpg" alt="Image Demo" height="400">|<img src="apps/gradio_app/assets/examples/Stable-Diffusion-2.1-Openpose-ControlNet/4/a_woman_raises_a_katana_0.png" alt="Image Demo" height="400">|

## Environment

SlimFace requires the following environment:

- **Python**: 3.10 or higher
- **Key Libraries**: Refer to [Requirements Compatible](./requirements/requirements_compatible.txt) for compatible dependencies.


## Project Credits and Resources

- This project builds upon code from the Diffusers Project by Hugging Face: [![GitHub](https://img.shields.io/badge/GitHub-huggingface%2Fdiffusers-blue?style=flat&logo=github)](https://github.com/huggingface/diffusers).
- Explore additional Model Hubs at: Hugging Face Model Hub: [![HuggingFace Model Hub](https://img.shields.io/badge/HuggingFace-danhtran2mind%2FStable--Diffusion--2.1--Openpose--ControlNet-yellow?style=flat&logo=huggingface)](https://huggingface.co/danhtran2mind/Stable-Diffusion-2.1-Openpose-ControlNet).
