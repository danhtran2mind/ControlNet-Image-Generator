# ControlNet Training Documentation

This document outlines the process for training a ControlNet model using the provided Python scripts (`train.py` and `train_controlnet.py`). The scripts facilitate training a ControlNet model integrated with a Stable Diffusion pipeline for conditional image generation. Below, we describe the training process and provide a detailed table of the command-line arguments used to configure the training.

## Overview

The training process involves two main scripts:
1. **`train.py`**: A wrapper script that executes `train_controlnet.py` with the provided command-line arguments.
2. **`train_controlnet.py`**: The core script that handles the training of the ControlNet model, including dataset preparation, model initialization, training loop, and validation.

### Training Workflow
1. **Argument Parsing**: The script parses command-line arguments to configure the training process, such as model paths, dataset details, and hyperparameters.
2. **Dataset Preparation**: Loads and preprocesses the dataset (either from HuggingFace Hub or a local directory) with transformations for images and captions.
3. **Model Initialization**: Loads pretrained models (e.g., Stable Diffusion, VAE, UNet, text encoder) and initializes or loads ControlNet weights.
4. **Training Loop**: Trains the ControlNet model using the Accelerate library for distributed training, with support for mixed precision, gradient checkpointing, and learning rate scheduling.
5. **Validation**: Periodically validates the model by generating images using validation prompts and images, logging results to TensorBoard or Weights & Biases.
6. **Checkpointing and Saving**: Saves model checkpoints during training and the final model to the output directory. Optionally pushes the model to the HuggingFace Hub.
7. **Model Card Creation**: Generates a model card with training details and example images for documentation.

## Command-Line Arguments

The following table describes the command-line arguments available in `train_controlnet.py` for configuring the training process:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pretrained_model_name_or_path` | `str` | None | Path to pretrained model or model identifier from huggingface.co/models. Required. |
| `--controlnet_model_name_or_path` | `str` | None | Path to pretrained ControlNet model or model identifier. If not specified, ControlNet weights are initialized from UNet. |
| `--revision` | `str` | None | Revision of pretrained model identifier from huggingface.co/models. |
| `--variant` | `str` | None | Variant of the model files (e.g., 'fp16'). |
| `--tokenizer_name` | `str` | None | Pretrained tokenizer name or path if different from model_name. |
| `--output_dir` | `str` | "controlnet-model" | Directory where model predictions and checkpoints are saved. |
| `--cache_dir` | `str` | None | Directory for storing downloaded models and datasets. |
| `--seed` | `int` | None | Seed for reproducible training. |
| `--resolution` | `int` | 512 | Resolution for input images (must be divisible by 8). |
| `--train_batch_size` | `int` | 4 | Batch size per device for the training dataloader. |
| `--num_train_epochs` | `int` | 1 | Number of training epochs. |
| `--max_train_steps` | `int` | None | Total number of training steps. Overrides `num_train_epochs` if provided. |
| `--checkpointing_steps` | `int` | 500 | Save a checkpoint every X updates. |
| `--checkpoints_total_limit` | `int` | None | Maximum number of checkpoints to store. |
| `--resume_from_checkpoint` | `str` | None | Resume training from a previous checkpoint path or "latest". |
| `--gradient_accumulation_steps` | `int` | 1 | Number of update steps to accumulate before a backward pass. |
| `--gradient_checkpointing` | `flag` | False | Enable gradient checkpointing to save memory at the cost of slower backward passes. |
| `--learning_rate` | `float` | 5e-6 | Initial learning rate after warmup. |
| `--scale_lr` | `flag` | False | Scale learning rate by number of GPUs, gradient accumulation steps, and batch size. |
| `--lr_scheduler` | `str` | "constant" | Learning rate scheduler type: ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]. |
| `--lr_warmup_steps` | `int` | 500 | Number of steps for learning rate warmup. |
| `--lr_num_cycles` | `int` | 1 | Number of hard resets for cosine_with_restarts scheduler. |
| `--lr_power` | `float` | 1.0 | Power factor for polynomial scheduler. |
| `--use_8bit_adam` | `flag` | False | Use 8-bit Adam optimizer from bitsandbytes for lower memory usage. |
| `--dataloader_num_workers` | `int` | 0 | Number of subprocesses for data loading (0 means main process). |
| `--adam_beta1` | `float` | 0.9 | Beta1 parameter for Adam optimizer. |
| `--adam_beta2` | `float` | 0.999 | Beta2 parameter for Adam optimizer. |
| `--adam_weight_decay` | `float` | 1e-2 | Weight decay for Adam optimizer. |
| `--adam_epsilon` | `float` | 1e-08 | Epsilon value for Adam optimizer. |
| `--max_grad_norm` | `float` | 1.0 | Maximum gradient norm for clipping. |
| `--push_to_hub` | `flag` | False | Push the model to the HuggingFace Hub. |
| `--hub_token` | `str` | None | Token for pushing to the HuggingFace Hub. |
| `--hub_model_id` | `str` | None | Repository name for syncing with `output_dir`. |
| `--logging_dir` | `str` | "logs" | TensorBoard log directory. |
| `--allow_tf32` | `flag` | False | Allow TF32 on Ampere GPUs for faster training. |
| `--report_to` | `str` | "tensorboard" | Integration for logging: ["tensorboard", "wandb", "comet_ml", "all"]. |
| `--mixed_precision` | `str` | None | Mixed precision training: ["no", "fp16", "bf16"]. |
| `--enable_xformers_memory_efficient_attention` | `flag` | False | Enable xformers for memory-efficient attention. |
| `--set_grads_to_none` | `flag` | False | Set gradients to None instead of zero to save memory. |
| `--dataset_name` | `str` | None | Name of the dataset from HuggingFace Hub or local path. |
| `--dataset_config_name` | `str` | None | Dataset configuration name. |
| `--train_data_dir` | `str` | None | Directory containing training data with `metadata.jsonl`. |
| `--image_column` | `str` | "image" | Dataset column for target images. |
| `--conditioning_image_column` | `str` | "conditioning_image" | Dataset column for ControlNet conditioning images. |
| `--caption_column` | `str` | "text" | Dataset column for captions. |
| `--max_train_samples` | `int` | None | Truncate training examples to this number for debugging or quicker training. |
| `--proportion_empty_prompts` | `float` | 0 | Proportion of prompts to replace with empty strings (0 to 1). |
| `--validation_prompt` | `str` | None | Prompts for validation, evaluated every `validation_steps`. |
| `--validation_image` | `str` | None | Paths to ControlNet conditioning images for validation. |
| `--num_validation_images` | `int` | 4 | Number of images generated per validation prompt-image pair. |
| `--validation_steps` | `int` | 100 | Run validation every X steps. |
| `--tracker_project_name` | `str` | "train_controlnet" | Project name for Accelerator trackers. |

## Usage Example

To train a ControlNet model, run the following command:

```bash
python src/controlnet_image_generator/train.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --dataset_name="huggingface/controlnet-dataset" \
  --output_dir="controlnet_output" \
  --resolution=512 \
  --train_batch_size=4 \
  --num_train_epochs=3 \
  --learning_rate=1e-5 \
  --validation_prompt="A cat sitting on a chair" \
  --validation_image="path/to/conditioning_image.png" \
  --push_to_hub \
  --hub_model_id="your-username/controlnet-model"
```

This command trains a ControlNet model using the Stable Diffusion 2.1 pretrained model, a specified dataset, and logs results to the HuggingFace Hub.

## Notes
- Ensure the dataset contains columns for target images, conditioning images, and captions as specified by `image_column`, `conditioning_image_column`, and `caption_column`.
- The resolution must be divisible by 8 to ensure compatibility with the VAE and ControlNet encoder.
- Mixed precision training (`fp16` or `bf16`) can reduce memory usage but requires compatible hardware.
- Validation images and prompts must be provided in matching quantities or as single values to be reused.

For further details, refer to the source scripts or the HuggingFace Diffusers documentation.