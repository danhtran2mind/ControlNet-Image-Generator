import torch
from controlnet_aux import OpenposeDetector
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    UniPCMultistepScheduler
)

def initialize_controlnet(config):
    model_id = config['model_id']
    local_dir = config.get('local_dir', model_id)
    return ControlNetModel.from_pretrained(
        local_dir if local_dir != model_id else model_id,
        torch_dtype=torch.float16
    )

def initialize_pipeline(controlnet, config):
    model_id = config['model_id']
    local_dir = config.get('local_dir', model_id)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        local_dir if local_dir != model_id else model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

def initialize_controlnet_detector(config):
    return OpenposeDetector.from_pretrained(config['model_id'])