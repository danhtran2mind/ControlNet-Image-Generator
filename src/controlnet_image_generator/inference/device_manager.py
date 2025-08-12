import torch

def setup_device(pipe):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    pipe.to(device)
    return device