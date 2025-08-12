from PIL import Image
from utils.download import load_image

def load_input_image(input_image_path=None, image_url=None):
    try:
        if input_image_path:
            return Image.open(input_image_path).convert("RGB")
        elif image_url:
            return load_image(image_url)
        else:
            raise ValueError("Either input_image or image_url must be provided")
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

def detect_poses(controlnet_detector, image):
    return [controlnet_detector(image)]