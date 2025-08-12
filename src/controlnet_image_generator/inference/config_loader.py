import yaml

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")
    
def find_config_by_model_id(configs, model_id):
    for config in configs:
        if config['model_id'] == model_id:
            return config
    raise ValueError(f"No configuration found for model_id: {model_id}")