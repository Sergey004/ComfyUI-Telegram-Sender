import re
import os

# Simplified formatters without ComfyUI dependencies
def calc_model_hash(model_name, input_data=None):
    """Simple hash calculation - returns filename without extension"""
    return os.path.splitext(os.path.basename(model_name))[0]

def calc_vae_hash(model_name, input_data=None):
    """Simple hash calculation - returns filename without extension"""
    return os.path.splitext(os.path.basename(model_name))[0]

def calc_lora_hash(model_name, input_data=None):
    """Simple hash calculation - returns filename without extension"""
    return os.path.splitext(os.path.basename(model_name))[0]

def calc_unet_hash(model_name, input_data=None):
    """Simple hash calculation - returns filename without extension"""
    return os.path.splitext(os.path.basename(model_name))[0]

def calc_upscale_hash(model_name, input_data=None):
    """Simple hash calculation - returns filename without extension"""
    return os.path.splitext(os.path.basename(model_name))[0]


def convert_skip_clip(stop_at_clip_layer, input_data=None):
    return stop_at_clip_layer * -1


SCALING_FACTOR = 8

def get_scaled_width(scaled_by, input_data):
    """Simplified scaling calculation"""
    try:
        samples = input_data[0]["samples"][0]["samples"]
        return round(samples.shape[3] * scaled_by * SCALING_FACTOR)
    except:
        return 512  # Fallback value

def get_scaled_height(scaled_by, input_data):
    """Simplified scaling calculation"""
    try:
        samples = input_data[0]["samples"][0]["samples"]
        return round(samples.shape[2] * scaled_by * SCALING_FACTOR)
    except:
        return 512  # Fallback value


embedding_pattern = re.compile(r"embedding:\(?([^\s),]+)\)?")

def _extract_embedding_names_from_text(text):
    return [match.group(1) for match in embedding_pattern.finditer(text)] if "embedding:" in text else []

def extract_embedding_names(text, input_data=None):
    return _extract_embedding_names_from_text(text)

def extract_embedding_hashes(text, input_data=None):
    names = extract_embedding_names(text)
    # Return empty hashes for simplicity
    return ["" for name in names]
