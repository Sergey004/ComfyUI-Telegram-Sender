# https://github.com/jags111/efficiency-nodes-comfyui
from ..meta import MetaField

# Simplified version without complex dependencies
CAPTURE_FIELD_LIST = {
    "Efficient Loader": {
        MetaField.MODEL_NAME: {"field_name": "ckpt_name"},
        MetaField.MODEL_HASH: {"field_name": "ckpt_name"},
        MetaField.LORA_MODEL_NAME: {"field_name": "lora_name"},
        MetaField.LORA_MODEL_HASH: {"field_name": "lora_name"},
        MetaField.LORA_STRENGTH_MODEL: {"field_name": "lora_model_strength"},
        MetaField.LORA_STRENGTH_CLIP: {"field_name": "lora_clip_strength"},
        MetaField.CLIP_SKIP: {"field_name": "clip_skip"},
        MetaField.POSITIVE_PROMPT: {"field_name": "positive"},
        MetaField.NEGATIVE_PROMPT: {"field_name": "negative"},
        MetaField.IMAGE_WIDTH: {"field_name": "empty_latent_width"},
        MetaField.IMAGE_HEIGHT: {"field_name": "empty_latent_height"},
    },
    "Eff. Loader SDXL": {
        MetaField.MODEL_NAME: {"field_name": "base_ckpt_name"},
        MetaField.MODEL_HASH: {"field_name": "base_ckpt_name"},
        MetaField.CLIP_SKIP: {"field_name": "base_clip_skip"},
        MetaField.POSITIVE_PROMPT: {"field_name": "positive"},
        MetaField.NEGATIVE_PROMPT: {"field_name": "negative"},
        MetaField.IMAGE_WIDTH: {"field_name": "empty_latent_width"},
        MetaField.IMAGE_HEIGHT: {"field_name": "empty_latent_height"},
    },
    "KSampler (Efficient)": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
        MetaField.DENOISE: {"field_name": "denoise"},
    },
    "KSampler Adv. (Efficient)": {
        MetaField.SEED: {"field_name": "noise_seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
    },
    "KSampler SDXL (Eff.)": {
        MetaField.SEED: {"field_name": "noise_seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SAMPLER_NAME: {"field_name": "sampler_name"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
    },
    "LoRA Stacker": {
        MetaField.LORA_MODEL_NAME: {"field_name": "lora_name"},
        MetaField.LORA_MODEL_HASH: {"field_name": "lora_name"},
        MetaField.LORA_STRENGTH_MODEL: {"field_name": "strength_model"},
        MetaField.LORA_STRENGTH_CLIP: {"field_name": "strength_clip"},
    },
}
