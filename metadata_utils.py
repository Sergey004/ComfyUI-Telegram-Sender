import os
import re
import json

# Import modular capture system
try:
    from .modules.capture import Capture
except ImportError:
    # Fallback for direct execution
    from modules.capture import Capture


class MetadataUtils:
    """
    Utility class for extracting metadata from ComfyUI workflows using modular capture system
    """
    
    @staticmethod
    def extract_metadata_from_workflow(prompt, extra_pnginfo=None):
        """
        Extract comprehensive metadata from workflow using modular capture system
        
        Returns: dict containing metadata fields
        """
        if not prompt:
            return {}
        
        pnginfo_dict = {}
        
        try:
            # Extract inputs using modular capture system
            inputs = Capture.get_inputs(prompt, extra_pnginfo)
            
            # Generate PNG info dictionary
            # For simplicity, we'll use inputs_before_sampler_node = inputs
            # In a more sophisticated implementation, we'd trace the workflow
            pnginfo_dict = Capture.gen_pnginfo_dict(inputs, inputs, prompt)
            
        except Exception as e:
            print(f"[MetadataUtils] Error extracting metadata: {e}")
        
        return pnginfo_dict
    
    @staticmethod
    def _extract_positive_prompt(prompt):
        """Extract positive prompt from workflow"""
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})
            
            if "clip" in class_type.lower() and "text" in inputs:
                text = inputs.get("text", "")
                if text and len(text.strip()) > 0:
                    return text.strip()
        
        return ""
    
    @staticmethod
    def _extract_negative_prompt(prompt):
        """Extract negative prompt from workflow"""
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})
            
            # Look for negative prompt specifically
            if "negative" in class_type.lower() and "text" in inputs:
                text = inputs.get("text", "")
                if text and len(text.strip()) > 0:
                    return text.strip()
        
        return ""
    
    @staticmethod
    def _extract_sampling_parameters(prompt):
        """Extract sampling parameters from workflow"""
        params = {}
        
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "")
            
            if "sampler" in class_type.lower() or "ksampler" in class_type.lower():
                inputs = node_data.get("inputs", {})
                
                seed = inputs.get("seed")
                steps = inputs.get("steps")
                cfg = inputs.get("cfg")
                sampler_name = inputs.get("sampler_name")
                scheduler = inputs.get("scheduler")
                denoise = inputs.get("denoise")
                
                if steps:
                    params["Steps"] = str(steps)
                if sampler_name:
                    params["Sampler"] = str(sampler_name)
                if scheduler:
                    params["Schedule type"] = str(scheduler)
                if cfg:
                    params["CFG scale"] = str(cfg)
                if seed is not None:
                    params["Seed"] = str(seed)
                if denoise is not None and denoise != 1.0:
                    params["Denoising strength"] = str(denoise)
                
                break  # Use first sampler found
        
        return params
    
    @staticmethod
    def _extract_model_info(prompt):
        """Extract model information from workflow"""
        model_info = {}
        
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})
            
            if "checkpoint" in class_type.lower():
                model_name = inputs.get("ckpt_name")
                if model_name:
                    model_info["Model"] = os.path.splitext(os.path.basename(model_name))[0]
            
            if "vae" in class_type.lower():
                vae_name = inputs.get("vae_name")
                if vae_name:
                    model_info["VAE"] = os.path.splitext(os.path.basename(vae_name))[0]
        
        return model_info
    
    @staticmethod
    def _extract_lora_info(prompt):
        """Extract LoRA information from workflow"""
        lora_info = {}
        loras = []
        
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "")
            
            if "lora" in class_type.lower():
                inputs = node_data.get("inputs", {})
                lora_name = inputs.get("lora_name")
                strength = inputs.get("strength_model", inputs.get("strength", 1.0))
                
                if lora_name:
                    clean_name = os.path.splitext(os.path.basename(lora_name))[0]
                    loras.append(f"{clean_name}: {strength}")
        
        if loras:
            lora_info["Lora hashes"] = f'"{", ".join(loras)}"'
        
        return lora_info
    
    @staticmethod
    def _extract_size_info(prompt):
        """Extract image size information from workflow"""
        size_info = {}
        
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})
            
            if "empty" in class_type.lower() and "latent" not in class_type.lower():
                width = inputs.get("width")
                height = inputs.get("height")
                
                if width and height:
                    size_info["Size"] = f"{width}x{height}"
                    break
        
        return size_info
    
    @staticmethod
    def build_a1111_style_metadata(pnginfo_dict):
        """
        Build A1111-style metadata string from PNG info dictionary
        """
        return Capture.gen_parameters_str(pnginfo_dict)
    
    @staticmethod
    def extract_loras_from_metadata(pnginfo_dict):
        """
        Extract LoRA information from metadata dictionary
        """
        loras = []
        
        if not pnginfo_dict:
            return loras
        
        # Extract LoRA hashes value - handle complex types safely
        lora_hashes_value = pnginfo_dict.get("Lora hashes", "")
        lora_hashes_str = ""
        
        def safe_convert_to_string(value):
            """Safely convert any value to string representation"""
            if value is None:
                return ""
            
            # Handle string types
            if isinstance(value, str):
                return value.strip()
            
            # Handle numeric types
            if isinstance(value, (int, float)):
                return str(value)
            
            # Handle list types with proper flattening
            if isinstance(value, list):
                try:
                    # Flatten nested lists recursively
                    def flatten(items):
                        flattened = []
                        for item in items:
                            if isinstance(item, list):
                                flattened.extend(flatten(item))
                            else:
                                # Convert each item to string safely with comprehensive handling
                                try:
                                    if item is None:
                                        continue
                                    elif isinstance(item, str):
                                        flattened.append(item.strip())
                                    elif isinstance(item, (int, float)):
                                        flattened.append(str(item))
                                    elif isinstance(item, dict):
                                        # Handle dict items specially
                                        dict_str = safe_convert_to_string(item)
                                        if dict_str:
                                            flattened.append(dict_str)
                                    else:
                                        # Use repr for other complex types
                                        flattened.append(repr(item))
                                except Exception as e:
                                    # If any item conversion fails, skip it and log the error
                                    print(f"[safe_convert_to_string] ⚠️ Error converting item {item}: {e}")
                                    continue
                        return flattened
                    
                    flattened_items = flatten(value)
                    # Join with commas, but avoid empty strings
                    non_empty_items = [item for item in flattened_items if item]
                    return ", ".join(non_empty_items) if non_empty_items else ""
                except Exception as e:
                    # Fallback: use repr for complex structures
                    print(f"[safe_convert_to_string] ⚠️ Error processing list {value}: {e}")
                    return repr(value)
            
            # Handle dictionary and other complex types
            if isinstance(value, dict):
                try:
                    # Convert dict to string representation
                    items = []
                    for k, v in value.items():
                        key_str = safe_convert_to_string(k)
                        val_str = safe_convert_to_string(v)
                        if key_str and val_str:
                            items.append(f"{key_str}: {val_str}")
                    return ", ".join(items) if items else ""
                except Exception:
                    return repr(value)
            
            # Handle other types with repr as fallback
            try:
                return str(value)
            except Exception:
                return repr(value)
        
        # Convert lora_hashes_value to string safely
        lora_hashes_str = safe_convert_to_string(lora_hashes_value)
        
        if lora_hashes_str:
            # Parse format: "Lora_Name_1: hash1, Lora_Name_2: hash2"
            lora_entries = lora_hashes_str.replace('"', '').split(', ')
            for entry in lora_entries:
                if ':' in entry:
                    try:
                        lora_name, lora_hash = entry.split(':', 1)
                        loras.append({
                            'name': lora_name.strip(),
                            'hash': lora_hash.strip()
                        })
                    except Exception as e:
                        print(f"[MetadataUtils] ⚠️ Error parsing LoRA entry '{entry}': {e}")
                        continue
        
        # Also check for LoRA strings in positive prompt
        positive_prompt = pnginfo_dict.get("Positive prompt", "")
        if positive_prompt:
            # Ensure positive_prompt is a string before processing
            positive_prompt_str = safe_convert_to_string(positive_prompt)
            if positive_prompt_str:
                # Regex to match <lora:name:weight> or <lyco:name:weight>
                lora_pattern = re.compile(r"<(lora|lyco):([a-zA-Z0-9_\./\\-]+):([0-9.]+)>")
                matches = lora_pattern.findall(positive_prompt_str)
                
                for tag, name, weight in matches:
                    try:
                        loras.append({
                            'name': name.strip(),
                            'weight': float(weight),
                            'type': tag
                        })
                    except Exception as e:
                        print(f"[MetadataUtils] ⚠️ Error parsing LoRA prompt entry '{name}': {e}")
                        continue
        
        return loras
    
    @staticmethod
    def extract_model_info(pnginfo_dict):
        """
        Extract model information from metadata dictionary
        """
        model_info = {}
        
        if not pnginfo_dict:
            return model_info
        
        model_info['name'] = pnginfo_dict.get("Model", "")
        model_info['vae'] = pnginfo_dict.get("VAE", "")
        
        return model_info
    
    @staticmethod
    def extract_sampling_params(pnginfo_dict):
        """
        Extract sampling parameters from metadata dictionary
        """
        params = {}
        
        if not pnginfo_dict:
            return params
        
        params['seed'] = pnginfo_dict.get("Seed", "")
        params['steps'] = pnginfo_dict.get("Steps", "")
        params['cfg_scale'] = pnginfo_dict.get("CFG scale", "")
        params['sampler'] = pnginfo_dict.get("Sampler", "")
        params['denoising_strength'] = pnginfo_dict.get("Denoising strength", "")
        
        # Extract size
        size_str = pnginfo_dict.get("Size", "")
        if size_str and 'x' in size_str:
            width, height = size_str.split('x', 1)
            params['width'] = width.strip()
            params['height'] = height.strip()
        
        return params
    
    @staticmethod
    def format_metadata_for_telegram(pnginfo_dict, include_prompts=True, include_params=True):
        """
        Format metadata for Telegram message display
        """
        if not pnginfo_dict:
            return ""
        
        lines = []
        
        if include_prompts:
            positive_prompt = pnginfo_dict.get("Positive prompt", "")
            negative_prompt = pnginfo_dict.get("Negative prompt", "")
            
            if positive_prompt:
                lines.append(f"📝 Positive: {positive_prompt}")
            
            if negative_prompt:
                lines.append(f"🚫 Negative: {negative_prompt}")
        
        if include_params:
            model_info = MetadataUtils.extract_model_info(pnginfo_dict)
            sampling_params = MetadataUtils.extract_sampling_params(pnginfo_dict)
            loras = MetadataUtils.extract_loras_from_metadata(pnginfo_dict)
            
            if model_info.get('name'):
                lines.append(f"🤖 Model: {model_info['name']}")
            
            if sampling_params.get('seed'):
                lines.append(f"🌱 Seed: {sampling_params['seed']}")
            
            if sampling_params.get('steps'):
                lines.append(f"⚡ Steps: {sampling_params['steps']}")
            
            if sampling_params.get('cfg_scale'):
                lines.append(f"🎛️ CFG Scale: {sampling_params['cfg_scale']}")
            
            if sampling_params.get('sampler'):
                lines.append(f"🌀 Sampler: {sampling_params['sampler']}")
            
            if loras:
                lora_names = [lora.get('name', '') for lora in loras if lora.get('name')]
                if lora_names:
                    # Ensure all lora names are strings before joining
                    lora_names_str = [str(name) for name in lora_names]
                    lines.append(f"🎨 LoRAs: {', '.join(lora_names_str)}")
        
        return "\n".join(lines) if lines else ""


# Convenience functions for direct use
def extract_metadata(prompt, extra_pnginfo=None):
    """Extract metadata from workflow"""
    return MetadataUtils.extract_metadata_from_workflow(prompt, extra_pnginfo)

def build_metadata_text(pnginfo_dict):
    """Build A1111-style metadata text"""
    return MetadataUtils.build_a1111_style_metadata(pnginfo_dict)

def format_telegram_metadata(pnginfo_dict):
    """Format metadata for Telegram display"""
    return MetadataUtils.format_metadata_for_telegram(pnginfo_dict)
