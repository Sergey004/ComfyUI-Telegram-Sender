import os
from .defs.captures import CAPTURE_FIELD_LIST
from .defs.meta import MetaField

# Import third-party node definitions
try:
    from .defs.ext.efficiency_nodes import CAPTURE_FIELD_LIST as EFFICIENCY_NODES_CAPTURE
    # Combine with main capture list
    CAPTURE_FIELD_LIST.update(EFFICIENCY_NODES_CAPTURE)
except ImportError:
    pass  # Efficiency nodes extension not available

# Import other extensions
try:
    from .defs.ext.easyuse_nodes import CAPTURE_FIELD_LIST as EASYUSE_NODES_CAPTURE
    CAPTURE_FIELD_LIST.update(EASYUSE_NODES_CAPTURE)
except ImportError:
    pass  # EasyUse nodes extension not available


class Capture:
    """Main capture class for extracting metadata from workflows"""
    
    @classmethod
    def get_inputs(cls, prompt, extra_data=None):
        """Extract inputs from workflow using node-specific definitions"""
        inputs = {}
        
        if not prompt:
            return inputs
        
        # Track prompt nodes to distinguish positive/negative
        prompt_nodes = []
        
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "")
            node_inputs = node_data.get("inputs", {})
            
            # Check if this node type has capture definitions
            if class_type in CAPTURE_FIELD_LIST:
                node_definitions = CAPTURE_FIELD_LIST[class_type]
                
                for meta_field, definition in node_definitions.items():
                    field_name = definition.get("field_name")
                    
                    if field_name in node_inputs:
                        value = node_inputs[field_name]
                        if value is not None:
                            # Initialize list for meta field if not exists
                            if meta_field not in inputs:
                                inputs[meta_field] = []
                            
                            # Add the value
                            inputs[meta_field].append((node_id, value))
            
            # Track CLIPTextEncode nodes for prompt distinction
            if class_type == "CLIPTextEncode":
                prompt_nodes.append((node_id, node_inputs.get("text", "")))
        
        # If we have multiple CLIPTextEncode nodes, try to distinguish them
        if len(prompt_nodes) >= 2:
            # Simple heuristic: shorter prompt = positive, longer = negative
            sorted_prompts = sorted(prompt_nodes, key=lambda x: len(str(x[1])))
            # Clear existing prompt entries and add properly distinguished ones
            if MetaField.POSITIVE_PROMPT in inputs:
                inputs[MetaField.POSITIVE_PROMPT] = [(sorted_prompts[0][0], sorted_prompts[0][1])]
            if MetaField.NEGATIVE_PROMPT in inputs:
                inputs[MetaField.NEGATIVE_PROMPT] = [(sorted_prompts[1][0], sorted_prompts[1][1])]
        
        return inputs
    
    @classmethod
    def gen_pnginfo_dict(cls, inputs_before_sampler_node, inputs_before_this_node, prompt):
        """Generate PNG info dictionary from captured inputs"""
        pnginfo = {}
        
        if not inputs_before_sampler_node:
            return pnginfo
        
        # Extract values from inputs
        def extract_value(inputs, meta_field, default_key=None):
            items = inputs.get(meta_field, [])
            if items:
                # Use the first value found
                return items[0][1]
            return None
        
        # Prompts - need to distinguish between positive and negative
        # Get all prompt values and try to determine which is which
        positive_items = inputs_before_sampler_node.get(MetaField.POSITIVE_PROMPT, [])
        negative_items = inputs_before_sampler_node.get(MetaField.NEGATIVE_PROMPT, [])
        
        # If we have specific positive/negative distinction, use it
        if positive_items:
            pnginfo["Positive prompt"] = positive_items[0][1]
        
        if negative_items:
            pnginfo["Negative prompt"] = negative_items[0][1]
        
        # Fallback: if no distinction, try to infer from content
        if not pnginfo.get("Positive prompt") and not pnginfo.get("Negative prompt"):
            all_prompt_items = []
            if MetaField.POSITIVE_PROMPT in inputs_before_sampler_node:
                all_prompt_items.extend(inputs_before_sampler_node[MetaField.POSITIVE_PROMPT])
            if MetaField.NEGATIVE_PROMPT in inputs_before_sampler_node:
                all_prompt_items.extend(inputs_before_sampler_node[MetaField.NEGATIVE_PROMPT])
            
            if len(all_prompt_items) >= 2:
                # Simple heuristic: shorter prompt = positive, longer = negative
                sorted_prompts = sorted(all_prompt_items, key=lambda x: len(str(x[1])))
                pnginfo["Positive prompt"] = sorted_prompts[0][1]
                pnginfo["Negative prompt"] = sorted_prompts[1][1]
            elif len(all_prompt_items) == 1:
                pnginfo["Positive prompt"] = all_prompt_items[0][1]
        
        # Sampling parameters
        steps = extract_value(inputs_before_sampler_node, MetaField.STEPS)
        seed = extract_value(inputs_before_sampler_node, MetaField.SEED)
        cfg = extract_value(inputs_before_sampler_node, MetaField.CFG)
        sampler_name = extract_value(inputs_before_sampler_node, MetaField.SAMPLER_NAME)
        scheduler = extract_value(inputs_before_sampler_node, MetaField.SCHEDULER)
        denoise = extract_value(inputs_before_sampler_node, MetaField.DENOISE)
        
        if steps:
            pnginfo["Steps"] = str(steps)
        if seed is not None:
            pnginfo["Seed"] = str(seed)
        if cfg:
            pnginfo["CFG scale"] = str(cfg)
        if sampler_name:
            pnginfo["Sampler"] = str(sampler_name)
        if scheduler:
            pnginfo["Scheduler"] = str(scheduler)
        if denoise is not None and denoise != 1.0:
            pnginfo["Denoising strength"] = str(denoise)
        
        # Model information
        model_name = extract_value(inputs_before_sampler_node, MetaField.MODEL_NAME)
        vae_name = extract_value(inputs_before_this_node, MetaField.VAE_NAME)
        
        if model_name:
            pnginfo["Model"] = os.path.splitext(os.path.basename(model_name))[0]
        if vae_name:
            pnginfo["VAE"] = os.path.splitext(os.path.basename(vae_name))[0]
        
        # LoRA information
        lora_names = inputs_before_sampler_node.get(MetaField.LORA_MODEL_NAME, [])
        lora_strengths = inputs_before_sampler_node.get(MetaField.LORA_STRENGTH_MODEL, [])

        loras = []
        for i in range(min(len(lora_names), len(lora_strengths))):
            lora_name_tuple = lora_names[i]
            lora_strength_tuple = lora_strengths[i]
            
            # Extract values from tuples (node_id, value)
            lora_name_value = lora_name_tuple[1] if isinstance(lora_name_tuple, tuple) and len(lora_name_tuple) > 1 else lora_name_tuple
            lora_strength_value = lora_strength_tuple[1] if isinstance(lora_strength_tuple, tuple) and len(lora_strength_tuple) > 1 else lora_strength_tuple
            
            # Ensure both values are strings before processing
            lora_name_str = str(lora_name_value) if lora_name_value is not None else ""
            lora_strength_str = str(lora_strength_value) if lora_strength_value is not None else "1.0"
            
            if lora_name_str:
                clean_name = os.path.splitext(os.path.basename(lora_name_str))[0]
                loras.append(f"{clean_name}: {lora_strength_str}")

        if loras:
            pnginfo["Lora hashes"] = f'"{", ".join(loras)}"'
        
        # Size information
        width = extract_value(inputs_before_sampler_node, MetaField.IMAGE_WIDTH)
        height = extract_value(inputs_before_sampler_node, MetaField.IMAGE_HEIGHT)
        
        if width and height:
            pnginfo["Size"] = f"{width}x{height}"
        
        return pnginfo
    
    @classmethod
    def gen_parameters_str(cls, pnginfo_dict):
        """Generate A1111-style parameters string"""
        if not pnginfo_dict:
            return ""
        
        parts = []
        
        # Add positive prompt
        positive_prompt = pnginfo_dict.get("Positive prompt", "")
        if positive_prompt:
            # Ensure positive_prompt is a string
            positive_prompt_str = str(positive_prompt) if not isinstance(positive_prompt, str) else positive_prompt
            parts.append(positive_prompt_str)
        
        # Add negative prompt
        negative_prompt = pnginfo_dict.get("Negative prompt", "")
        if negative_prompt:
            # Ensure negative_prompt is a string
            negative_prompt_str = str(negative_prompt) if not isinstance(negative_prompt, str) else negative_prompt
            parts.append(f"Negative prompt: {negative_prompt_str}")
        
        # Add parameters
        params_list = []
        for key, value in pnginfo_dict.items():
            if key not in ["Positive prompt", "Negative prompt"] and value:
                # Robust value conversion with comprehensive type handling
                def safe_convert_to_string(value):
                    """Safely convert any value to string representation"""
                    if value is None:
                        return ""
                    
                    # Handle string types - always convert to string
                    if isinstance(value, str):
                        return value.strip()
                    
                    # Handle numeric types - convert to string
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
                                            # Always convert to string regardless of type
                                            item_str = safe_convert_to_string(item)
                                            if item_str:
                                                flattened.append(item_str)
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
                    
                    # Handle other types - always convert to string
                    try:
                        return str(value)
                    except Exception:
                        return repr(value)
                
                value_str = safe_convert_to_string(value)
                if value_str:  # Only add non-empty values
                    # Ensure value_str is a string before using it
                    value_str = str(value_str) if not isinstance(value_str, str) else value_str
                    params_list.append(f"{key}: {value_str}")
        
        if params_list:
            parts.append(", ".join(params_list))
        
        return "\n".join(parts)
