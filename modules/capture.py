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
        for i, (lora_name, lora_strength) in enumerate(zip(lora_names, lora_strengths)):
            if i < len(lora_strengths):
                clean_name = os.path.splitext(os.path.basename(lora_name[1]))[0]
                strength = lora_strengths[i][1] if i < len(lora_strengths) else 1.0
                loras.append(f"{clean_name}: {strength}")
        
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
            parts.append(positive_prompt)
        
        # Add negative prompt
        negative_prompt = pnginfo_dict.get("Negative prompt", "")
        if negative_prompt:
            parts.append(f"Negative prompt: {negative_prompt}")
        
        # Add parameters
        params_list = []
        for key, value in pnginfo_dict.items():
            if key not in ["Positive prompt", "Negative prompt"] and value:
                params_list.append(f"{key}: {value}")
        
        if params_list:
            parts.append(", ".join(params_list))
        
        return "\n".join(parts)
