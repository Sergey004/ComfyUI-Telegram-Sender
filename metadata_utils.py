import os
import re
import json


class MetadataUtils:
    """
    Utility class for extracting metadata from ComfyUI workflows
    Based on comfyui_image_metadata_extension architecture but independently implemented
    """
    
    @staticmethod
    def debug_prompt_structure(prompt, verbose=False):
        """
        Debug function to print all nodes and their inputs from workflow
        Useful for understanding the actual structure of prompts from real ComfyUI
        
        Usage: MetadataUtils.debug_prompt_structure(prompt, verbose=True)
        """
        if not prompt or not isinstance(prompt, dict):
            print("[MetadataUtils DEBUG] Empty or invalid prompt structure")
            return
        
        print("\n" + "="*80)
        print("[MetadataUtils DEBUG] COMFYUI WORKFLOW STRUCTURE ANALYSIS")
        print("="*80)
        print(f"Total nodes: {len(prompt)}\n")
        
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "UNKNOWN")
            inputs = node_data.get("inputs", {})
            
            print(f"📌 Node ID: {node_id}")
            print(f"   Class Type: {class_type}")
            
            if inputs and verbose:
                print(f"   Inputs ({len(inputs)} items):")
                for key, value in inputs.items():
                    value_type = type(value).__name__
                    
                    # Show actual value with type info
                    if isinstance(value, list):
                        if value:
                            print(f"     ├─ {key}: LIST[{value_type}] ({len(value)} items)")
                            for i, item in enumerate(value[:2]):  # Show first 2 items
                                item_type = type(item).__name__
                                item_str = str(item)[:60]
                                print(f"     │  [{i}] {item_type}: {item_str}")
                            if len(value) > 2:
                                print(f"     │  ... and {len(value) - 2} more items")
                        else:
                            print(f"     ├─ {key}: LIST (empty)")
                    elif isinstance(value, dict):
                        print(f"     ├─ {key}: DICT ({len(value)} keys)")
                    elif isinstance(value, (int, float)):
                        print(f"     ├─ {key}: {value_type} = {value}")
                    elif isinstance(value, str):
                        value_short = value[:50] + "..." if len(value) > 50 else value
                        print(f"     ├─ {key}: {value_type} = '{value_short}'")
                    else:
                        print(f"     ├─ {key}: {value_type} = {str(value)[:50]}")
            elif inputs:
                # Non-verbose mode - just show key names and types
                input_types = {k: type(v).__name__ for k, v in inputs.items()}
                print(f"   Input keys: {list(inputs.keys())}")
                print(f"   Input types: {input_types}")
            print()
        
        print("="*80 + "\n")
    
    @staticmethod
    def extract_metadata_from_workflow(prompt, extra_pnginfo=None, debug=False, verbose_debug=False):
        """
        Extract comprehensive metadata from workflow by directly parsing prompt
        
        Args:
            prompt: dict - ComfyUI workflow prompt (nodes dict)
            extra_pnginfo: dict - Additional PNG info (optional)
            debug: bool - Enable debug output
            verbose_debug: bool - Enable very detailed workflow analysis
        
        Returns: dict containing metadata fields
        """
        if not prompt or not isinstance(prompt, dict):
            return {}
        
        pnginfo_dict = {}
        
        try:
            # Print workflow structure if verbose debug enabled
            if verbose_debug:
                MetadataUtils.debug_prompt_structure(prompt, verbose=True)
            
            if debug or verbose_debug:
                print(f"\n[MetadataUtils DEBUG] Starting metadata extraction")
                print(f"[MetadataUtils DEBUG] Total nodes in workflow: {len(prompt)}")
            
            # Extract each type of metadata directly from prompt nodes
            
            # 1. Extract prompts (positive and negative)
            if debug or verbose_debug:
                print(f"[MetadataUtils DEBUG] Step 1: Extracting prompts...")
            prompts = MetadataUtils._extract_prompts_from_workflow(prompt, debug=debug or verbose_debug)
            if prompts.get("positive"):
                pnginfo_dict["Positive prompt"] = prompts["positive"]
            if prompts.get("negative"):
                pnginfo_dict["Negative prompt"] = prompts["negative"]
            
            # 2. Extract sampling parameters
            if debug or verbose_debug:
                print(f"[MetadataUtils DEBUG] Step 2: Extracting sampling parameters...")
            sampling_params = MetadataUtils._extract_sampling_parameters(prompt, debug=debug or verbose_debug)
            pnginfo_dict.update(sampling_params)
            
            # 3. Extract model information
            if debug or verbose_debug:
                print(f"[MetadataUtils DEBUG] Step 3: Extracting model information...")
            model_info = MetadataUtils._extract_model_info(prompt, debug=debug or verbose_debug)
            pnginfo_dict.update(model_info)
            
            # 4. Extract LoRA information
            if debug or verbose_debug:
                print(f"[MetadataUtils DEBUG] Step 4: Extracting LoRA information...")
            lora_info = MetadataUtils._extract_lora_info(prompt, debug=debug or verbose_debug)
            pnginfo_dict.update(lora_info)
            
            # 5. Extract image size
            if debug or verbose_debug:
                print(f"[MetadataUtils DEBUG] Step 5: Extracting image size...")
            size_info = MetadataUtils._extract_size_info(prompt, debug=debug or verbose_debug)
            pnginfo_dict.update(size_info)
            
            print(f"[MetadataUtils] ✅ Extracted metadata with {len(pnginfo_dict)} fields")
            
            if debug or verbose_debug:
                print("\n[MetadataUtils DEBUG] FINAL EXTRACTED METADATA:")
                for key, value in pnginfo_dict.items():
                    value_short = value[:80] + "..." if len(str(value)) > 80 else value
                    print(f"  {key}: {value_short}")
            
        except Exception as e:
            print(f"[MetadataUtils] ⚠️ Error extracting metadata: {e}")
            import traceback
            traceback.print_exc()
        
        return pnginfo_dict
    
    @staticmethod
    def _extract_prompts_from_workflow(prompt, debug=False):
        """Extract positive and negative prompts from workflow"""
        prompts = {"positive": "", "negative": ""}
        
        if not prompt:
            return prompts
        
        if debug:
            print("  📝 Looking for CLIPTextEncode nodes...")
        
        clip_nodes = []
        
        # Find all CLIPTextEncode nodes
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})
            
            if class_type == "CLIPTextEncode":
                text = inputs.get("text", "")
                if text:
                    # Convert text to string - can be str or list
                    if isinstance(text, list):
                        text = " ".join(str(t) for t in text if t)
                    else:
                        text = str(text)
                    
                    if text.strip():
                        clip_nodes.append({
                            "node_id": node_id,
                            "text": text,
                            "length": len(text)
                        })
        
        if not clip_nodes:
            return prompts
        
        # If we have exactly 2 CLIPTextEncode nodes, try to determine positive/negative
        # Use heuristic: the one with more content is usually negative (more exclusions)
        # But if we find common negative keywords, use those
        if len(clip_nodes) == 2:
            negative_keywords = ["ugly", "bad", "blurry", "low quality", "poor", "worst", "horrible", "awful", "gross", "disgusting"]
            
            # Check which prompt contains negative keywords
            has_negative = [False, False]
            for i, node in enumerate(clip_nodes):
                text_lower = node["text"].lower()
                for keyword in negative_keywords:
                    if keyword in text_lower:
                        has_negative[i] = True
                        break
            
            # If one has negative keywords and the other doesn't, use that as negative
            if has_negative[0] and not has_negative[1]:
                prompts["positive"] = clip_nodes[1]["text"]
                prompts["negative"] = clip_nodes[0]["text"]
                print(f"[MetadataUtils] 📝 Found 2 prompts (keyword detection: negative keywords in first)")
            elif has_negative[1] and not has_negative[0]:
                prompts["positive"] = clip_nodes[0]["text"]
                prompts["negative"] = clip_nodes[1]["text"]
                print(f"[MetadataUtils] 📝 Found 2 prompts (keyword detection: negative keywords in second)")
            else:
                # Fallback: use length-based heuristic (shorter = positive usually)
                sorted_nodes = sorted(clip_nodes, key=lambda x: x["length"])
                prompts["positive"] = sorted_nodes[0]["text"]
                prompts["negative"] = sorted_nodes[1]["text"]
                print(f"[MetadataUtils] 📝 Found 2 prompts (length-based: positive {sorted_nodes[0]['length']}ch, negative {sorted_nodes[1]['length']}ch)")
        
        # If we have 1 or more nodes, use first as positive
        elif len(clip_nodes) >= 1:
            prompts["positive"] = clip_nodes[0]["text"]
            if len(clip_nodes) > 1:
                prompts["negative"] = clip_nodes[1]["text"]
            print(f"[MetadataUtils] 📝 Found {len(clip_nodes)} prompt node(s)")
        
        if debug:
            print(f"  ✓ Prompts extracted:")
            if prompts["positive"]:
                pos_short = prompts["positive"][:60] + "..." if len(prompts["positive"]) > 60 else prompts["positive"]
                print(f"    Positive: {pos_short}")
            if prompts["negative"]:
                neg_short = prompts["negative"][:60] + "..." if len(prompts["negative"]) > 60 else prompts["negative"]
                print(f"    Negative: {neg_short}")
        
        return prompts
    
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
    def _extract_sampling_parameters(prompt, debug=False):
        """Extract sampling parameters from workflow (seed, steps, cfg, etc.)"""
        params = {}
        
        if not prompt:
            return params
        
        if debug:
            print("  ⚙️ Looking for KSampler node...")
        
        # Find KSampler or similar sampling nodes
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})
            
            if "sampler" in class_type.lower() or "ksampler" in class_type.lower():
                # Helper function to safely extract values from lists or return as-is
                def get_value(val):
                    if isinstance(val, list):
                        return val[0] if val else None
                    return val
                
                # Extract all sampling parameters
                seed = get_value(inputs.get("seed"))
                steps = get_value(inputs.get("steps"))
                cfg = get_value(inputs.get("cfg"))
                sampler_name = inputs.get("sampler_name")
                scheduler = inputs.get("scheduler")
                denoise = get_value(inputs.get("denoise"))
                
                # Convert list text fields to string
                if isinstance(sampler_name, list):
                    sampler_name = sampler_name[0] if sampler_name else None
                if isinstance(scheduler, list):
                    scheduler = scheduler[0] if scheduler else None
                
                # Safe type conversions with error handling
                try:
                    if steps is not None:
                        params["Steps"] = str(int(steps))
                except (ValueError, TypeError):
                    pass
                
                if sampler_name:
                    params["Sampler"] = str(sampler_name)
                
                if scheduler:
                    params["Scheduler"] = str(scheduler)
                
                try:
                    if cfg is not None:
                        params["CFG scale"] = str(float(cfg))
                except (ValueError, TypeError):
                    pass
                
                try:
                    if seed is not None:
                        params["Seed"] = str(int(seed))
                except (ValueError, TypeError):
                    pass
                
                try:
                    if denoise is not None and float(denoise) != 1.0:
                        params["Denoising strength"] = str(float(denoise))
                except (ValueError, TypeError):
                    pass
                if denoise is not None and float(denoise) != 1.0:
                    params["Denoising strength"] = str(float(denoise))
                
                print(f"[MetadataUtils] ⚙️ Found sampling parameters in node {node_id}")
                break  # Use first sampler found
        
        if debug:
            print(f"  ✓ Sampling params extracted: {list(params.keys())}")
        
        return params
    
    @staticmethod
    def _extract_model_info(prompt, debug=False):
        """Extract model information from workflow"""
        model_info = {}
        
        if not prompt:
            return model_info
        
        if debug:
            print("  🤖 Looking for model/VAE nodes...")
        
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})
            
            # Extract checkpoint/model
            if "checkpoint" in class_type.lower() and not model_info.get("Model"):
                model_name = inputs.get("ckpt_name")
                if model_name:
                    model_info["Model"] = os.path.splitext(os.path.basename(model_name))[0]
                    print(f"[MetadataUtils] 🤖 Found model: {model_info['Model']}")
            
            # Extract VAE
            if "vae" in class_type.lower() and not model_info.get("VAE"):
                vae_name = inputs.get("vae_name")
                if vae_name:
                    model_info["VAE"] = os.path.splitext(os.path.basename(vae_name))[0]
                    print(f"[MetadataUtils] 🎨 Found VAE: {model_info['VAE']}")
        
        if debug:
            if "Model" in model_info:
                print(f"  ✓ Model: {model_info['Model']}")
            if "VAE" in model_info:
                print(f"  ✓ VAE: {model_info['VAE']}")
        
        return model_info
    
    @staticmethod
    def _extract_lora_info(prompt, debug=False):
        """Extract LoRA information from workflow"""
        lora_info = {}
        loras = []
        
        if not prompt:
            return lora_info
        
        if debug:
            print("  🔗 Looking for LoRA nodes...")
        
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "")
            
            if "lora" in class_type.lower():
                inputs = node_data.get("inputs", {})
                lora_name = inputs.get("lora_name")
                
                # Try multiple strength field names
                strength = inputs.get("strength_model", 
                          inputs.get("strength_clip", 
                          inputs.get("strength", 1.0)))
                
                if lora_name:
                    clean_name = os.path.splitext(os.path.basename(lora_name))[0]
                    loras.append(f"{clean_name}: {strength}")
                    print(f"[MetadataUtils] 🎨 Found LoRA: {clean_name} (strength: {strength})")
        
        if loras:
            lora_info["Lora hashes"] = ", ".join(loras)
        
        if debug:
            if loras:
                print(f"  ✓ LoRAs found: {loras}")
            else:
                print(f"  ✓ No LoRA nodes found")
        
        return lora_info
    
    @staticmethod
    def _extract_size_info(prompt, debug=False):
        """Extract image size information from workflow"""
        size_info = {}
        
        if not prompt:
            return size_info
        
        if debug:
            print("  📐 Looking for image size...")
        
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})
            
            # Look for nodes that generate images (EmptyLatentImage, etc)
            if ("empty" in class_type.lower() or "latent" in class_type.lower()) and "latent" not in class_type.lower():
                width = inputs.get("width")
                height = inputs.get("height")
                
                if width and height:
                    size_info["Size"] = f"{width}x{height}"
                    print(f"[MetadataUtils] 📐 Found image size: {width}x{height}")
                    break
        
        if debug:
            if "Size" in size_info:
                print(f"  ✓ Size: {size_info['Size']}")
        
        return size_info
    
    @staticmethod
    def build_a1111_style_metadata(pnginfo_dict):
        """
        Build A1111-style metadata string from PNG info dictionary
        Format: "positive_prompt\nNegative prompt: negative_prompt\nSteps: 20, Sampler: euler, ..."
        """
        if not pnginfo_dict:
            return ""
        
        parts = []
        
        # Add positive prompt first
        positive = pnginfo_dict.get("Positive prompt", "")
        if positive:
            parts.append(str(positive))
        
        # Add negative prompt with prefix
        negative = pnginfo_dict.get("Negative prompt", "")
        if negative:
            parts.append(f"Negative prompt: {negative}")
        
        # Add all other parameters in a single line
        param_parts = []
        
        param_order = ["Steps", "Sampler", "Scheduler", "CFG scale", "Seed", "Size", "Model", "VAE", "Lora hashes", "Denoising strength"]
        
        for param in param_order:
            if param in pnginfo_dict:
                value = pnginfo_dict[param]
                if param == "Lora hashes":
                    # Special formatting for LoRA
                    param_parts.append(f'{param}: "{value}"')
                elif param in ["CFG scale", "Denoising strength"]:
                    # Float parameters
                    try:
                        float_val = float(value)
                        param_parts.append(f"{param}: {float_val}")
                    except (ValueError, TypeError):
                        param_parts.append(f"{param}: {value}")
                else:
                    param_parts.append(f"{param}: {value}")
        
        # Add any remaining parameters not in our list
        for key, value in pnginfo_dict.items():
            if key not in ["Positive prompt", "Negative prompt"] and key not in param_order:
                param_parts.append(f"{key}: {value}")
        
        if param_parts:
            parts.append(", ".join(param_parts))
        
        return "\n".join(parts)
    
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
