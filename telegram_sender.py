import os
import re
import requests
import threading
import time
import json
import sys
sys.path.append("../../")
import folder_paths
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch
from datetime import datetime

# Import metadata utilities
try:
    from .metadata_utils import MetadataUtils, extract_metadata, build_metadata_text, format_telegram_metadata
except ImportError:
    # Fallback for direct execution
    from metadata_utils import MetadataUtils, extract_metadata, build_metadata_text, format_telegram_metadata

# Config file path
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "telegram_config.json")

def load_config():
    """Load configuration from file"""
    if not os.path.exists(CONFIG_FILE):
        return {"bot_token": "", "default_chat_id": ""}
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"bot_token": "", "default_chat_id": ""}

def save_config(config):
    """Save configuration to file"""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"[TelegramSender] ⚠️ Could not save config: {e}")
        return False

class TelegramConfig:
    """
    Node to configure Telegram bot token (stores securely in config file)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        config = load_config()
        return {
            "required": {
                "bot_token": ("STRING", {
                    "default": config.get("bot_token", ""),
                    "multiline": False
                }),
                "default_chat_id": ("STRING", {
                    "default": config.get("default_chat_id", ""),
                    "multiline": False
                }),
                "lora_mapping": ("STRING", {
                    "default": config.get("lora_mapping", ""),
                    "multiline": True,
                    "tooltip": "LoRA to channel mapping (one per line): lora_name:chat_id"
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_config"
    OUTPUT_NODE = True
    CATEGORY = "image/telegram"
    DESCRIPTION = "Configure Telegram bot token, default chat ID and LoRA mapping. Token is stored securely in a config file."

    def save_config(self, bot_token, default_chat_id, lora_mapping):
        config = {
            "bot_token": bot_token.strip(),
            "default_chat_id": default_chat_id.strip(),
            "lora_mapping": lora_mapping.strip()
        }
        
        if save_config(config):
            print("[TelegramConfig] ✅ Configuration saved successfully")
        else:
            print("[TelegramConfig] ❌ Failed to save configuration")
        
        return ()

class TelegramSender:
    """
    Send generated images to Telegram channels/chats
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        
    @classmethod
    def INPUT_TYPES(cls):
        config = load_config()
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "chat_id": ("STRING", {
                    "default": config.get("default_chat_id", ""),
                    "multiline": False,
                    "tooltip": "Leave empty to use default from config"
                }),
                "bot_token_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Override token from config (optional)"
                }),
                "positive_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Positive prompt text"
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Negative prompt text"
                }),
                "send_as_document": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Send as file in addition to photo (no compression, sends both)"
                }),
                "max_size": ("INT", {
                    "default": 2560, 
                    "min": 512, 
                    "max": 10240,
                    "step": 64,
                    "tooltip": "Max size for portrait/square images (pixels)"
                }),
                "landscape_max_width": ("INT", {
                    "default": 5120, 
                    "min": 512, 
                    "max": 10240,
                    "step": 64,
                    "tooltip": "Max width for landscape images (pixels)"
                }),
                "enable_nsfw_detection": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Redirect to NSFW channel if 'nsfw' in prompt"
                }),
                "nsfw_channel_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Channel ID for NSFW content"
                }),
                "unsorted_channel_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Fallback channel if no chat_id specified"
                }),
                "enable_lora_routing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable automatic routing based on LoRA mapping from config"
                }),
                "retry_count": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Number of retry attempts on failure"
                }),
                "retry_delay": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 60,
                    "tooltip": "Delay between retries (seconds)"
                }),
                "enable_enhanced_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use enhanced metadata extraction from comfyui_image_metadata_extension"
                }),
                "filename_prefix": ("STRING", {
                    "default": "telegram_%date%_%model%_%seed%",
                    "multiline": False,
                    "tooltip": "Filename prefix with placeholders: %date%, %seed%, %model%, %width%, %height%, %pprompt%, %nprompt%"
                }),
                "subdirectory_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Subdirectory for saving files (e.g., tg_temp). Leave empty for default output directory."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "send_to_telegram"
    OUTPUT_NODE = True
    CATEGORY = "image/telegram"
    DESCRIPTION = "Sends generated images to Telegram. Always sends as photo, optionally sends as document. Configure bot token using TelegramConfig node first."

    def send_to_telegram(self, images, chat_id="", bot_token_override="", 
                        positive_prompt="", negative_prompt="",
                        send_as_document=False, max_size=2560, 
                        landscape_max_width=5120, enable_nsfw_detection=False,
                        nsfw_channel_id="", unsorted_channel_id="",
                        enable_lora_routing=True,
                        retry_count=3, retry_delay=5,
                        enable_enhanced_metadata=True,
                        filename_prefix="telegram_%date%_%model%_%seed%",
                        subdirectory_name="",
                        prompt=None, extra_pnginfo=None):
        
        # Get bot token from override or config
        config = load_config()
        bot_token = bot_token_override.strip() if bot_token_override else config.get("bot_token", "")
        
        # Validation
        if not bot_token:
            print("[TelegramSender] ⚠️ No bot token configured. Use TelegramConfig node to set token.")
            return (images,)
        
        # Use default chat_id from config if not provided
        if not chat_id:
            chat_id = config.get("default_chat_id", "")
            
        if not chat_id and not nsfw_channel_id and not unsorted_channel_id:
            print("[TelegramSender] ⚠️ No chat ID provided. Skipping send.")
            return (images,)

        # Extract metadata from workflow
        metadata_text = self._build_metadata_text(
            positive_prompt, negative_prompt, prompt, extra_pnginfo, enable_enhanced_metadata
        )
        
        # Extract LoRAs from workflow for routing (using enhanced extraction)
        loras_in_workflow = self._extract_loras_from_workflow(prompt, extra_pnginfo, enable_enhanced_metadata) if prompt else []

        # Process each image
        for i, image in enumerate(images):
            try:
                # Convert tensor to PIL Image
                img_np = (255. * image.cpu().numpy()).astype(np.uint8)
                pil_image = Image.fromarray(img_np)
                
                # Format filename and subdirectory
                pnginfo_dict = {}
                if enable_enhanced_metadata and prompt:
                    try:
                        pnginfo_dict = MetadataUtils.extract_metadata_from_workflow(prompt, extra_pnginfo)
                    except Exception as e:
                        print(f"[TelegramSender] ⚠️ Enhanced metadata extraction failed: {e}")
                
                # Format filename prefix
                formatted_filename = self._format_filename(filename_prefix.strip(), pnginfo_dict)
                if not formatted_filename:
                    formatted_filename = f"telegram_temp_{int(time.time())}_{i}"
                
                # Handle subdirectory
                save_dir = self.output_dir
                if subdirectory_name.strip():
                    formatted_subdir = self._format_filename(subdirectory_name.strip(), pnginfo_dict)
                    save_dir = os.path.join(self.output_dir, formatted_subdir)
                    os.makedirs(save_dir, exist_ok=True)
                
                # Save temporarily with metadata
                temp_path = os.path.join(save_dir, f"{formatted_filename}.png")
                
                # Add metadata to PNG
                pnginfo = PngInfo()
                if metadata_text:
                    pnginfo.add_text("parameters", metadata_text)
                
                # Add workflow info if available
                if prompt is not None:
                    pnginfo.add_text("prompt", json.dumps(prompt))
                
                # Add enhanced metadata if available and enabled
                if enable_enhanced_metadata:
                    try:
                        pnginfo_dict = MetadataUtils.extract_metadata_from_workflow(prompt, extra_pnginfo)
                        
                        if pnginfo_dict:
                            # Add comprehensive metadata fields
                            for key, value in pnginfo_dict.items():
                                if value and isinstance(value, (str, int, float)):
                                    pnginfo.add_text(key, str(value))
                            print(f"[TelegramSender] ✅ Enhanced metadata embedded in PNG")
                    
                    except Exception as e:
                        print(f"[TelegramSender] ⚠️ Enhanced metadata embedding failed: {e}")
                
                # Add original workflow info
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        pnginfo.add_text(key, json.dumps(value) if isinstance(value, dict) else str(value))
                
                pil_image.save(temp_path, "PNG", pnginfo=pnginfo)
                
                # Determine target chat_id
                target_chat_id = self._determine_chat_id(
                    chat_id, positive_prompt, negative_prompt, 
                    enable_nsfw_detection, nsfw_channel_id, unsorted_channel_id,
                    enable_lora_routing, loras_in_workflow
                )
                
                if not target_chat_id:
                    print("[TelegramSender] ⚠️ No valid chat_id determined. Skipping image.")
                    self._cleanup_file(temp_path)
                    continue
                
                # Always send as photo (resized/compressed)
                photo_path = self._resize_image(temp_path, max_size, landscape_max_width)
                
                # Compress if too large
                try:
                    if os.path.getsize(photo_path) > 10 * 1024 * 1024:
                        photo_path = self._compress_image(photo_path)
                except Exception as e:
                    print(f"[TelegramSender] Compression check failed: {e}")
                
                # Send photo in background thread
                threading.Thread(
                    target=self._send_telegram_request,
                    args=(photo_path, target_chat_id, bot_token, 
                          False, temp_path, retry_count, retry_delay),
                    daemon=True
                ).start()
                
                # Optionally send as document (original PNG)
                if send_as_document:
                    threading.Thread(
                        target=self._send_telegram_request,
                        args=(temp_path, target_chat_id, bot_token, 
                              True, temp_path, retry_count, retry_delay),
                        daemon=True
                    ).start()
                    print(f"[TelegramSender] 📄 Sending as document in addition to photo (original PNG, no resize)")
                else:
                    print(f"[TelegramSender] 📷 Sending as photo only")
                
            except Exception as e:
                print(f"[TelegramSender] ❌ Error processing image {i}: {e}")
                continue
        
        return (images,)

    def _build_metadata_text(self, positive_prompt, negative_prompt, prompt_dict, extra_pnginfo, enable_enhanced=True):
        """Build A1111-style metadata text using custom metadata extraction"""
        
        # Use the enhanced metadata extraction if available and enabled
        if enable_enhanced and prompt_dict:
            try:
                # Extract comprehensive metadata using custom implementation
                pnginfo_dict = MetadataUtils.extract_metadata_from_workflow(prompt_dict, extra_pnginfo)
                
                # Build A1111-style metadata string
                if pnginfo_dict:
                    metadata_text = MetadataUtils.build_a1111_style_metadata(pnginfo_dict)
                    
                    # If we got valid metadata, use it
                    if metadata_text and metadata_text.strip():
                        print(f"[TelegramSender] ✅ Using enhanced metadata extraction")
                        return metadata_text
                
            except Exception as e:
                print(f"[TelegramSender] ⚠️ Enhanced metadata extraction failed: {e}")
        
        # Fallback to original method if enhanced extraction fails
        parts = []
        
        # Add positive prompt
        if positive_prompt:
            parts.append(positive_prompt)
        
        # Add negative prompt
        if negative_prompt:
            parts.append(f"Negative prompt: {negative_prompt}")
        
        # Try to extract additional parameters from workflow
        if prompt_dict:
            params = self._extract_parameters_from_workflow(prompt_dict)
            if params:
                parts.append(params)
        
        return "\n".join(parts)

    def _extract_parameters_from_workflow(self, prompt_dict):
        """Extract generation parameters from ComfyUI workflow"""
        params = []
        
        try:
            # Search for KSampler nodes to get seed, steps, cfg, etc.
            for node_id, node_data in prompt_dict.items():
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
                        params.append(f"Steps: {steps}")
                    if sampler_name:
                        params.append(f"Sampler: {sampler_name}")
                    if scheduler:
                        params.append(f"Schedule type: {scheduler}")
                    if cfg:
                        params.append(f"CFG scale: {cfg}")
                    if seed is not None:
                        params.append(f"Seed: {seed}")
                    if denoise is not None and denoise != 1.0:
                        params.append(f"Denoising strength: {denoise}")
                    
                    break  # Use first sampler found
                
                # Look for model loader
                if "checkpointloader" in class_type.lower():
                    inputs = node_data.get("inputs", {})
                    model_name = inputs.get("ckpt_name")
                    if model_name:
                        params.append(f"Model: {model_name}")
                
                # Look for VAE
                if "vaeloader" in class_type.lower():
                    inputs = node_data.get("inputs", {})
                    vae_name = inputs.get("vae_name")
                    if vae_name:
                        params.append(f"VAE: {vae_name}")
                
                # Look for LoRAs
                if "lora" in class_type.lower():
                    inputs = node_data.get("inputs", {})
                    lora_name = inputs.get("lora_name")
                    strength = inputs.get("strength_model", inputs.get("strength", 1.0))
                    if lora_name:
                        params.append(f"Lora: {lora_name}: {strength}")
        
        except Exception as e:
            print(f"[TelegramSender] Error extracting parameters: {e}")
        
        # Ensure all params are strings before joining
        if params:
            params_str = [str(param) for param in params]
            return ", ".join(params_str)
        return ""

    def _extract_loras_from_workflow(self, prompt_dict, extra_pnginfo=None, enable_enhanced=True):
        """Extract list of LoRA names used in workflow using enhanced metadata extraction"""
        loras = []
        
        try:
            # Try enhanced metadata extraction first if enabled
            if enable_enhanced and prompt_dict:
                pnginfo_dict = MetadataUtils.extract_metadata_from_workflow(prompt_dict, extra_pnginfo)
                
                if pnginfo_dict:
                    enhanced_loras = MetadataUtils.extract_loras_from_metadata(pnginfo_dict)
                    for lora_info in enhanced_loras:
                        lora_name = lora_info.get('name', '')
                        if lora_name:
                            loras.append(lora_name.lower())
                    print(f"[TelegramSender] ✅ Enhanced LoRA extraction found {len(loras)} LoRAs")
                    return loras
        
        except Exception as e:
            print(f"[TelegramSender] ⚠️ Enhanced LoRA extraction failed: {e}")
        
        # Fallback to original method
        try:
            for node_id, node_data in prompt_dict.items():
                class_type = node_data.get("class_type", "")
                
                if "lora" in class_type.lower():
                    inputs = node_data.get("inputs", {})
                    lora_name = inputs.get("lora_name")
                    if lora_name:
                        # Remove file extension and path
                        clean_name = os.path.splitext(os.path.basename(lora_name))[0]
                        loras.append(clean_name.lower())
        
        except Exception as e:
            print(f"[TelegramSender] Error extracting LoRAs: {e}")
        
        return loras

    def _parse_lora_mapping(self):
        """Parse LoRA mapping from config"""
        config = load_config()
        mapping_str = config.get("lora_mapping", "")
        
        if not mapping_str:
            return {}
        
        mapping = {}
        for line in mapping_str.strip().split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            try:
                lora_key, chat_id = line.split(':', 1)
                lora_key = lora_key.strip().lower()
                chat_id = chat_id.strip()
                
                if lora_key and chat_id:
                    mapping[lora_key] = chat_id
            except:
                continue
        
        return mapping

    def _determine_chat_id(self, default_chat_id, positive_prompt, negative_prompt,
                          enable_nsfw, nsfw_channel_id, unsorted_channel_id,
                          enable_lora_routing, loras_in_workflow):
        """Determine which chat to send to based on content"""
        target_chat_id = default_chat_id
        
        # LoRA-based routing (highest priority after explicit chat_id)
        if enable_lora_routing and not target_chat_id and loras_in_workflow:
            lora_mapping = self._parse_lora_mapping()
            
            if lora_mapping:
                # Check each LoRA in workflow against mapping
                for lora_in_workflow in loras_in_workflow:
                    for lora_key, mapped_chat_id in lora_mapping.items():
                        # Match if mapping key is substring of actual LoRA name
                        if lora_key in lora_in_workflow:
                            print(f"[TelegramSender] 🎯 LoRA routing: '{lora_in_workflow}' matched '{lora_key}' → {mapped_chat_id}")
                            target_chat_id = mapped_chat_id
                            break
                    
                    if target_chat_id:
                        break
        
        # NSFW detection (overrides LoRA routing if NSFW found)
        if enable_nsfw and nsfw_channel_id:
            positive_lower = positive_prompt.lower() if positive_prompt else ""
            negative_lower = negative_prompt.lower() if negative_prompt else ""
            
            # If NSFW in positive but not in negative, redirect
            if "nsfw" in positive_lower and "nsfw" not in negative_lower:
                print("[TelegramSender] 🔞 NSFW detected, redirecting to NSFW channel")
                target_chat_id = nsfw_channel_id
        
        # Fallback to unsorted channel if no chat_id
        if not target_chat_id and unsorted_channel_id:
            print("[TelegramSender] 📦 Using unsorted channel")
            target_chat_id = unsorted_channel_id
        
        return target_chat_id

    def _resize_image(self, image_path, max_size, landscape_max_width):
        """Resize image if it exceeds maximum dimensions"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Determine max value based on orientation
                if width >= height:
                    max_val = landscape_max_width
                    if width <= max_val:
                        return image_path
                else:
                    max_val = max_size
                    if max(width, height) <= max_val:
                        return image_path

                # Calculate new dimensions
                ratio = max_val / max(width, height)
                new_size = (int(width * ratio), int(height * ratio))
                
                print(f"[TelegramSender] 🔄 Resizing from {width}x{height} to {new_size}")
                
                img = img.resize(new_size, Image.LANCZOS)
                temp_path = os.path.splitext(image_path)[0] + "_resized.jpg"
                img.save(temp_path, "JPEG", quality=85, optimize=True)
                
                return temp_path
        except Exception as e:
            print(f"[TelegramSender] ⚠️ Resize error: {e}")
            return image_path

    def _compress_image(self, image_path, target_size=10 * 1024 * 1024):
        """Compress image to meet Telegram size limits"""
        temp_path = os.path.splitext(image_path)[0] + "_compressed.jpg"
        quality = 85
        
        try:
            with Image.open(image_path) as img:
                while quality >= 30:
                    img.save(temp_path, "JPEG", quality=quality, optimize=True)
                    file_size = os.path.getsize(temp_path)
                    
                    if file_size <= target_size:
                        print(f"[TelegramSender] 🗜️ Compressed to {file_size // 1024}KB (quality: {quality})")
                        return temp_path
                    
                    quality -= 10
                
                print(f"[TelegramSender] ⚠️ Could not compress below target size")
                return temp_path
        except Exception as e:
            print(f"[TelegramSender] ⚠️ Compress error: {e}")
            return image_path

    def _send_telegram_request(self, image_path, chat_id, bot_token, 
                               as_document, original_path, retry_count, retry_delay):
        """Send image to Telegram with retry logic"""
        method = "sendDocument" if as_document else "sendPhoto"
        param_name = "document" if as_document else "photo"
        url = f"https://api.telegram.org/bot{bot_token}/{method}"
        
        for attempt in range(retry_count):
            try:
                with open(image_path, 'rb') as f:
                    files = {param_name: f}
                    data = {'chat_id': chat_id}
                    response = requests.post(url, data=data, files=files, timeout=30)
                
                if response.ok:
                    file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
                    file_type = "document" if as_document else "photo"
                    print(f"[TelegramSender] ✅ Sent to {chat_id} as {file_type} ({file_size:.2f}MB)")
                    
                    # Clean up only temporary files, not the original if sent as document
                    if as_document and image_path == original_path:
                        # Don't delete original when sent as document
                        pass
                    else:
                        self._cleanup_files(image_path, original_path if image_path != original_path else None)
                    return
                else:
                    error_msg = response.json().get('description', response.text)
                    print(f"[TelegramSender] ❌ [{attempt+1}/{retry_count}] Error: {error_msg}")
                    
            except requests.exceptions.Timeout:
                print(f"[TelegramSender] ⏱️ [{attempt+1}/{retry_count}] Timeout")
            except Exception as e:
                print(f"[TelegramSender] ❌ [{attempt+1}/{retry_count}] Exception: {e}")
            
            if attempt < retry_count - 1:
                time.sleep(retry_delay)
        
        print(f"[TelegramSender] ❌ Failed to send after {retry_count} attempts")
        if as_document and image_path == original_path:
            pass  # Don't delete original
        else:
            self._cleanup_files(image_path, original_path if image_path != original_path else None)

    def _cleanup_files(self, *paths):
        """Clean up temporary files"""
        for path in paths:
            self._cleanup_file(path)

    def _format_filename(self, filename_template, pnginfo_dict):
        """Format filename template with placeholders like comfyui_image_metadata_extension"""
        if "%" not in filename_template:
            return filename_template
            
        now = datetime.now()
        date_table = {
            "yyyy": f"{now.year}",
            "MM": f"{now.month:02d}",
            "dd": f"{now.day:02d}",
            "hh": f"{now.hour:02d}",
            "mm": f"{now.minute:02d}",
            "ss": f"{now.second:02d}",
        }
        
        pattern_format = re.compile(r"(%[^%]+%)")
        segments = pattern_format.findall(filename_template)
        
        for segment in segments:
            parts = segment.strip("%").split(":")
            key = parts[0]

            if key == "seed":
                seed = pnginfo_dict.get("Seed", pnginfo_dict.get("seed", ""))
                filename_template = filename_template.replace(segment, str(seed))

            elif key == "width":
                width = pnginfo_dict.get("Width", pnginfo_dict.get("width", ""))
                filename_template = filename_template.replace(segment, str(width))

            elif key == "height":
                height = pnginfo_dict.get("Height", pnginfo_dict.get("height", ""))
                filename_template = filename_template.replace(segment, str(height))

            elif key == "pprompt":
                prompt = pnginfo_dict.get("Positive prompt", pnginfo_dict.get("pprompt", ""))
                length = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                prompt = prompt.replace("\n", " ")
                filename_template = filename_template.replace(segment, prompt[:length].strip() if length else prompt.strip())

            elif key == "nprompt":
                prompt = pnginfo_dict.get("Negative prompt", pnginfo_dict.get("nprompt", ""))
                length = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                prompt = prompt.replace("\n", " ")
                filename_template = filename_template.replace(segment, prompt[:length].strip() if length else prompt.strip())

            elif key == "model":
                model = pnginfo_dict.get("Model", pnginfo_dict.get("model", ""))
                if model:
                    model = os.path.splitext(os.path.basename(model))[0]
                length = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                filename_template = filename_template.replace(segment, model[:length] if length else model)

            elif key == "date":
                date_format = parts[1] if len(parts) > 1 else "yyyyMMddhhmmss"
                for k, v in date_table.items():
                    date_format = date_format.replace(k, v)
                filename_template = filename_template.replace(segment, date_format)

        return filename_template

    def _cleanup_file(self, path):
        """Clean up a single temporary file"""
        if not path or not os.path.exists(path):
            return
            
        if any(marker in path for marker in ["_resized", "_compressed", "telegram_temp"]):
            try:
                os.remove(path)
            except Exception as e:
                print(f"[TelegramSender] ⚠️ Could not delete {path}: {e}")


NODE_CLASS_MAPPINGS = {
    "TelegramConfig": TelegramConfig,
    "TelegramSender": TelegramSender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TelegramConfig": "⚙️ Telegram Config",
    "TelegramSender": "📤 Send to Telegram"
}
