import os
import re
import requests
import threading
import time
import json
import sys
import socket
from queue import Queue, Empty
sys.path.append("../../")
import folder_paths # type: ignore
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch
from datetime import datetime
from urllib3.exceptions import ProtocolError

# Global queue and semaphore for managing concurrent uploads
# Limit to 2 simultaneous uploads to avoid overwhelming the connection
_upload_semaphore = threading.Semaphore(2)
_upload_lock = threading.Lock()

# Import metadata wrapper based on comfyui_image_metadata_extension
try:
    from .telegram_metadata import TelegramMetadata
except (ImportError, ModuleNotFoundError):
    try:
        from telegram_metadata import TelegramMetadata
    except (ImportError, ModuleNotFoundError):
        # Fallback if not available
        class TelegramMetadata:
            @staticmethod
            def get_metadata(prompt, prefer_nearest=False):
                return {}
            
            @staticmethod
            def get_parameters_str(pnginfo_dict):
                return ""

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
        print(f"[Telegram Sender] ⚠️ Could not save config: {e}")
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
            },
            "optional": {
                "nsfw_channel_id": ("STRING", {
                    "default": config.get("nsfw_channel_id", ""),
                    "multiline": False,
                    "tooltip": "Channel ID for NSFW content"
                }),
                "unsorted_channel_id": ("STRING", {
                    "default": config.get("unsorted_channel_id", ""),
                    "multiline": False,
                    "tooltip": "Default channel for unrouted images"
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_config"
    OUTPUT_NODE = True
    CATEGORY = "image/telegram"
    DESCRIPTION = "Configure Telegram bot token, default chat ID, LoRA mapping and channel IDs. Token is stored securely in a config file."

    def save_config(self, bot_token, default_chat_id, lora_mapping, nsfw_channel_id="", unsorted_channel_id=""):
        config = {
            "bot_token": bot_token.strip(),
            "default_chat_id": default_chat_id.strip(),
            "lora_mapping": lora_mapping.strip(),
            "nsfw_channel_id": nsfw_channel_id.strip(),
            "unsorted_channel_id": unsorted_channel_id.strip()
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
                    "default": config.get("nsfw_channel_id", ""),
                    "multiline": False,
                    "tooltip": "Channel ID for NSFW content (loads from config if empty)"
                }),
                "unsorted_channel_id": ("STRING", {
                    "default": config.get("unsorted_channel_id", ""),
                    "multiline": False,
                    "tooltip": "Fallback channel if no chat_id specified (loads from config if empty)"
                }),
                "enable_lora_routing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable automatic routing based on LoRA mapping from config"
                }),
                "retry_count": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 99,
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
                "debug_metadata": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable debug output for metadata extraction (shows extraction steps in console)"
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
                        retry_count=99, retry_delay=5,
                        enable_enhanced_metadata=True,
                        filename_prefix="telegram_%date%_%model%_%seed%",
                        subdirectory_name="",
                        debug_metadata=False,
                        prompt=None, extra_pnginfo=None):
        
        # Log input parameters for debugging
        print(f"[Telegram Sender] 🔧 Parameters: retry_count={retry_count}, retry_delay={retry_delay}")
        
        # Get bot token from override or config
        config = load_config()
        bot_token = bot_token_override.strip() if bot_token_override else config.get("bot_token", "")
        
        # Validation
        if not bot_token:
            print("[Telegram Sender] ⚠️ No bot token configured. Use TelegramConfig node to set token.")
            return (images,)
        
        # Use default chat_id from config if not provided
        if not chat_id:
            chat_id = config.get("default_chat_id", "")
            
        if not chat_id and not nsfw_channel_id and not unsorted_channel_id:
            print("[Telegram Sender] ⚠️ No chat ID provided. Skipping send.")
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
                        pnginfo_dict = TelegramMetadata.get_metadata(prompt)
                    except Exception as e:
                        print(f"[Telegram Sender] ⚠️ Enhanced metadata extraction failed: {e}")
                
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
                
                # Add enhanced metadata if available and enabled
                if enable_enhanced_metadata and pnginfo_dict:
                    try:
                        # Generate A1111-style parameters string and add as main "parameters" field
                        parameters_str = TelegramMetadata.get_parameters_str(pnginfo_dict)
                        if parameters_str:
                            pnginfo.add_text("parameters", parameters_str)
                        
                        # Also add individual metadata fields for tools that read them separately
                        for key, value in pnginfo_dict.items():
                            if value and isinstance(value, (str, int, float)):
                                pnginfo.add_text(key, str(value))
                        print(f"[Telegram Sender] ✅ Enhanced metadata embedded in PNG")
                    
                    except Exception as e:
                        print(f"[Telegram Sender] ⚠️ Enhanced metadata embedding failed: {e}")
                elif metadata_text:
                    # Fallback to user-provided metadata if no enhanced metadata
                    pnginfo.add_text("parameters", metadata_text)
                
                # Add workflow info if available
                if prompt is not None:
                    pnginfo.add_text("prompt", json.dumps(prompt))
                
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
                    print("[Telegram Sender] ⚠️ No valid chat_id determined. Skipping image.")
                    self._cleanup_file(temp_path)
                    continue
                
                # Always send as photo (resized/compressed)
                photo_path = self._resize_image(temp_path, max_size, landscape_max_width)
                
                # Compress if too large
                try:
                    if os.path.getsize(photo_path) > 10 * 1024 * 1024:
                        photo_path = self._compress_image(photo_path)
                except Exception as e:
                    print(f"[Telegram Sender] Compression check failed: {e}")
                
                # Ensure file is fully written to disk
                self._ensure_file_synced(photo_path)
                
                # Send photo in background thread
                threading.Thread(
                    target=self._send_telegram_request,
                    args=(photo_path, target_chat_id, bot_token, 
                          False, temp_path, retry_count, retry_delay),
                    daemon=True
                ).start()
                
                # Optionally send as document (original PNG)
                # Wait a bit before sending document to avoid overwhelming the connection
                if send_as_document:
                    # Ensure temp_path is synced before sending
                    self._ensure_file_synced(temp_path)
                    
                    # Small delay to avoid simultaneous uploads
                    time.sleep(2)
                    
                    threading.Thread(
                        target=self._send_telegram_request,
                        args=(temp_path, target_chat_id, bot_token, 
                              True, temp_path, retry_count, retry_delay),
                        daemon=True
                    ).start()
                    print(f"[Telegram Sender] 📄 Sending as document in addition to photo (original PNG, no resize)")
                else:
                    print(f"[Telegram Sender] 📷 Sending as photo only")
                
            except Exception as e:
                print(f"[Telegram Sender] ❌ Error processing image {i}: {e}")
                continue
        
        return (images,)

    def _build_metadata_text(self, positive_prompt, negative_prompt, prompt_dict, extra_pnginfo, enable_enhanced=True):
        """Build A1111-style metadata text using custom metadata extraction"""
        
        # Use the enhanced metadata extraction if available and enabled
        if enable_enhanced and prompt_dict:
            try:
                # Extract comprehensive metadata using custom implementation
                pnginfo_dict = {}  # Metadata extraction optional
                
                # Build A1111-style metadata string
                if pnginfo_dict:
                    try:
                        # Format metadata for display
                        metadata_text = "\n".join([f"{k}: {v}" for k, v in pnginfo_dict.items()])
                        
                        # If we got valid metadata, use it
                        if metadata_text and metadata_text.strip():
                            print(f"[Telegram Sender] ✅ Using enhanced metadata extraction from workflow")
                            return metadata_text
                    except Exception as e:
                        print(f"[Telegram Sender] ⚠️ Enhanced metadata building failed: {e}")
                
            except Exception as e:
                print(f"[Telegram Sender] ⚠️ Enhanced metadata extraction failed: {e}")
        
        # Fallback to original method if enhanced extraction fails
        # But still use enhanced extraction for metadata if available
        try:
            if enable_enhanced and prompt_dict:
                pnginfo_dict = {}  # Metadata extraction optional
                if pnginfo_dict:
                    # Use extracted metadata instead of passed prompts if extraction was successful
                    extracted_positive = pnginfo_dict.get("Positive prompt", positive_prompt)
                    extracted_negative = pnginfo_dict.get("Negative prompt", negative_prompt)
                    
                    # Use extracted values if they exist and are not empty
                    if extracted_positive or extracted_negative:
                        positive_prompt = extracted_positive if extracted_positive else positive_prompt
                        negative_prompt = extracted_negative if extracted_negative else negative_prompt
        except Exception as e:
            print(f"[Telegram Sender] ⚠️ Fallback metadata extraction failed: {e}")
        
        # Build fallback metadata
        parts = []
        
        # Add positive prompt
        if positive_prompt:
            parts.append(positive_prompt)
        
        # Add negative prompt
        if negative_prompt:
            parts.append(f"Negative prompt: {negative_prompt}")
        
        # Try to extract additional parameters from workflow
        if prompt_dict:
            try:
                params = self._extract_parameters_from_workflow(prompt_dict)
                if params:
                    parts.append(params)
            except Exception as e:
                print(f"[Telegram Sender] ⚠️ Parameter extraction failed: {e}")
        
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
            print(f"[Telegram Sender] Error extracting parameters: {e}")
        
        # Ensure all params are strings before joining
        if params:
            params_str = [str(param) for param in params]
            return ", ".join(params_str)
        return ""

    def _extract_loras_from_workflow(self, prompt_dict, extra_pnginfo=None, enable_enhanced=True):
        """Extract list of LoRA names used in workflow using enhanced metadata extraction"""
        loras = []
        
        if not prompt_dict:
            return loras
        
        # Try enhanced metadata extraction first if enabled
        if enable_enhanced:
            try:
                pnginfo_dict = TelegramMetadata.get_metadata(prompt_dict)
                
                if pnginfo_dict and "Lora hashes" in pnginfo_dict:
                    lora_str = pnginfo_dict["Lora hashes"]
                    # Parse "name1: strength1, name2: strength2" format
                    for part in lora_str.split(","):
                        lora_name = part.split(":")[0].strip()
                        if lora_name:
                            loras.append(lora_name.lower().strip())
                    
                    if loras:
                        print(f"[Telegram Sender] ✅ Extracted {len(loras)} LoRAs from metadata")
                        return loras
            except Exception as e:
                print(f"[Telegram Sender] ⚠️ Enhanced metadata extraction failed: {e}")
        
        # Fallback: direct inspection for basic ComfyUI LoRA nodes
        try:
            for node_id, node_data in prompt_dict.items():
                class_type = node_data.get("class_type", "")
                
                if "LoraLoader" in class_type:  # Standard ComfyUI LoRA node
                    inputs = node_data.get("inputs", {})
                    lora_name = inputs.get("lora_name")
                    if lora_name:
                        clean_name = os.path.splitext(os.path.basename(lora_name))[0]
                        clean_name = clean_name.lower().strip()
                        if clean_name not in loras:
                            loras.append(clean_name)
        
        except Exception as e:
            print(f"[Telegram Sender] ⚠️ LoRA fallback extraction failed: {e}")
        
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
                
                # Remove "lora " prefix if present (common format: "lora name:chat_id")
                if lora_key.startswith("lora "):
                    lora_key = lora_key[5:].strip()
                
                chat_id = chat_id.strip()
                
                if lora_key and chat_id:
                    mapping[lora_key] = chat_id
                    print(f"[Telegram Sender] 📋 Loaded LoRA mapping: '{lora_key}' → {chat_id}")
            except Exception as e:
                print(f"[Telegram Sender] ⚠️ Error parsing LoRA mapping line '{line}': {e}")
                continue
        
        return mapping

    def _determine_chat_id(self, default_chat_id, positive_prompt, negative_prompt,
                          enable_nsfw, nsfw_channel_id, unsorted_channel_id,
                          enable_lora_routing, loras_in_workflow):
        """Determine which chat to send to based on content"""
        target_chat_id = None
        
        print(f"[Telegram Sender] 🔀 Routing logic:")
        print(f"  - Default chat_id: {default_chat_id}")
        print(f"  - Enable LoRA routing: {enable_lora_routing}")
        print(f"  - LoRAs in workflow: {loras_in_workflow}")
        print(f"  - Enable NSFW detection: {enable_nsfw}")
        
        # Priority 1: LoRA-based routing (highest priority if enabled and LoRAs found)
        if enable_lora_routing and loras_in_workflow:
            lora_mapping = self._parse_lora_mapping()
            
            if lora_mapping:
                print(f"[Telegram Sender] 📚 Available LoRA mappings: {lora_mapping}")
                
                # Check each LoRA in workflow against mapping
                for lora_in_workflow in loras_in_workflow:
                    for lora_key, mapped_chat_id in lora_mapping.items():
                        # Match if mapping key is substring of actual LoRA name
                        if lora_key in lora_in_workflow:
                            print(f"[Telegram Sender] ✅ LoRA routing: '{lora_in_workflow}' matched key '{lora_key}' → {mapped_chat_id}")
                            target_chat_id = mapped_chat_id
                            break
                    
                    if target_chat_id:
                        break
                
                if not target_chat_id:
                    print(f"[Telegram Sender] ⚠️ No LoRA mapping found for: {loras_in_workflow}")
            else:
                print(f"[Telegram Sender] ⚠️ LoRA routing enabled but no mappings configured")
        
        # Priority 2: NSFW detection (overrides LoRA routing if NSFW found)
        if enable_nsfw and nsfw_channel_id:
            positive_lower = positive_prompt.lower() if positive_prompt else ""
            negative_lower = negative_prompt.lower() if negative_prompt else ""
            
            # If NSFW in positive but not in negative, redirect
            if "nsfw" in positive_lower and "nsfw" not in negative_lower:
                print(f"[Telegram Sender] 🔞 NSFW detected in positive prompt, redirecting to {nsfw_channel_id}")
                target_chat_id = nsfw_channel_id
        
        # Priority 3: Use default chat_id if no routing matched
        if not target_chat_id and default_chat_id:
            print(f"[Telegram Sender] 📌 Using default chat_id: {default_chat_id}")
            target_chat_id = default_chat_id
        
        # Priority 4: Fallback to unsorted channel if no other chat_id
        if not target_chat_id and unsorted_channel_id:
            print(f"[Telegram Sender] 📦 Using unsorted channel: {unsorted_channel_id}")
            target_chat_id = unsorted_channel_id
        
        if target_chat_id:
            print(f"[Telegram Sender] ✅ Final destination: {target_chat_id}")
        else:
            print(f"[Telegram Sender] ❌ No target chat_id determined!")
        
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
                
                print(f"[Telegram Sender] 🔄 Resizing from {width}x{height} to {new_size}")
                
                img = img.resize(new_size, Image.LANCZOS)
                temp_path = os.path.splitext(image_path)[0] + "_resized.jpg"
                img.save(temp_path, "JPEG", quality=85, optimize=True)
                
                return temp_path
        except Exception as e:
            print(f"[Telegram Sender] ⚠️ Resize error: {e}")
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
                        print(f"[Telegram Sender] 🗜️ Compressed to {file_size // 1024}KB (quality: {quality})")
                        return temp_path
                    
                    quality -= 10
                
                print(f"[Telegram Sender] ⚠️ Could not compress below target size")
                return temp_path
        except Exception as e:
            print(f"[Telegram Sender] ⚠️ Compress error: {e}")
            return image_path

    def _send_telegram_request(self, image_path, chat_id, bot_token, 
                               as_document, original_path, retry_count, retry_delay):
        """Send image to Telegram with robust retry logic for connection and timeout issues"""
        # Use semaphore to limit concurrent uploads
        with _upload_semaphore:
            self._send_telegram_request_impl(image_path, chat_id, bot_token, as_document, original_path, retry_count, retry_delay)

    def _send_telegram_request_impl(self, image_path, chat_id, bot_token, 
                                     as_document, original_path, retry_count, retry_delay):
        """Implementation of send_telegram_request with actual upload logic"""
        method = "sendDocument" if as_document else "sendPhoto"
        param_name = "document" if as_document else "photo"
        url = f"https://api.telegram.org/bot{bot_token}/{method}"
        
        # Wait for file to be fully saved (with timeout)
        max_wait = 30
        wait_time = 0
        while not os.path.exists(image_path) and wait_time < max_wait:
            time.sleep(0.1)
            wait_time += 0.1
        
        if not os.path.exists(image_path):
            print(f"[Telegram Sender] ❌ File not found after {max_wait}s: {image_path}")
            return
        
        # Get file size for dynamic timeout calculation
        try:
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            # Estimate timeout: 1 second per MB + 30 seconds base (for slow connections)
            # Minimum 60 seconds, maximum 300 seconds
            estimated_timeout = max(60, min(300, 30 + int(file_size_mb * 1.5)))
            print(f"[Telegram Sender] 📁 File ready: {file_size_mb:.2f}MB (timeout: {estimated_timeout}s)")
        except:
            estimated_timeout = 60
        
        for attempt in range(retry_count):
            try:
                with open(image_path, 'rb') as f:
                    files = {param_name: f}
                    data = {'chat_id': chat_id}
                    
                    # Use estimated timeout based on file size
                    response = requests.post(
                        url, 
                        data=data, 
                        files=files, 
                        timeout=(10, estimated_timeout)  # (connect_timeout, read_timeout)
                    )
                
                if response.ok:
                    file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
                    file_type = "document" if as_document else "photo"
                    print(f"[Telegram Sender] ✅ Sent to {chat_id} as {file_type} ({file_size:.2f}MB)")
                    
                    # Clean up only temporary files, not the original if sent as document
                    if as_document and image_path == original_path:
                        # Don't delete original when sent as document
                        pass
                    else:
                        self._cleanup_files(image_path, original_path if image_path != original_path else None)
                    return
                else:
                    try:
                        error_msg = response.json().get('description', response.text)
                    except:
                        error_msg = response.text
                    print(f"[Telegram Sender] ❌ [{attempt+1}/{retry_count}] Error: {error_msg}")
                    
            except requests.exceptions.ConnectTimeout:
                print(f"[Telegram Sender] ⏱️ [{attempt+1}/{retry_count}] Connection timeout (establishing connection failed)")
                
            except requests.exceptions.ReadTimeout:
                print(f"[Telegram Sender] ⏱️ [{attempt+1}/{retry_count}] Read timeout (server not responding)")
                
            except requests.exceptions.Timeout:
                print(f"[Telegram Sender] ⏱️ [{attempt+1}/{retry_count}] Request timeout")
                
            except (socket.timeout, BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # Socket-level errors
                print(f"[Telegram Sender] 🔌 [{attempt+1}/{retry_count}] Socket error - connection interrupted, retrying...")
                
            except (requests.exceptions.ConnectionError, OSError, ProtocolError) as e:
                # Catch "Connection aborted" and other socket errors
                error_str = str(e)
                if "Connection aborted" in error_str or "The write operation timed out" in error_str:
                    print(f"[Telegram Sender] 🔌 [{attempt+1}/{retry_count}] Connection aborted/write timeout - retrying...")
                elif "Connection reset" in error_str:
                    print(f"[Telegram Sender] 🔌 [{attempt+1}/{retry_count}] Connection reset - retrying...")
                else:
                    print(f"[Telegram Sender] 🔌 [{attempt+1}/{retry_count}] Connection error: {e}")
                
            except Exception as e:
                print(f"[Telegram Sender] ❌ [{attempt+1}/{retry_count}] Exception: {e}")
            
            if attempt < retry_count - 1:
                wait_time = retry_delay + (attempt * 5)  # Linear backoff: delay + (attempt * 5)
                print(f"[Telegram Sender] ⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        print(f"[Telegram Sender] ❌ Failed to send after {retry_count} attempts")
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
                print(f"[Telegram Sender] ⚠️ Could not delete {path}: {e}")

    def _ensure_file_synced(self, file_path):
        """Ensure file is fully written to disk by checking file stability"""
        if not os.path.exists(file_path):
            return
        
        try:
            # Check file size twice with delay to ensure it's not still being written
            size1 = os.path.getsize(file_path)
            time.sleep(0.2)
            size2 = os.path.getsize(file_path)
            
            # If sizes differ, file is still being written
            if size1 != size2:
                time.sleep(0.5)
                return self._ensure_file_synced(file_path)  # Recurse until stable
            
            # On Unix-like systems, sync the file to disk
            if hasattr(os, 'fsync'):
                try:
                    with open(file_path, 'rb') as f:
                        os.fsync(f.fileno())
                except:
                    pass
        except:
            pass


NODE_CLASS_MAPPINGS = {
    "TelegramConfig": TelegramConfig,
    "TelegramSender": TelegramSender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TelegramConfig": "⚙️ Telegram Config",
    "TelegramSender": "📤 Send to Telegram"
}
