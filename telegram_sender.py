import os
import re
import requests
import threading
import time
import json
import folder_paths
from PIL import Image
import numpy as np
import torch

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
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_config"
    OUTPUT_NODE = True
    CATEGORY = "image/telegram"
    DESCRIPTION = "Configure Telegram bot token and default chat ID. Token is stored securely in a config file."

    def save_config(self, bot_token, default_chat_id):
        config = {
            "bot_token": bot_token.strip(),
            "default_chat_id": default_chat_id.strip()
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
                "prompt": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Prompt text to save in metadata"
                }),
                "send_as_document": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Send as file instead of photo (no compression)"
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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "send_to_telegram"
    OUTPUT_NODE = True
    CATEGORY = "image/telegram"
    DESCRIPTION = "Sends generated images to Telegram. Configure bot token using TelegramConfig node first."

    def send_to_telegram(self, images, chat_id="", bot_token_override="", prompt="", 
                        send_as_document=False, max_size=2560, 
                        landscape_max_width=5120, enable_nsfw_detection=False,
                        nsfw_channel_id="", unsorted_channel_id="",
                        retry_count=3, retry_delay=5):
        
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

        # Process each image
        for i, image in enumerate(images):
            try:
                # Convert tensor to PIL Image
                img_np = (255. * image.cpu().numpy()).astype(np.uint8)
                pil_image = Image.fromarray(img_np)
                
                # Save temporarily with metadata
                temp_path = os.path.join(
                    self.output_dir, 
                    f"telegram_temp_{int(time.time())}_{i}.png"
                )
                
                # Add prompt as metadata if provided
                if prompt:
                    try:
                        from PIL.PngImagePlugin import PngInfo
                        metadata = PngInfo()
                        metadata.add_text("parameters", prompt)
                        pil_image.save(temp_path, "PNG", pnginfo=metadata)
                    except Exception as e:
                        print(f"[TelegramSender] Could not save metadata: {e}")
                        pil_image.save(temp_path, "PNG")
                else:
                    pil_image.save(temp_path, "PNG")
                
                # Determine target chat_id
                target_chat_id = self._determine_chat_id(
                    chat_id, prompt, enable_nsfw_detection, 
                    nsfw_channel_id, unsorted_channel_id
                )
                
                if not target_chat_id:
                    print("[TelegramSender] ⚠️ No valid chat_id determined. Skipping image.")
                    self._cleanup_file(temp_path)
                    continue
                
                # Resize if needed
                processed_path = self._resize_image(temp_path, max_size, landscape_max_width)
                
                # Compress if too large and not sending as document
                if not send_as_document:
                    try:
                        if os.path.getsize(processed_path) > 10 * 1024 * 1024:
                            processed_path = self._compress_image(processed_path)
                    except Exception as e:
                        print(f"[TelegramSender] Compression check failed: {e}")
                
                # Send in background thread
                threading.Thread(
                    target=self._send_telegram_request,
                    args=(processed_path, target_chat_id, bot_token, 
                          send_as_document, temp_path, retry_count, retry_delay),
                    daemon=True
                ).start()
                
            except Exception as e:
                print(f"[TelegramSender] ❌ Error processing image {i}: {e}")
                continue
        
        return (images,)

    def _determine_chat_id(self, default_chat_id, prompt, enable_nsfw, 
                          nsfw_channel_id, unsorted_channel_id):
        """Determine which chat to send to based on content"""
        target_chat_id = default_chat_id
        
        # NSFW detection
        if enable_nsfw and nsfw_channel_id and prompt:
            prompt_lower = prompt.lower()
            
            # Check if NSFW is in negative prompt
            if "negative prompt:" in prompt_lower:
                parts = prompt_lower.split("negative prompt:", 1)
                positive = parts[0]
                negative = parts[1] if len(parts) > 1 else ""
                
                # If NSFW in positive but not in negative, redirect
                if "nsfw" in positive and "nsfw" not in negative:
                    print("[TelegramSender] 🔞 NSFW detected, redirecting to NSFW channel")
                    target_chat_id = nsfw_channel_id
            elif "nsfw" in prompt_lower:
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
                    print(f"[TelegramSender] ✅ Sent to {chat_id}")
                    self._cleanup_files(image_path, original_path)
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
        self._cleanup_files(image_path, original_path)

    def _cleanup_files(self, *paths):
        """Clean up temporary files"""
        for path in paths:
            self._cleanup_file(path)

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