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
import folder_paths
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
from datetime import datetime
from urllib3.exceptions import ProtocolError
from server import PromptServer
from aiohttp import web

# --- PART 1: API FOR MIGRATION ---
@PromptServer.instance.routes.get("/telegram_sender/get_legacy_config")
async def get_legacy_config(request):
    import os
    import json
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    config_file = os.path.join(config_dir, "telegram_config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": str(e)})
    return web.json_response({})

# Global queue (SEMAPHORE = 2)
_upload_semaphore = threading.Semaphore(2)
_upload_lock = threading.Lock()

# --- PART 2: IMPORT SETTINGS ---
try:
    from .telegram_settings import get_config
except:
    def get_config(): return {}

# Import metadata wrapper
try:
    from .telegram_metadata import TelegramMetadata
except:
    class TelegramMetadata:
        @staticmethod
        def get_metadata(p, n=False): return {}
        @staticmethod
        def get_parameters_str(p): return ""

# --- PART 3: OLD NSFW LOGIC ---
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
NSFW_TRIGGERS_FILE = os.path.join(ASSETS_DIR , "nsfw_triggers.json")
_nsfw_triggers_cache = None
_nsfw_triggers_mtime = None

def load_nsfw_triggers():
    global _nsfw_triggers_cache, _nsfw_triggers_mtime
    try:
        if not os.path.exists(NSFW_TRIGGERS_FILE): return None
        mtime = os.path.getmtime(NSFW_TRIGGERS_FILE)
        if _nsfw_triggers_cache is not None and _nsfw_triggers_mtime == mtime: return _nsfw_triggers_cache
        with open(NSFW_TRIGGERS_FILE, 'r', encoding='utf-8') as f: data = json.load(f)
        
        keywords = []
        regex = []
        negative_keywords = []
        negative_regex = []
        if isinstance(data, list):
            keywords = [str(x).lower() for x in data]
        elif isinstance(data, dict):
            kw = data.get('nsfw_triggers') or data.get('keywords') or []
            rx = data.get('regex') or []
            nkw = data.get('negative_keywords') or []
            nrx = data.get('negative_regex') or []
            keywords = [str(x).lower() for x in kw]
            regex = [str(x) for x in rx]
            negative_keywords = [str(x).lower() for x in nkw]
            negative_regex = [str(x) for x in nrx]
        result = {'keywords': keywords, 'regex': regex, 'negative_keywords': negative_keywords, 'negative_regex': negative_regex}
        _nsfw_triggers_cache = result
        _nsfw_triggers_mtime = mtime
        return result
    except: return None

class TelegramConfig:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {}}
    RETURN_TYPES = ()
    FUNCTION = "save_config"
    OUTPUT_NODE = True
    CATEGORY = "image/telegram"
    def save_config(self): return ()

class TelegramSender:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        
    @classmethod
    def INPUT_TYPES(cls):
        c = get_config()
        return {
            "required": { "images": ("IMAGE",) },
            "optional": {
                "chat_id": ("STRING", {"default": c.get("default_chat_id", ""), "multiline": False}),
                "bot_token_override": ("STRING", {"default": "", "multiline": False}),
                "positive_prompt": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "send_as_document": ("BOOLEAN", {"default": False}),
                "max_size": ("INT", {"default": 2560}),
                "landscape_max_width": ("INT", {"default": 5120}),
                "enable_nsfw_detection": ("BOOLEAN", {"default": False}),
                "nsfw_channel_id": ("STRING", {"default": c.get("nsfw_channel_id", ""), "multiline": False}),
                "unsorted_channel_id": ("STRING", {"default": c.get("unsorted_channel_id", ""), "multiline": False}),
                "enable_lora_routing": ("BOOLEAN", {"default": True}),
                "retry_count": ("INT", {"default": 3}),
                "retry_delay": ("INT", {"default": 5}),
                "enable_enhanced_metadata": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "telegram_%date%_%model%_%seed%"}),
                "subdirectory_name": ("STRING", {"default": ""}),
                "debug_metadata": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "send_to_telegram"
    OUTPUT_NODE = True
    CATEGORY = "image/telegram"

    # --- HELPER: Manual Config Load ---
    def _get_full_config(self):
        config = get_config()
        if not config.get("bot_token"):
            try:
                path = os.path.join(folder_paths.base_path, "user", "default", "comfy.settings.json")
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if not config.get("bot_token"): config["bot_token"] = data.get("Telegram.BotToken", "")
                        if not config.get("default_chat_id"): config["default_chat_id"] = data.get("Telegram.DefaultChatId", "")
                        if not config.get("lora_mapping"): config["lora_mapping"] = data.get("Telegram.LoraMapping", "")
                        if not config.get("nsfw_channel_id"): config["nsfw_channel_id"] = data.get("Telegram.NSFWChannelId", "")
                        if not config.get("unsorted_channel_id"): config["unsorted_channel_id"] = data.get("Telegram.UnsortedChannelId", "")
            except: pass
        return config

    def send_to_telegram(self, images, chat_id="", bot_token_override="", 
                        positive_prompt="", negative_prompt="",
                        send_as_document=False, max_size=2560, 
                        landscape_max_width=5120, enable_nsfw_detection=False,
                        nsfw_channel_id="", unsorted_channel_id="",
                        enable_lora_routing=True, retry_count=3, retry_delay=5,
                        enable_enhanced_metadata=True, filename_prefix="",
                        subdirectory_name="", debug_metadata=False,
                        prompt=None, extra_pnginfo=None):
        
        full_config = self._get_full_config()
        bot_token = bot_token_override.strip() if bot_token_override else full_config.get("bot_token", "")
        
        if not bot_token:
            print("[Telegram Sender] ⚠️ No bot token configured. Images will be saved locally.")
            bot_token = ""
        
        if not chat_id: chat_id = full_config.get("default_chat_id", "")
        if not nsfw_channel_id: nsfw_channel_id = full_config.get("nsfw_channel_id", "")
        if not unsorted_channel_id: unsorted_channel_id = full_config.get("unsorted_channel_id", "")

        metadata_text = self._build_metadata_text(positive_prompt, negative_prompt, prompt, extra_pnginfo, enable_enhanced_metadata)
        loras_in_workflow = self._extract_loras_from_workflow(prompt, extra_pnginfo, enable_enhanced_metadata) if prompt else []

        for i, image in enumerate(images):
            try:
                img_np = (255. * image.cpu().numpy()).astype(np.uint8)
                pil_image = Image.fromarray(img_np)
                
                pnginfo_dict = {}
                if enable_enhanced_metadata and prompt:
                    try: pnginfo_dict = TelegramMetadata.get_metadata(prompt)
                    except: pass
                
                formatted_filename = self._format_filename(filename_prefix.strip(), pnginfo_dict)
                if not formatted_filename: formatted_filename = f"telegram_temp_{int(time.time())}_{i}"
                
                save_dir = self.output_dir
                if subdirectory_name.strip():
                    formatted_subdir = self._format_filename(subdirectory_name.strip(), pnginfo_dict)
                    save_dir = os.path.join(self.output_dir, formatted_subdir)
                    os.makedirs(save_dir, exist_ok=True)
                
                temp_path = os.path.join(save_dir, f"{formatted_filename}.png")
                
                pnginfo = PngInfo()
                if enable_enhanced_metadata and pnginfo_dict:
                    try:
                        params_str = TelegramMetadata.get_parameters_str(pnginfo_dict)
                        if params_str: pnginfo.add_text("parameters", params_str)
                        for key, value in pnginfo_dict.items():
                            if value and isinstance(value, (str, int, float)):
                                pnginfo.add_text(key, str(value))
                    except: pass
                elif metadata_text:
                    pnginfo.add_text("parameters", metadata_text)
                
                if prompt: pnginfo.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo:
                    for k, v in extra_pnginfo.items():
                        pnginfo.add_text(k, json.dumps(v) if isinstance(v, dict) else str(v))
                
                pil_image.save(temp_path, "PNG", pnginfo=pnginfo)
                
                target_chat_id = self._determine_chat_id(
                    chat_id, positive_prompt, negative_prompt, 
                    enable_nsfw_detection, nsfw_channel_id, unsorted_channel_id,
                    enable_lora_routing, loras_in_workflow, full_config
                )
                
                if not target_chat_id:
                    print("[Telegram Sender] ⚠️ No valid chat_id determined. Skipping image.")
                    self._cleanup_file(temp_path)
                    continue
                
                # --- STEP 1: SEND PHOTO ---
                photo_path = self._resize_image(temp_path, max_size, landscape_max_width)
                if os.path.getsize(photo_path) > 10 * 1024 * 1024:
                    photo_path = self._compress_image(photo_path)
                
                self._ensure_file_synced(photo_path)
                
                if bot_token and target_chat_id:
                    threading.Thread(
                        target=self._send_telegram_request,
                        args=(photo_path, target_chat_id, bot_token, False, temp_path, retry_count, retry_delay),
                        daemon=True
                    ).start()
                else:
                    print(f"[Telegram Sender] 💾 Saved image to {photo_path} (not sent)")
                
                # --- STEP 2: SEND DOCUMENT (AFTER PHOTO) ---
                if send_as_document:
                    self._ensure_file_synced(temp_path)
                    time.sleep(2) # Delay
                    if bot_token and target_chat_id:
                        print(f"[Telegram Sender] 📄 Sending as Document...") # LOG
                        threading.Thread(
                            target=self._send_telegram_request,
                            args=(temp_path, target_chat_id, bot_token, True, temp_path, retry_count, retry_delay),
                            daemon=True
                        ).start()
                    else:
                        print(f"[Telegram Sender] 💾 Saved original PNG to {temp_path} (not sent)")

            except Exception as e:
                print(f"[Telegram Sender] ❌ Error processing image {i}: {e}")
        
        return (images,)

    def _build_metadata_text(self, positive_prompt, negative_prompt, prompt_dict, extra_pnginfo, enable_enhanced=True):
        parts = []
        if positive_prompt: parts.append(positive_prompt)
        if negative_prompt: parts.append(f"Negative prompt: {negative_prompt}")
        if prompt_dict:
            try:
                params = self._extract_parameters_from_workflow(prompt_dict)
                if params: parts.append(params)
            except: pass
        return "\n".join(parts)

    def _extract_parameters_from_workflow(self, prompt_dict):
        params = []
        try:
            for node in prompt_dict.values():
                class_type = node.get("class_type", "").lower()
                inputs = node.get("inputs", {})
                if "sampler" in class_type:
                    if "steps" in inputs: params.append(f"Steps: {inputs['steps']}")
                    if "sampler_name" in inputs: params.append(f"Sampler: {inputs['sampler_name']}")
                    if "cfg" in inputs: params.append(f"CFG scale: {inputs['cfg']}")
                    if "seed" in inputs: params.append(f"Seed: {inputs['seed']}")
                    if "noise_seed" in inputs: params.append(f"Seed: {inputs['noise_seed']}")
                if "checkpoint" in class_type:
                    name = inputs.get("ckpt_name", inputs.get("model_name"))
                    if name: params.append(f"Model: {name}")
                if "lora" in class_type:
                    name = inputs.get("lora_name")
                    if name: params.append(f"Lora: {name}")
        except: pass
        return ", ".join(params)

    # --- ORIGINAL LORA EXTRACTION (CRITICAL FOR ROUTING) ---
    def _extract_loras_from_workflow(self, prompt_dict, extra_pnginfo=None, enable_enhanced=True):
        loras = []
        if not prompt_dict: return loras
        if enable_enhanced:
            try:
                pnginfo_dict = TelegramMetadata.get_metadata(prompt_dict)
                if pnginfo_dict and "Lora hashes" in pnginfo_dict:
                    for part in pnginfo_dict["Lora hashes"].split(","):
                        lora_name = part.split(":")[0].strip()
                        if lora_name: loras.append(lora_name.lower().strip())
                    if loras: return loras
            except: pass
        try:
            for node in prompt_dict.values():
                class_type = node.get("class_type", "")
                if "LoraLoader" in class_type:
                    name = node.get("inputs", {}).get("lora_name", "")
                    if name:
                        clean = os.path.splitext(os.path.basename(name))[0].lower().strip()
                        if clean not in loras: loras.append(clean)
        except: pass
        return loras

    # --- NEW REGEX PARSER (ONLY CHANGE TO SUPPORT NEW CONFIG FORMAT) ---
    def _parse_lora_mapping(self, full_config):
        mapping_str = str(full_config.get("lora_mapping", ""))
        val = full_config.get("lora_mapping")
        if isinstance(val, list): mapping_str = "\n".join(str(x) for x in val)
        if not mapping_str: return {}
        
        mapping = {}
        pattern = r"(?:lora\s+|lora_|_)?([^:\n\r]+?)\s*:\s*(-?\d+)"
        matches = re.finditer(pattern, mapping_str, re.IGNORECASE)
        for match in matches:
            try:
                lora_key = match.group(1).strip().lower()
                chat_id = match.group(2).strip()
                lora_key = lora_key.strip(" _")
                if lora_key and chat_id: 
                    mapping[lora_key] = chat_id
            except: pass
        return mapping

    def _determine_chat_id(self, default_chat_id, positive_prompt, negative_prompt,
                          enable_nsfw, nsfw_channel_id, unsorted_channel_id,
                          enable_lora_routing, loras_in_workflow, full_config):
        target_chat_id = None
        
        if enable_lora_routing and loras_in_workflow:
            lora_mapping = self._parse_lora_mapping(full_config)
            for lora_in_workflow in loras_in_workflow:
                for lora_key, mapped_chat_id in lora_mapping.items():
                    if lora_key in lora_in_workflow:
                        print(f"[Telegram Sender] ✅ LoRA routing: '{lora_in_workflow}' matched key '{lora_key}' → {mapped_chat_id}")
                        target_chat_id = mapped_chat_id
                        break
                if target_chat_id: break
        
        if not target_chat_id and enable_nsfw and nsfw_channel_id:
            positive_lower = (positive_prompt or "").lower()
            triggers = load_nsfw_triggers()
            nsfw_found = False
            if triggers:
                for kw in triggers.get('keywords', []):
                    if kw in positive_lower: nsfw_found = True; break
                if not nsfw_found:
                    for pattern in triggers.get('regex', []):
                        if re.search(pattern, positive_prompt or "", re.IGNORECASE): nsfw_found = True; break
                if not nsfw_found and "nsfw" in positive_lower: nsfw_found = True
            else:
                if "nsfw" in positive_lower: nsfw_found = True
            if nsfw_found:
                print(f"[Telegram Sender] 🔞 NSFW detected, redirecting to {nsfw_channel_id}")
                target_chat_id = nsfw_channel_id
        
        if not target_chat_id: target_chat_id = default_chat_id
        if not target_chat_id: target_chat_id = unsorted_channel_id
        
        return target_chat_id

    def _resize_image(self, image_path, max_size, landscape_max_width):
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                if w >= h:
                    if w <= landscape_max_width: return image_path
                    max_val = landscape_max_width
                else:
                    if max(w, h) <= max_size: return image_path
                    max_val = max_size
                ratio = max_val / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                temp_path = os.path.splitext(image_path)[0] + "_resized.jpg"
                img.save(temp_path, "JPEG", quality=85, optimize=True)
                return temp_path
        except: return image_path

    def _compress_image(self, image_path, target_size=10 * 1024 * 1024):
        temp_path = os.path.splitext(image_path)[0] + "_compressed.jpg"
        quality = 85
        try:
            with Image.open(image_path) as img:
                while quality >= 30:
                    img.save(temp_path, "JPEG", quality=quality, optimize=True)
                    if os.path.getsize(temp_path) <= target_size: return temp_path
                    quality -= 10
                return temp_path
        except: return image_path

    # --- ORIGINAL SEND LOGIC + TIMEOUT FIX ---
    def _send_telegram_request(self, image_path, chat_id, bot_token, 
                               as_document, original_path, retry_count, retry_delay):
        with _upload_semaphore:
            method = "sendDocument" if as_document else "sendPhoto"
            param_name = "document" if as_document else "photo"
            url = f"https://api.telegram.org/bot{bot_token}/{method}"
            
            wait_time = 0
            while not os.path.exists(image_path) and wait_time < 30:
                time.sleep(0.1); wait_time += 0.1
            
            # TIMEOUT FIX: Increase timeout for slow internet
            try:
                file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
                timeout_val = max(60, min(600, 60 + int(file_size_mb * 10))) # Up to 10 min
            except: timeout_val = 60

            for attempt in range(retry_count):
                try:
                    with open(image_path, 'rb') as f:
                        response = requests.post(url, data={'chat_id': chat_id}, files={param_name: f}, timeout=timeout_val)
                    if response.ok:
                        # LOG with Type
                        ftype = "Document" if as_document else "Photo"
                        print(f"[Telegram Sender] ✅ Sent to {chat_id} ({ftype})")
                        if not as_document and image_path != original_path:
                            try: os.remove(image_path)
                            except: pass
                        return
                    else:
                        print(f"[Telegram Sender] ❌ Error: {response.text}")
                except Exception as e:
                    print(f"[Telegram Sender] ❌ Error: {e}")
                time.sleep(retry_delay)

    def _cleanup_file(self, path):
        if path and os.path.exists(path) and "telegram_temp" in path:
            try: os.remove(path)
            except: pass

    def _ensure_file_synced(self, file_path):
        if not os.path.exists(file_path): return
        try:
            if hasattr(os, 'fsync'):
                with open(file_path, 'rb') as f: os.fsync(f.fileno())
        except: pass

    def _format_filename(self, filename_template, pnginfo_dict):
        if "%" not in filename_template: return filename_template
        now = datetime.now()
        date_table = {"yyyy": f"{now.year}", "MM": f"{now.month:02d}", "dd": f"{now.day:02d}", "hh": f"{now.hour:02d}", "mm": f"{now.minute:02d}", "ss": f"{now.second:02d}"}
        
        seed = pnginfo_dict.get("Seed", pnginfo_dict.get("seed", ""))
        model = pnginfo_dict.get("Model", pnginfo_dict.get("model", ""))
        if model: model = os.path.splitext(os.path.basename(str(model)))[0]
        
        filename_template = filename_template.replace("%seed%", str(seed)).replace("%model%", str(model))
        filename_template = filename_template.replace("%width%", str(pnginfo_dict.get("Width", ""))).replace("%height%", str(pnginfo_dict.get("Height", "")))
        
        pattern_format = re.compile(r"(%[^%]+%)")
        segments = pattern_format.findall(filename_template)
        for segment in segments:
            parts = segment.strip("%").split(":")
            key = parts[0]
            if key == "date":
                fmt = parts[1] if len(parts) > 1 else "yyyyMMddhhmmss"
                for k, v in date_table.items(): fmt = fmt.replace(k, v)
                filename_template = filename_template.replace(segment, fmt)
            elif key in ["pprompt", "nprompt"]:
                k_lookup = "Positive prompt" if key == "pprompt" else "Negative prompt"
                txt = pnginfo_dict.get(k_lookup, pnginfo_dict.get(key, ""))
                length = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                if txt:
                    txt = str(txt).replace("\n", " ").replace("/", "")
                    filename_template = filename_template.replace(segment, txt[:length].strip() if length else txt.strip())
        return filename_template

# Mappings
NODE_CLASS_MAPPINGS = { "TelegramConfig": TelegramConfig, "TelegramSender": TelegramSender }
NODE_DISPLAY_NAME_MAPPINGS = { "TelegramConfig": "⚙️ Telegram Config", "TelegramSender": "📤 Send to Telegram" }