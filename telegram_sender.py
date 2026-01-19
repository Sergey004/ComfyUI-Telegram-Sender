import os
import re
import requests
import threading
import time
import json
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
sys.path.append("../../")
import folder_paths # type: ignore
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
from datetime import datetime
from server import PromptServer # type: ignore
from aiohttp import web

# --- API FOR MIGRATION ---
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

# Global vars
_upload_semaphore = threading.Semaphore(1)
_upload_lock = threading.Lock()
_session_lock = threading.Lock()
_global_session = None

def get_global_session():
    """Global session with aggressive Keep-Alive for slow connections"""
    global _global_session
    if _global_session is None:
        with _session_lock:
            if _global_session is None:
                retry_strategy = Retry(
                    total=10, 
                    backoff_factor=1, 
                    status_forcelist=[408, 429, 500, 502, 503, 504],
                    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
                    raise_on_status=False,
                    raise_on_redirect=False,
                    respect_retry_after_header=True
                )
                adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20, pool_block=False)
                _global_session = requests.Session()
                _global_session.mount("http://", adapter)
                _global_session.mount("https://", adapter)
                _global_session.headers.update({'Connection': 'keep-alive', 'Keep-Alive': 'timeout=600, max=10000'})
    return _global_session

# Import Settings safely
try:
    from .telegram_settings import get_config
except:
    def get_config(): return {}

# Fallback Metadata
try:
    from .telegram_metadata import TelegramMetadata
except:
    class TelegramMetadata:
        @staticmethod
        def get_metadata(p, n=False): return {}
        @staticmethod
        def get_parameters_str(p): return ""

# --- RESTORED: NSFW TRIGGERS LOGIC ---
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
NSFW_TRIGGERS_FILE = os.path.join(CONFIG_DIR, "nsfw_triggers.json")
_nsfw_triggers_cache = None
_nsfw_triggers_mtime = None

def load_nsfw_triggers():
    global _nsfw_triggers_cache, _nsfw_triggers_mtime
    try:
        if not os.path.exists(NSFW_TRIGGERS_FILE):
            return None
        mtime = os.path.getmtime(NSFW_TRIGGERS_FILE)
        if _nsfw_triggers_cache is not None and _nsfw_triggers_mtime == mtime:
            return _nsfw_triggers_cache
        with open(NSFW_TRIGGERS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        keywords = []
        regex = []
        negative_keywords = []
        negative_regex = []
        if isinstance(data, list):
            keywords = [str(x).lower() for x in data]
        elif isinstance(data, dict):
            kw = data.get('nsfw_triggers') or data.get('keywords') or data.get('include') or []
            rx = data.get('regex') or []
            nkw = data.get('negative_keywords') or data.get('exclude') or []
            nrx = data.get('negative_regex') or []
            keywords = [str(x).lower() for x in kw]
            regex = [str(x) for x in rx]
            negative_keywords = [str(x).lower() for x in nkw]
            negative_regex = [str(x) for x in nrx]
        
        result = {
            'keywords': keywords, 'regex': regex,
            'negative_keywords': negative_keywords, 'negative_regex': negative_regex,
        }
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

    def _manual_find_token(self):
        try:
            path = os.path.join(folder_paths.base_path, "user", "default", "comfy.settings.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    token = data.get("Telegram.BotToken", "")
                    if token: return token
        except: pass
        return ""

    def send_to_telegram(self, images, chat_id="", bot_token_override="", 
                        positive_prompt="", negative_prompt="",
                        send_as_document=False, max_size=2560, 
                        landscape_max_width=5120, enable_nsfw_detection=False,
                        nsfw_channel_id="", unsorted_channel_id="",
                        enable_lora_routing=True, retry_count=3, retry_delay=5,
                        enable_enhanced_metadata=True, filename_prefix="",
                        subdirectory_name="", debug_metadata=False,
                        prompt=None, extra_pnginfo=None):
        
        # Log params
        print(f"[Telegram Sender] 🔧 Parameters: retry_count={retry_count}, retry_delay={retry_delay}")

        # 1. Config & Token
        config = get_config()
        bot_token = bot_token_override.strip() if bot_token_override else config.get("bot_token", "")
        if not bot_token: bot_token = self._manual_find_token()
        if not chat_id: chat_id = config.get("default_chat_id", "")
        
        # Metadata logic (Full)
        metadata_text = self._build_metadata_text(positive_prompt, negative_prompt, prompt, extra_pnginfo, enable_enhanced_metadata)
        loras_in_workflow = self._extract_loras_from_workflow(prompt, extra_pnginfo, enable_enhanced_metadata) if prompt else []
        
        # Extracted params for filename
        extracted_params = {}
        if prompt:
             extracted_params = self._extract_parameters_dict(prompt)

        for i, image in enumerate(images):
            try:
                # Save Temp
                img_np = (255. * image.cpu().numpy()).astype(np.uint8)
                pil_image = Image.fromarray(img_np)
                
                # Metadata dictionary
                pnginfo_dict = extracted_params.copy()
                if enable_enhanced_metadata and prompt:
                    try: 
                        meta = TelegramMetadata.get_metadata(prompt)
                        if meta: pnginfo_dict.update(meta)
                    except Exception as e: 
                        if debug_metadata: print(f"Metadata Error: {e}")
                
                # Fill missing prompts for filename regex
                if "Positive prompt" not in pnginfo_dict: pnginfo_dict["Positive prompt"] = positive_prompt
                if "Negative prompt" not in pnginfo_dict: pnginfo_dict["Negative prompt"] = negative_prompt

                # Filename
                formatted_filename = self._format_filename(filename_prefix.strip(), pnginfo_dict)
                if not formatted_filename: formatted_filename = f"telegram_{int(time.time())}_{i}"
                
                # Subdir
                save_dir = self.output_dir
                if subdirectory_name.strip():
                    formatted_subdir = self._format_filename(subdirectory_name.strip(), pnginfo_dict)
                    save_dir = os.path.join(self.output_dir, formatted_subdir)
                    os.makedirs(save_dir, exist_ok=True)
                
                temp_path = os.path.join(save_dir, f"{formatted_filename}.png")
                
                # PNG Info preparation
                pnginfo = PngInfo()
                
                # 1. Enhanced metadata (A1111 style)
                if enable_enhanced_metadata and pnginfo_dict:
                    try:
                        params_str = TelegramMetadata.get_parameters_str(pnginfo_dict)
                        if params_str: pnginfo.add_text("parameters", params_str)
                        for key, value in pnginfo_dict.items():
                            if value and isinstance(value, (str, int, float)):
                                pnginfo.add_text(key, str(value))
                        print(f"[Telegram Sender] ✅ Enhanced metadata embedded")
                    except Exception as e:
                        print(f"[Telegram Sender] ⚠️ Enhanced metadata error: {e}")
                elif metadata_text:
                    pnginfo.add_text("parameters", metadata_text)
                
                # 2. WORKFLOW EMBEDDING (RESTORED EXACTLY FROM ORIGINAL)
                if prompt is not None:
                    pnginfo.add_text("prompt", json.dumps(prompt))
                
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        pnginfo.add_text(key, json.dumps(value) if isinstance(value, dict) else str(value))

                # Save to disk
                pil_image.save(temp_path, "PNG", pnginfo=pnginfo)
                
                # Routing
                target_chat_id = self._determine_chat_id(
                    chat_id, positive_prompt, negative_prompt, 
                    enable_nsfw_detection, nsfw_channel_id, unsorted_channel_id,
                    enable_lora_routing, loras_in_workflow
                )
                
                # SENDING
                if bot_token and target_chat_id:
                    # Resize / Compress
                    photo_path = self._resize_image(temp_path, max_size, landscape_max_width)
                    if os.path.getsize(photo_path) > 10 * 1024 * 1024:
                        photo_path = self._compress_image(photo_path)
                    
                    self._ensure_file_synced(photo_path)

                    threading.Thread(
                        target=self._send_telegram_request,
                        args=(photo_path, target_chat_id, bot_token, False, temp_path, retry_count, retry_delay),
                        daemon=True
                    ).start()
                    
                    if send_as_document:
                        print(f"[Telegram Sender] 📄 Sending as document in addition to photo...")
                        time.sleep(2)
                        # Ensure temp_path (original PNG with metadata) is synced
                        self._ensure_file_synced(temp_path)
                        threading.Thread(
                            target=self._send_telegram_request,
                            args=(temp_path, target_chat_id, bot_token, True, temp_path, retry_count, retry_delay),
                            daemon=True
                        ).start()
                    else:
                        print(f"[Telegram Sender] 📷 Sending as photo only")
                else:
                    print(f"[Telegram Sender] 💾 Saved locally to {temp_path} (No Token/Chat)")

            except Exception as e:
                print(f"[Telegram Sender] ❌ Error: {e}")
        
        return (images,)

    # --- RESTORED: Parameter Extraction ---
    def _extract_parameters_dict(self, prompt_dict):
        """Deep extraction of parameters for filenames/metadata"""
        params = {}
        try:
            for node in prompt_dict.values():
                class_type = node.get("class_type", "").lower()
                inputs = node.get("inputs", {})
                
                if "sampler" in class_type:
                    if "seed" in inputs: params["Seed"] = inputs["seed"]
                    if "noise_seed" in inputs: params["Seed"] = inputs["noise_seed"]
                    if "steps" in inputs: params["Steps"] = inputs["steps"]
                    if "cfg" in inputs: params["CFG scale"] = inputs["cfg"]
                    if "sampler_name" in inputs: params["Sampler"] = inputs["sampler_name"]
                    if "scheduler" in inputs: params["Schedule"] = inputs["scheduler"]
                    if "denoise" in inputs: params["Denoise"] = inputs["denoise"]
                
                if "checkpoint" in class_type or "loader" in class_type:
                    if "ckpt_name" in inputs: params["Model"] = inputs["ckpt_name"]
                    if "model_name" in inputs: params["Model"] = inputs["model_name"]
                
                if "vae" in class_type and "loader" in class_type:
                    if "vae_name" in inputs: params["VAE"] = inputs["vae_name"]
        except: pass
        return params

    def _extract_parameters_from_workflow(self, prompt_dict):
        params = []
        d = self._extract_parameters_dict(prompt_dict)
        if "Steps" in d: params.append(f"Steps: {d['Steps']}")
        if "Sampler" in d: params.append(f"Sampler: {d['Sampler']}")
        if "CFG scale" in d: params.append(f"CFG scale: {d['CFG scale']}")
        if "Seed" in d: params.append(f"Seed: {d['Seed']}")
        if "Model" in d: params.append(f"Model: {d['Model']}")
        if "VAE" in d: params.append(f"VAE: {d['VAE']}")
        return ", ".join(params)

    # --- RESTORED: Filename Formatting ---
    def _format_filename(self, filename_template, pnginfo_dict):
        if "%" not in filename_template: return filename_template
            
        now = datetime.now()
        date_table = {
            "yyyy": f"{now.year}", "MM": f"{now.month:02d}", "dd": f"{now.day:02d}",
            "hh": f"{now.hour:02d}", "mm": f"{now.minute:02d}", "ss": f"{now.second:02d}",
        }
        
        seed = pnginfo_dict.get("Seed", pnginfo_dict.get("seed", ""))
        model = pnginfo_dict.get("Model", pnginfo_dict.get("model", ""))
        if model: model = os.path.splitext(os.path.basename(str(model)))[0]
        
        filename_template = filename_template.replace("%seed%", str(seed))
        filename_template = filename_template.replace("%model%", str(model))
        filename_template = filename_template.replace("%width%", str(pnginfo_dict.get("Width", "")))
        filename_template = filename_template.replace("%height%", str(pnginfo_dict.get("Height", "")))
        
        pattern_format = re.compile(r"(%[^%]+%)")
        segments = pattern_format.findall(filename_template)
        
        for segment in segments:
            parts = segment.strip("%").split(":")
            key = parts[0]

            if key == "date":
                date_format = parts[1] if len(parts) > 1 else "yyyyMMddhhmmss"
                for k, v in date_table.items():
                    date_format = date_format.replace(k, v)
                filename_template = filename_template.replace(segment, date_format)

            elif key == "pprompt" or key == "nprompt":
                prompt_key = "Positive prompt" if key == "pprompt" else "Negative prompt"
                prompt_text = pnginfo_dict.get(prompt_key, pnginfo_dict.get(key, ""))
                length = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                if prompt_text:
                    prompt_text = str(prompt_text).replace("\n", " ").replace("/", "")
                    filename_template = filename_template.replace(segment, prompt_text[:length].strip() if length else prompt_text.strip())
        
        return filename_template

    def _parse_lora_mapping(self):
        """REGEX PARSER"""
        config = get_config()
        mapping_str = config.get("lora_mapping", "")
        if not mapping_str: return {}
        mapping = {}
        pattern = r"(?:lora\s+|lora_|_)?([^:\n\r]+?)\s*:\s*(-?\d+)"
        matches = re.finditer(pattern, mapping_str, re.IGNORECASE)
        for match in matches:
            try:
                lora_key = match.group(1).strip().lower()
                chat_id = match.group(2).strip()
                lora_key = lora_key.strip(" _")
                if lora_key and chat_id: mapping[lora_key] = chat_id
            except: pass
        if mapping: print(f"[Telegram Sender] 📋 Parsed {len(mapping)} LoRA rules via Regex")
        return mapping

    def _determine_chat_id(self, default_chat_id, positive_prompt, negative_prompt,
                          enable_nsfw, nsfw_channel_id, unsorted_channel_id,
                          enable_lora_routing, loras_in_workflow):
        target_chat_id = None
        
        # 1. LoRA
        if enable_lora_routing and loras_in_workflow:
            lora_mapping = self._parse_lora_mapping()
            for lora_in_workflow in loras_in_workflow:
                for lora_key, mapped_chat_id in lora_mapping.items():
                    if lora_key in lora_in_workflow:
                        print(f"[Telegram Sender] ✅ Routing: '{lora_in_workflow}' -> {mapped_chat_id}")
                        target_chat_id = mapped_chat_id
                        break
                if target_chat_id: break
        
        # 2. NSFW (RESTORED FULL LOGIC)
        if not target_chat_id and enable_nsfw and nsfw_channel_id:
            positive_lower = (positive_prompt or "").lower()
            negative_lower = (negative_prompt or "").lower()
            triggers = load_nsfw_triggers()
            nsfw_found = False
            
            if triggers:
                for kw in triggers.get('keywords', []):
                    if kw and kw in positive_lower: nsfw_found = True; break
                if not nsfw_found:
                    for pattern in triggers.get('regex', []):
                        if re.search(pattern, positive_prompt or "", re.IGNORECASE):
                            nsfw_found = True; break
                if not nsfw_found and "nsfw" in positive_lower: nsfw_found = True
            else:
                if "nsfw" in positive_lower: nsfw_found = True

            if nsfw_found:
                print(f"[Telegram Sender] 🔞 NSFW -> {nsfw_channel_id}")
                target_chat_id = nsfw_channel_id

        # 3. Default / Unsorted
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
                print(f"[Telegram Sender] 🔄 Resizing {w}x{h} -> {new_size}")
                img = img.resize(new_size, Image.LANCZOS)
                temp_path = os.path.splitext(image_path)[0] + "_resized.jpg"
                img.save(temp_path, "JPEG", quality=85, optimize=True)
                return temp_path
        except Exception as e:
            print(f"[Telegram Sender] ⚠️ Resize error: {e}")
            return image_path

    # --- RESTORED: Loop Compression ---
    def _compress_image(self, image_path, target_size=10 * 1024 * 1024):
        temp_path = os.path.splitext(image_path)[0] + "_compressed.jpg"
        quality = 85
        try:
            with Image.open(image_path) as img:
                while quality >= 30:
                    img.save(temp_path, "JPEG", quality=quality, optimize=True)
                    if os.path.getsize(temp_path) <= target_size:
                        print(f"[Telegram Sender] 🗜️ Compressed (Q={quality})")
                        return temp_path
                    quality -= 10
                return temp_path
        except: return image_path

    def _send_telegram_request(self, image_path, chat_id, bot_token, as_document, original_path, retry_count, retry_delay):
        """SLOW INTERNET EDITION: Massive timeouts"""
        with _upload_semaphore:
            method = "sendDocument" if as_document else "sendPhoto"
            param_name = "document" if as_document else "photo"
            url = f"https://api.telegram.org/bot{bot_token}/{method}"
            
            # Wait for file
            wait_time = 0
            while not os.path.exists(image_path) and wait_time < 30:
                time.sleep(0.1)
                wait_time += 0.1
            
            # Timeout Calc
            try:
                file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
                estimated_timeout = max(300, min(3600, 300 + int(file_size_mb * 60)))
                print(f"[Telegram Sender] 🐌 Slow-Mode Upload: {file_size_mb:.2f}MB. Timeout: {estimated_timeout}s")
            except:
                estimated_timeout = 300

            session = get_global_session()
            
            for attempt in range(retry_count):
                try:
                    with open(image_path, 'rb') as f:
                        files = {param_name: f}
                        data = {'chat_id': chat_id}
                        response = session.post(url, data=data, files=files, timeout=(30, estimated_timeout), stream=True)
                    
                    if response.ok:
                        print(f"[Telegram Sender] ✅ Sent to {chat_id}")
                        if not as_document and image_path != original_path:
                            try: os.remove(image_path)
                            except: pass
                        return
                    else:
                        print(f"[Telegram Sender] ❌ Error: {response.text}")

                except requests.exceptions.ConnectTimeout:
                    print(f"[Telegram Sender] ⏱️ Connection timeout")
                except requests.exceptions.ReadTimeout:
                    print(f"[Telegram Sender] 🐌 Upload too slow (ReadTimeout), retrying...")
                except Exception as e:
                    print(f"[Telegram Sender] ❌ Exception: {e}")
                
                if attempt < retry_count - 1:
                    print(f"[Telegram Sender] ⏳ Waiting {retry_delay}s...")
                    time.sleep(retry_delay)

    def _ensure_file_synced(self, file_path):
        if not os.path.exists(file_path): return
        try:
            if hasattr(os, 'fsync'):
                with open(file_path, 'rb') as f: os.fsync(f.fileno())
        except: pass

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

    def _extract_loras_from_workflow(self, prompt_dict, extra_pnginfo=None, enable_enhanced=True):
        loras = []
        try:
            if prompt_dict:
                for node in prompt_dict.values():
                    if "LoraLoader" in node.get("class_type", ""):
                        name = node.get("inputs", {}).get("lora_name", "")
                        if name: loras.append(os.path.splitext(os.path.basename(name))[0].lower())
        except: pass
        return list(set(loras))

# Mappings
NODE_CLASS_MAPPINGS = { "TelegramConfig": TelegramConfig, "TelegramSender": TelegramSender }
NODE_DISPLAY_NAME_MAPPINGS = { "TelegramConfig": "⚙️ Telegram Config", "TelegramSender": "📤 Send to Telegram" }