import asyncio
import ast
import os
import re
import requests
import threading
import json
import shutil
import tempfile
import time
import uuid
from queue import Queue
import folder_paths
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
from datetime import datetime
from server import PromptServer
from aiohttp import web
from .telegram_modules.utils.schedulerl_sel import SchedulerSelectorKSampler
from .telegram_modules.utils.civitai_scan import civitai_scan_manager


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


@PromptServer.instance.routes.post("/telegram_sender/civitai_fetch")
async def start_civitai_fetch(request):
    try:
        from .telegram_settings import get_civitai_nsfw_level

        status = civitai_scan_manager.start(
            max_nsfw_level=get_civitai_nsfw_level(),
            force=request.query.get("force") == "1",
        )
        return web.json_response(status)
    except Exception as error:
        print(f"[Telegram Sender] Could not start CivitAI scan: {error}")
        return web.json_response({"error": str(error)}, status=500)


@PromptServer.instance.routes.post("/telegram_sender/civitai_fetch/cancel")
async def cancel_civitai_fetch(request):
    civitai_scan_manager.stop(timeout=0)
    return web.json_response(civitai_scan_manager.get_status())


@PromptServer.instance.routes.get("/telegram_sender/civitai_fetch_status")
async def get_civitai_fetch_status(request):
    return web.json_response(civitai_scan_manager.get_status())


class TelegramUploadQueue:
    def __init__(self):
        self._tasks = Queue(maxsize=2)
        self._lock = threading.Lock()
        self._workers = []

    def submit(self, function, *args):
        with self._lock:
            while len(self._workers) < 2:
                worker = threading.Thread(
                    target=self._work,
                    name=f"TelegramUpload-{len(self._workers) + 1}",
                    daemon=True,
                )
                worker.start()
                self._workers.append(worker)
        self._tasks.put((function, args))

    def _work(self):
        while True:
            function, args = self._tasks.get()
            try:
                function(*args)
            except Exception as error:
                print(f"[Telegram Sender] Telegram upload worker failed: {error}")
            finally:
                self._tasks.task_done()


_upload_queue = TelegramUploadQueue()

from .telegram_settings import get_config
from .telegram_metadata import TelegramMetadata

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
            for pattern in rx:
                try:
                    re.compile(str(pattern))
                    regex.append(str(pattern))
                except re.error as error:
                    print(f"[Telegram Sender] Ignoring invalid NSFW pattern {pattern!r}: {error}")
            negative_keywords = [str(x).lower() for x in nkw]
            negative_regex = [str(x) for x in nrx]
        result = {'keywords': keywords, 'regex': regex, 'negative_keywords': negative_keywords, 'negative_regex': negative_regex}
        _nsfw_triggers_cache = result
        _nsfw_triggers_mtime = mtime
        return result
    except (OSError, json.JSONDecodeError, TypeError) as error:
        print(f"[Telegram Sender] Could not load NSFW triggers: {error}")
        return None

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
                    if not config.get("civitai_api_key"): config["civitai_api_key"] = data.get("Telegram.CivitaiApiKey", "")
                    if not config.get("civitai_nsfw_level"): config["civitai_nsfw_level"] = data.get("Telegram.CivitaiNsfwLevel", "")
            except (OSError, json.JSONDecodeError, TypeError) as error:
                print(f"[Telegram Sender] Could not read ComfyUI settings: {error}")
        return config

    async def send_to_telegram(
        self,
        images,
        chat_id="",
        bot_token_override="",
        positive_prompt="",
        negative_prompt="",
        send_as_document=False,
        max_size=2560,
        landscape_max_width=5120,
        enable_nsfw_detection=False,
        nsfw_channel_id="",
        unsorted_channel_id="",
        enable_lora_routing=True,
        retry_count=3,
        retry_delay=5,
        enable_enhanced_metadata=True,
        filename_prefix="",
        subdirectory_name="",
        debug_metadata=False,
        prompt=None,
        extra_pnginfo=None,
    ):
        full_config = self._get_full_config()
        bot_token = bot_token_override.strip() if bot_token_override else full_config.get("bot_token", "")
        chat_id = chat_id or full_config.get("default_chat_id", "")
        nsfw_channel_id = nsfw_channel_id or full_config.get("nsfw_channel_id", "")
        unsorted_channel_id = unsorted_channel_id or full_config.get("unsorted_channel_id", "")

        if not bot_token:
            print("[Telegram Sender] No bot token configured; images will only be saved locally.")

        metadata_text = self._build_metadata_text(
            positive_prompt,
            negative_prompt,
            prompt,
            extra_pnginfo,
            enable_enhanced_metadata,
        )
        loras_in_workflow = (
            await self._extract_loras_from_workflow(prompt, extra_pnginfo, enable_enhanced_metadata)
            if prompt else []
        )
        results = []

        for index, image in enumerate(images):
            try:
                pnginfo_dict = {}
                direct_seed = self._extract_direct_seed(prompt)
                if enable_enhanced_metadata and prompt:
                    try:
                        pnginfo_dict = await TelegramMetadata.get_metadata(prompt) or {}
                    except Exception as error:
                        print(f"[Telegram Sender] Could not extract metadata: {error}")
                if direct_seed is not None and not pnginfo_dict.get("Seed"):
                    pnginfo_dict["Seed"] = direct_seed

                formatted_filename = self._safe_filename(
                    self._format_filename(filename_prefix.strip(), pnginfo_dict)
                )
                if not formatted_filename:
                    formatted_filename = f"telegram_{uuid.uuid4().hex}"
                save_dir = self._resolve_output_directory(subdirectory_name, pnginfo_dict)
                os.makedirs(save_dir, exist_ok=True)
                output_path = self._reserve_output_path(save_dir, formatted_filename)

                pnginfo = self._create_pnginfo(
                    pnginfo_dict,
                    metadata_text,
                    enable_enhanced_metadata,
                    prompt,
                    extra_pnginfo,
                )
                try:
                    await asyncio.to_thread(self._write_output_image, image, output_path, pnginfo)
                except Exception:
                    if os.path.exists(output_path):
                        try:
                            os.remove(output_path)
                        except OSError as cleanup_error:
                            print(f"[Telegram Sender] Could not remove incomplete output: {cleanup_error}")
                    raise
                results.append({
                    "filename": os.path.basename(output_path),
                    "subfolder": os.path.relpath(save_dir, self.output_dir),
                    "type": self.type,
                })

                target_chat_id = self._determine_chat_id(
                    chat_id,
                    positive_prompt,
                    negative_prompt,
                    enable_nsfw_detection,
                    nsfw_channel_id,
                    unsorted_channel_id,
                    enable_lora_routing,
                    loras_in_workflow,
                    full_config,
                )
                if not target_chat_id:
                    print("[Telegram Sender] No valid chat ID determined; image was saved locally.")
                    continue
                if not bot_token:
                    continue

                await asyncio.to_thread(
                    _upload_queue.submit,
                    self._send_telegram_request,
                    output_path,
                    target_chat_id,
                    bot_token,
                    send_as_document,
                    retry_count,
                    retry_delay,
                    max_size,
                    landscape_max_width,
                )
            except Exception as error:
                print(f"[Telegram Sender] Failed to process image {index}: {error}")

        return {"ui": {"images": results}, "result": (images,)}

    def _extract_direct_seed(self, prompt):
        if not prompt:
            return None
        try:
            for node in prompt.values():
                class_type = node.get("class_type", "")
                inputs = node.get("inputs", {})
                if "sampler" in class_type.lower() or "ksampler" in class_type.lower():
                    seed = inputs.get("seed") or inputs.get("noise_seed") or inputs.get("seed_value")
                    normalized = self._normalize_seed_value(seed)
                    if normalized:
                        return normalized
        except (AttributeError, TypeError) as error:
            print(f"[Telegram Sender] Could not read seed from workflow: {error}")
        return None

    def _resolve_output_directory(self, subdirectory_name, metadata):
        output_root = os.path.realpath(self.output_dir)
        subdirectory = self._format_filename(subdirectory_name.strip(), metadata) if subdirectory_name.strip() else ""
        if subdirectory:
            relative = os.path.normpath(subdirectory)
            if os.path.isabs(relative) or relative == os.pardir or relative.startswith(os.pardir + os.sep):
                raise ValueError("Subdirectory must stay inside the ComfyUI output directory")
            output_dir = os.path.realpath(os.path.join(output_root, relative))
        else:
            output_dir = output_root
        if os.path.commonpath((output_root, output_dir)) != output_root:
            raise ValueError("Subdirectory must stay inside the ComfyUI output directory")
        return output_dir

    @staticmethod
    def _safe_filename(filename):
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
        return filename.strip(" .")

    @staticmethod
    def _reserve_output_path(output_dir, filename):
        while True:
            suffix = f"_{uuid.uuid4().hex[:8]}" if os.path.exists(
                os.path.join(output_dir, f"{filename}.png")
            ) else ""
            path = os.path.join(output_dir, f"{filename}{suffix}.png")
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
            except FileExistsError:
                continue
            os.close(descriptor)
            return path

    @staticmethod
    def _create_pnginfo(pnginfo_dict, metadata_text, enhanced, prompt, extra_pnginfo):
        pnginfo = PngInfo()
        if enhanced and pnginfo_dict:
            try:
                params_str = TelegramMetadata.get_parameters_str(pnginfo_dict)
                if params_str:
                    pnginfo.add_text("parameters", params_str)
                for key, value in pnginfo_dict.items():
                    if value and isinstance(value, (str, int, float)):
                        pnginfo.add_text(str(key), str(value))
            except Exception as error:
                print(f"[Telegram Sender] Could not encode enhanced metadata: {error}")
        elif metadata_text:
            pnginfo.add_text("parameters", metadata_text)
        if prompt:
            pnginfo.add_text("prompt", json.dumps(prompt, default=str))
        if extra_pnginfo:
            for key, value in extra_pnginfo.items():
                pnginfo.add_text(str(key), json.dumps(value, default=str))
        return pnginfo

    @staticmethod
    def _write_output_image(image, output_path, pnginfo):
        image_array = (255.0 * image.cpu().numpy()).astype(np.uint8)
        Image.fromarray(image_array).save(output_path, "PNG", pnginfo=pnginfo)

    def _build_metadata_text(self, positive_prompt, negative_prompt, prompt_dict, extra_pnginfo, enable_enhanced=True):
        parts = []
        if positive_prompt: parts.append(positive_prompt)
        if negative_prompt: parts.append(f"Negative prompt: {negative_prompt}")
        if prompt_dict:
            try:
                params = self._extract_parameters_from_workflow(prompt_dict)
                if params: parts.append(params)
            except (AttributeError, TypeError) as error:
                print(f"[Telegram Sender] Could not extract workflow parameters: {error}")
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
        except (AttributeError, TypeError) as error:
            print(f"[Telegram Sender] Could not parse workflow parameters: {error}")
        return ", ".join(params)

    # --- ORIGINAL LORA EXTRACTION (CRITICAL FOR ROUTING) ---
    async def _extract_loras_from_workflow(self, prompt_dict, extra_pnginfo=None, enable_enhanced=True):
        loras = []
        if not prompt_dict: return loras
        if enable_enhanced:
            try:
                pnginfo_dict = await TelegramMetadata.get_metadata(prompt_dict)
                if pnginfo_dict and "Lora hashes" in pnginfo_dict:
                    for part in pnginfo_dict["Lora hashes"].split(","):
                        lora_name = part.split(":")[0].strip()
                        if lora_name: loras.append(lora_name.lower().strip())
                    if loras: return loras
            except Exception as error:
                print(f"[Telegram Sender] Could not extract LoRA metadata: {error}")
        try:
            for node in prompt_dict.values():
                class_type = node.get("class_type", "")
                if "LoraLoader" in class_type:
                    name = node.get("inputs", {}).get("lora_name", "")
                    if name:
                        clean = os.path.splitext(os.path.basename(name))[0].lower().strip()
                        if clean not in loras: loras.append(clean)
        except (AttributeError, TypeError) as error:
            print(f"[Telegram Sender] Could not inspect workflow LoRAs: {error}")
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
            except (AttributeError, TypeError) as error:
                print(f"[Telegram Sender] Could not parse LoRA mapping entry: {error}")
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
        with Image.open(image_path) as image:
            width, height = image.size
            maximum = max(1, landscape_max_width if width >= height else max_size)
            if max(width, height) <= maximum:
                return image_path
            ratio = maximum / max(width, height)
            new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
            resampling = getattr(Image, "Resampling", Image)
            resized = image.convert("RGB").resize(new_size, resampling.LANCZOS)
            temp_path = os.path.splitext(image_path)[0] + "_resized.jpg"
            resized.save(temp_path, "JPEG", quality=85, optimize=True)
            return temp_path

    def _compress_image(self, image_path, target_size=10 * 1024 * 1024):
        temp_path = os.path.splitext(image_path)[0] + "_compressed.jpg"
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            for quality in range(85, 24, -10):
                rgb_image.save(temp_path, "JPEG", quality=quality, optimize=True)
                if os.path.getsize(temp_path) <= target_size:
                    return temp_path
        if os.path.getsize(temp_path) > target_size:
            raise ValueError(f"Could not compress Telegram photo below {target_size} bytes")
        return temp_path

    def _send_telegram_request(
        self,
        output_path,
        chat_id,
        bot_token,
        send_as_document,
        retry_count,
        retry_delay,
        max_size,
        landscape_max_width,
    ):
        with tempfile.TemporaryDirectory(prefix="telegram-upload-") as temp_dir:
            photo_copy = os.path.join(temp_dir, "photo.png")
            shutil.copyfile(output_path, photo_copy)
            photo_path = self._resize_image(photo_copy, max_size, landscape_max_width)
            if os.path.getsize(photo_path) > 10 * 1024 * 1024:
                photo_path = self._compress_image(photo_path)
            self._send_upload(
                photo_path,
                chat_id,
                bot_token,
                as_document=False,
                retry_count=retry_count,
                retry_delay=retry_delay,
            )
            if send_as_document:
                self._send_upload(
                    output_path,
                    chat_id,
                    bot_token,
                    as_document=True,
                    retry_count=retry_count,
                    retry_delay=retry_delay,
                )

    @staticmethod
    def _send_upload(image_path, chat_id, bot_token, as_document, retry_count, retry_delay):
        method = "sendDocument" if as_document else "sendPhoto"
        param_name = "document" if as_document else "photo"
        url = f"https://api.telegram.org/bot{bot_token}/{method}"
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        timeout = (10, max(60, min(180, 60 + int(file_size_mb * 5))))
        attempts = max(1, int(retry_count))

        for attempt in range(attempts):
            try:
                with open(image_path, "rb") as image_file:
                    with requests.post(
                        url,
                        data={"chat_id": chat_id},
                        files={param_name: image_file},
                        timeout=timeout,
                    ) as response:
                        if response.ok:
                            try:
                                result = response.json()
                            except ValueError:
                                result = {"ok": True}
                            if isinstance(result, dict) and result.get("ok", True):
                                file_type = "Document" if as_document else "Photo"
                                print(f"[Telegram Sender] Sent to {chat_id} ({file_type})")
                                return True
                        response_text = response.text.replace(bot_token, "[redacted]") if bot_token else response.text
                        print(f"[Telegram Sender] Telegram API rejected upload: {response_text}")
            except (OSError, requests.RequestException) as error:
                message = str(error).replace(bot_token, "[redacted]") if bot_token else str(error)
                print(
                    f"[Telegram Sender] Telegram upload attempt {attempt + 1} "
                    f"failed ({type(error).__name__}): {message}"
                )
            if attempt + 1 < attempts:
                time.sleep(max(0, retry_delay))
        return False

    def _normalize_seed_value(self, seed_value):
        """
        Normalize seed value to ensure it's a clean integer or string.
        Handles various formats: int, list [183, 0], string representations, etc.
        Based on revived_comfyui_image_metadata_extension logic.
        """
        if seed_value is None:
            return None
        
        # If it's a list or tuple, extract first element (like [183, 0])
        if isinstance(seed_value, (list, tuple)):
            if len(seed_value) > 0:
                seed_value = seed_value[0]
            else:
                return None
        
        # Convert to string first to handle all cases
        seed_str = str(seed_value).strip()
        if not seed_str or seed_str == "None":
            return None
        
        # Handle string representation of list: "[183, 0]" -> "183"
        if seed_str.startswith("[") and seed_str.endswith("]"):
            try:
                parsed = ast.literal_eval(seed_str)
                if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                    seed_value = parsed[0]
                    if not isinstance(seed_value, (int, float, str)) or isinstance(seed_value, bool):
                        return None
                    seed_str = str(seed_value).strip()
                else:
                    return None
            except (ValueError, SyntaxError, TypeError):
                return None
        
        # Try to convert to integer for validation
        try:
            seed_int = int(float(seed_str))  # int(float()) handles "183.0" cases
            return str(seed_int) if seed_int != 0 else None
        except (ValueError, TypeError, OverflowError):
            # If can't convert to int, return as string if it's not empty
            return seed_str if seed_str else None

    def _format_filename(self, filename_template, pnginfo_dict):
        if "%" not in filename_template: return filename_template
        
        now = datetime.now()
        date_table = {"yyyy": f"{now.year}", "MM": f"{now.month:02d}", "dd": f"{now.day:02d}", "hh": f"{now.hour:02d}", "mm": f"{now.minute:02d}", "ss": f"{now.second:02d}"}
        
        # Extract seed - check multiple possible keys and normalize
        seed = pnginfo_dict.get("Seed") or pnginfo_dict.get("seed") or ""
        seed = self._normalize_seed_value(seed)
        
        # Extract model - also handle lists
        model = pnginfo_dict.get("Model") or pnginfo_dict.get("model") or ""
        if isinstance(model, (list, tuple)) and len(model) > 0:
            model = model[0]
        model = str(model).strip() if model else ""
        if model: 
            model = os.path.splitext(os.path.basename(model))[0]
        
        # Only replace if value is not empty
        if seed:
            filename_template = filename_template.replace("%seed%", seed)
        if model:
            filename_template = filename_template.replace("%model%", model)
        
        # Width/height - also handle lists
        width = pnginfo_dict.get("Width") or pnginfo_dict.get("width") or ""
        if isinstance(width, (list, tuple)) and len(width) > 0:
            width = width[0]
        height = pnginfo_dict.get("Height") or pnginfo_dict.get("height") or ""
        if isinstance(height, (list, tuple)) and len(height) > 0:
            height = height[0]
        if width:
            filename_template = filename_template.replace("%width%", str(width))
        if height:
            filename_template = filename_template.replace("%height%", str(height))
        
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
                txt = pnginfo_dict.get(k_lookup) or pnginfo_dict.get(key) or ""
                # Handle list case
                if isinstance(txt, (list, tuple)) and len(txt) > 0:
                    txt = txt[0]
                length = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                if txt:
                    txt = str(txt).replace("\n", " ").replace("/", "")
                    filename_template = filename_template.replace(segment, txt[:length].strip() if length else txt.strip())
            elif key not in ["seed", "model", "width", "height"]:
                # Unknown placeholder - leave it as is
                pass
        
        return filename_template

# Mappings
NODE_CLASS_MAPPINGS = { "TelegramConfig": TelegramConfig, "TelegramSender": TelegramSender, "Schedulerselector": SchedulerSelectorKSampler, }
NODE_DISPLAY_NAME_MAPPINGS = { "TelegramConfig": "⚙️ Telegram Config", "TelegramSender": "📤 Send to Telegram", "Schedulerselector":"Scheduler Selector (for KSampler (Efficient))"}