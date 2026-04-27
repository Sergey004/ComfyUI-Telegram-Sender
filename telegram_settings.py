"""
ComfyUI Settings API integration for Telegram Sender
Robust version: Uses folder_paths to find config reliably.
"""

import os
import json
import sys
from comfy.settings import Settings # type: ignore

try:
    import folder_paths # type: ignore
except ImportError:
    folder_paths = None

_settings = None

NSFW_LEVEL_CHOICES = {
    "PG": 1,
    "PG-13": 2,
    "R": 3,
    "X": 4,
    "XXX": 5,
}

def get_settings():
    """Get or create the Settings object"""
    global _settings
    if _settings is None:
        _settings = Settings("Telegram")
        _settings.add_setting("BotToken", default="", type="string", secret=True)
        _settings.add_setting("DefaultChatId", default="", type="string")
        _settings.add_setting("LoraMapping", default="", type="text")
        _settings.add_setting("NSFWChannelId", default="", type="string")
        _settings.add_setting("UnsortedChannelId", default="", type="string")
        _settings.add_setting("CivitaiApiKey", default="", type="string", secret=True)
        _settings.add_setting("CivitaiNsfwLevel", default="PG", type="combo", choices=list(NSFW_LEVEL_CHOICES.keys()))
    return _settings

def get_civitai_api_key():
    """Get CivitAI API key from settings, with env var fallback."""
    settings = get_settings()
    key = settings.get("CivitaiApiKey") or ""
    if not key:
        try:
            file_data = _manual_read_from_file()
            key = file_data.get("Telegram.CivitaiApiKey", "")
        except Exception:
            pass
    if not key:
        key = os.getenv("CIVITAI_API_KEY", "")
    return key

def get_civitai_nsfw_level():
    """Get max NSFW level for CivitAI previews from settings."""
    settings = get_settings()
    level_name = settings.get("CivitaiNsfwLevel") or ""
    if not level_name:
        try:
            file_data = _manual_read_from_file()
            level_name = file_data.get("Telegram.CivitaiNsfwLevel", "")
        except Exception:
            pass
    level = NSFW_LEVEL_CHOICES.get(level_name, 1)
    return level

def _manual_read_from_file():
    """Fallback: Manually read user/default/comfy.settings.json"""
    try:
        if folder_paths:
            base_path = folder_paths.base_path
        else:
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        settings_path = os.path.join(base_path, "user", "default", "comfy.settings.json")

        print(f"[Telegram Settings] Trying to read config from: {settings_path}")

        if not os.path.exists(settings_path):
            print(f"[Telegram Settings] File not found at path! Checking nearby folders...")
            user_dir = os.path.join(base_path, "user")
            if os.path.exists(user_dir):
                print(f"Contents of 'user' folder: {os.listdir(user_dir)}")
            return {}

        with open(settings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        token = data.get("Telegram.BotToken", "")
        if token:
            print(f"[Telegram Settings] Settings file loaded manually.")
        else:
            print(f"[Telegram Settings] File opened, but 'Telegram.BotToken' is missing or empty inside.")

        return data

    except Exception as e:
        print(f"[Telegram Settings] CRITICAL FAIL: {e}")
        return {}

def get_config():
    """Get all settings"""
    settings = get_settings()

    bot_token = settings.get("BotToken") or ""
    chat_id = settings.get("DefaultChatId") or ""
    lora_map = settings.get("LoraMapping") or ""
    nsfw_id = settings.get("NSFWChannelId") or ""
    unsorted_id = settings.get("UnsortedChannelId") or ""
    civitai_key = settings.get("CivitaiApiKey") or ""
    civitai_nsfw_level = settings.get("CivitaiNsfwLevel") or ""

    if not bot_token:
        file_data = _manual_read_from_file()

    if not bot_token: bot_token = file_data.get("Telegram.BotToken", "")
    if not chat_id: chat_id = file_data.get("Telegram.DefaultChatId", "")
    if not lora_map: lora_map = file_data.get("Telegram.LoraMapping", "")
    if not nsfw_id: nsfw_id = file_data.get("Telegram.NSFWChannelId", "")
    if not unsorted_id: unsorted_id = file_data.get("Telegram.UnsortedChannelId", "")
    if not civitai_key: civitai_key = file_data.get("Telegram.CivitaiApiKey", "")
    if not civitai_nsfw_level: civitai_nsfw_level = file_data.get("Telegram.CivitaiNsfwLevel", "")

    return {
        "bot_token": bot_token,
        "default_chat_id": chat_id,
        "lora_mapping": lora_map,
        "nsfw_channel_id": nsfw_id,
        "unsorted_channel_id": unsorted_id,
        "civitai_api_key": civitai_key,
        "civitai_nsfw_level": civitai_nsfw_level,
    }

def force_migrate_from_legacy_config(path):
    return False