"""
ComfyUI Settings API integration for Telegram Sender
Robust version: Uses folder_paths to find config reliably.
"""

import os
import json
import sys

try:
    import folder_paths # type: ignore
except ImportError:
    folder_paths = None

_settings = None
_settings_available = None

NSFW_LEVEL_CHOICES = {
    "PG": 1,
    "PG-13": 2,
    "R": 3,
    "X": 4,
    "XXX": 5,
}

def _try_init_settings():
    global _settings, _settings_available
    if _settings_available is not None:
        return _settings_available
    try:
        from comfy.settings import Settings # type: ignore
        _settings = Settings("Telegram")
        _settings.add_setting("BotToken", default="", type="string", secret=True)
        _settings.add_setting("DefaultChatId", default="", type="string")
        _settings.add_setting("LoraMapping", default="", type="text")
        _settings.add_setting("NSFWChannelId", default="", type="string")
        _settings.add_setting("UnsortedChannelId", default="", type="string")
        _settings.add_setting("CivitaiApiKey", default="", type="string", secret=True)
        _settings.add_setting("CivitaiNsfwLevel", default="PG", type="combo", choices=list(NSFW_LEVEL_CHOICES.keys()))
        _settings_available = True
    except Exception:
        _settings_available = False
    return _settings_available

def get_settings():
    _try_init_settings()
    return _settings

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

def _get_setting_value(key, file_data=None):
    if _settings_available and _settings:
        try:
            val = _settings.get(key)
            if val:
                return val
        except Exception:
            pass
    if file_data is None:
        file_data = _manual_read_from_file()
    return file_data.get(f"Telegram.{key}", "")

def get_civitai_api_key():
    key = _get_setting_value("CivitaiApiKey")
    if not key:
        key = os.getenv("CIVITAI_API_KEY", "")
    return key

def get_civitai_nsfw_level():
    level_name = _get_setting_value("CivitaiNsfwLevel")
    level = NSFW_LEVEL_CHOICES.get(level_name, 1)
    return level

def get_config():
    """Get all settings"""
    _try_init_settings()
    file_data = _manual_read_from_file() if not _settings_available else {}

    bot_token = _get_setting_value("BotToken", file_data)
    chat_id = _get_setting_value("DefaultChatId", file_data)
    lora_map = _get_setting_value("LoraMapping", file_data)
    nsfw_id = _get_setting_value("NSFWChannelId", file_data)
    unsorted_id = _get_setting_value("UnsortedChannelId", file_data)
    civitai_key = _get_setting_value("CivitaiApiKey", file_data)
    civitai_nsfw_level = _get_setting_value("CivitaiNsfwLevel", file_data)

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