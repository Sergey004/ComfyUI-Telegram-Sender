"""
ComfyUI Settings API integration for Telegram Sender
Robust version: Uses folder_paths to find config reliably.
"""

import os
import json

try:
    import folder_paths # type: ignore
except ImportError:
    folder_paths = None

NSFW_LEVEL_CHOICES = {
    "PG": 1,
    "PG-13": 2,
    "R": 3,
    "X": 4,
    "XXX": 5,
}

def get_settings():
    """Read ComfyUI's persisted settings file."""
    return _manual_read_from_file()

def _manual_read_from_file():
    """Fallback: Manually read user/default/comfy.settings.json"""
    try:
        if folder_paths:
            user_directory = folder_paths.get_user_directory()
        else:
            user_directory = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "user",
            )

        settings_path = os.path.join(user_directory, "default", "comfy.settings.json")

        if not os.path.exists(settings_path):
            return {}

        with open(settings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as error:
        print(f"[Telegram Settings] Could not read settings file: {error}")
        return {}

def _get_setting_value(key, file_data=None):
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
    """Get settings persisted by ComfyUI's frontend settings API."""
    file_data = _manual_read_from_file()

    bot_token = _get_setting_value("BotToken", file_data)
    chat_id = _get_setting_value("DefaultChatId", file_data)
    lora_map = _get_setting_value("LoraMapping", file_data)
    nsfw_id = _get_setting_value("NSFWChannelId", file_data)
    unsorted_id = _get_setting_value("UnsortedChannelId", file_data)
    civitai_key = _get_setting_value("CivitaiApiKey", file_data)
    civitai_nsfw_level = _get_setting_value("CivitaiNsfwLevel", file_data)
    auto_civitai_fetch = _get_setting_value("AutoCivitaiFetch", file_data)
    if auto_civitai_fetch == "":
        auto_civitai_fetch = True
    elif isinstance(auto_civitai_fetch, str):
        auto_civitai_fetch = auto_civitai_fetch.lower() not in ("false", "0", "no", "")

    return {
        "bot_token": bot_token,
        "default_chat_id": chat_id,
        "lora_mapping": lora_map,
        "nsfw_channel_id": nsfw_id,
        "unsorted_channel_id": unsorted_id,
        "civitai_api_key": civitai_key,
        "civitai_nsfw_level": civitai_nsfw_level,
        "auto_civitai_fetch": bool(auto_civitai_fetch),
    }

def force_migrate_from_legacy_config(path):
    return False