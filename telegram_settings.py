"""
ComfyUI Settings API integration for Telegram Sender
Robust version: Uses folder_paths to find config reliably.
"""

import os
import json
import sys
from comfy.settings import Settings # type: ignore

# Пытаемся импортировать менеджер путей ComfyUI
try:
    import folder_paths # type: ignore
except ImportError:
    folder_paths = None

# Global Settings object
_settings = None

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
    return _settings

def _manual_read_from_file():
    """Fallback: Manually read user/default/comfy.settings.json"""
    try:
        # Способ 1: Через официальный folder_paths (самый надежный)
        if folder_paths:
            base_path = folder_paths.base_path
        else:
            # Способ 2: Вычисление путей вручную (если импорт не сработал)
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
        settings_path = os.path.join(base_path, "user", "default", "comfy.settings.json")
        
        print(f"[Telegram Settings] 🔍 Trying to read config from: {settings_path}")
        
        if not os.path.exists(settings_path):
            print(f"[Telegram Settings] ❌ File not found at path! Checking nearby folders...")
            # Пытаемся помочь найти ошибку, выводя список папок
            user_dir = os.path.join(base_path, "user")
            if os.path.exists(user_dir):
                print(f"Contents of 'user' folder: {os.listdir(user_dir)}")
            return {}

        with open(settings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Проверяем, есть ли там наши настройки
        token = data.get("Telegram.BotToken", "")
        if token:
            print(f"[Telegram Settings] ✅ SUCCESS: Settings file loaded manually.")
        else:
            print(f"[Telegram Settings] ⚠️ File opened, but 'Telegram.BotToken' is missing or empty inside.")
            
        return data
        
    except Exception as e:
        print(f"[Telegram Settings] ❌ CRITICAL FAIL: {e}")
        return {}

def get_config():
    """Get all settings"""
    settings = get_settings()
    
    # Сначала пробуем API
    bot_token = settings.get("BotToken") or ""
    chat_id = settings.get("DefaultChatId") or ""
    lora_map = settings.get("LoraMapping") or ""
    nsfw_id = settings.get("NSFWChannelId") or ""
    unsorted_id = settings.get("UnsortedChannelId") or ""
    
    # Если пусто - читаем файл с диска
    if not bot_token:
        file_data = _manual_read_from_file()
        
        if not bot_token: bot_token = file_data.get("Telegram.BotToken", "")
        if not chat_id: chat_id = file_data.get("Telegram.DefaultChatId", "")
        if not lora_map: lora_map = file_data.get("Telegram.LoraMapping", "")
        if not nsfw_id: nsfw_id = file_data.get("Telegram.NSFWChannelId", "")
        if not unsorted_id: unsorted_id = file_data.get("Telegram.UnsortedChannelId", "")

    return {
        "bot_token": bot_token,
        "default_chat_id": chat_id,
        "lora_mapping": lora_map,
        "nsfw_channel_id": nsfw_id,
        "unsorted_channel_id": unsorted_id
    }

def force_migrate_from_legacy_config(path):
    return False
"""
ComfyUI Settings API integration for Telegram Sender
Robust version: Uses folder_paths to find config reliably.
"""

import os
import json
import sys
from comfy.settings import Settings # type: ignore

# Пытаемся импортировать менеджер путей ComfyUI
try:
    import folder_paths # type: ignore
except ImportError:
    folder_paths = None

# Global Settings object
_settings = None

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
    return _settings

def _manual_read_from_file():
    """Fallback: Manually read user/default/comfy.settings.json"""
    try:
        # Способ 1: Через официальный folder_paths (самый надежный)
        if folder_paths:
            base_path = folder_paths.base_path
        else:
            # Способ 2: Вычисление путей вручную (если импорт не сработал)
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
        settings_path = os.path.join(base_path, "user", "default", "comfy.settings.json")
        
        print(f"[Telegram Settings] 🔍 Trying to read config from: {settings_path}")
        
        if not os.path.exists(settings_path):
            print(f"[Telegram Settings] ❌ File not found at path! Checking nearby folders...")
            # Пытаемся помочь найти ошибку, выводя список папок
            user_dir = os.path.join(base_path, "user")
            if os.path.exists(user_dir):
                print(f"Contents of 'user' folder: {os.listdir(user_dir)}")
            return {}

        with open(settings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Проверяем, есть ли там наши настройки
        token = data.get("Telegram.BotToken", "")
        if token:
            print(f"[Telegram Settings] ✅ SUCCESS: Settings file loaded manually.")
        else:
            print(f"[Telegram Settings] ⚠️ File opened, but 'Telegram.BotToken' is missing or empty inside.")
            
        return data
        
    except Exception as e:
        print(f"[Telegram Settings] ❌ CRITICAL FAIL: {e}")
        return {}

def get_config():
    """Get all settings"""
    settings = get_settings()
    
    # Сначала пробуем API
    bot_token = settings.get("BotToken") or ""
    chat_id = settings.get("DefaultChatId") or ""
    lora_map = settings.get("LoraMapping") or ""
    nsfw_id = settings.get("NSFWChannelId") or ""
    unsorted_id = settings.get("UnsortedChannelId") or ""
    
    # Если пусто - читаем файл с диска
    if not bot_token:
        file_data = _manual_read_from_file()
        
        if not bot_token: bot_token = file_data.get("Telegram.BotToken", "")
        if not chat_id: chat_id = file_data.get("Telegram.DefaultChatId", "")
        if not lora_map: lora_map = file_data.get("Telegram.LoraMapping", "")
        if not nsfw_id: nsfw_id = file_data.get("Telegram.NSFWChannelId", "")
        if not unsorted_id: unsorted_id = file_data.get("Telegram.UnsortedChannelId", "")

    return {
        "bot_token": bot_token,
        "default_chat_id": chat_id,
        "lora_mapping": lora_map,
        "nsfw_channel_id": nsfw_id,
        "unsorted_channel_id": unsorted_id
    }

def force_migrate_from_legacy_config(path):
    return False