import json
import os
from aiohttp import web

# Import from telegram_sender
from .telegram_sender import CONFIG_FILE, load_config, save_config, set_in_memory_config

async def save_settings_handler(request):
    """API endpoint to save settings from JavaScript"""
    try:
        data = await request.json()
        
        # Validate required fields
        config = {
            "bot_token": data.get("bot_token", ""),
            "default_chat_id": data.get("default_chat_id", ""),
            "lora_mapping": data.get("lora_mapping", ""),
            "nsfw_channel_id": data.get("nsfw_channel_id", ""),
            "unsorted_channel_id": data.get("unsorted_channel_id", "")
        }
        
        # Save to in-memory config for immediate use
        set_in_memory_config(config)
        
        # Also save to file for persistence
        if save_config(config):
            print("[Telegram API] ✅ Settings saved successfully")
            return web.json_response({"success": True})
        else:
            print("[Telegram API] ❌ Failed to save settings")
            return web.json_response({"success": False, "error": "Failed to save settings"}, status=500)
            
    except Exception as e:
        print(f"[Telegram API] ❌ Error saving settings: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def get_settings_handler(request):
    """API endpoint to get current settings"""
    try:
        config = load_config()
        return web.json_response({"config": config})
    except Exception as e:
        print(f"[Telegram API] ❌ Error getting settings: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def migrate_config_handler(request):
    """API endpoint to migrate settings from old config file"""
    try:
        # Check if old config file exists
        if not os.path.exists(CONFIG_FILE):
            return web.json_response({"migrated": False, "message": "No old config found"})
        
        # Load old config
        with open(CONFIG_FILE, 'r') as f:
            old_config = json.load(f)
        
        # Set in-memory config
        set_in_memory_config(old_config)
        
        print("[Telegram API] ✅ Settings migrated from old config file")
        return web.json_response({
            "migrated": True,
            "config": old_config
        })
        
    except Exception as e:
        print(f"[Telegram API] ❌ Error migrating config: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)

# Register API routes
def register_routes(routes):
    """Register Telegram API routes"""
    routes.append(web.get('/telegram/get_settings', get_settings_handler))
    routes.append(web.post('/telegram/save_settings', save_settings_handler))
    routes.append(web.post('/telegram/migrate_config', migrate_config_handler))
    print("[Telegram API] ✅ API routes registered")
