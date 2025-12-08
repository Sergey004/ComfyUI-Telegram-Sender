print("=" * 50)
print("[TelegramSender] __init__.py IS LOADING NOW!")
print("=" * 50)

from .telegram_sender import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
import execution

try:
    print("[TelegramSender] Trying to import civitai_fetch...")
    from .telegram_modules.utils.civitai_fetch import fetch_missing
    import threading
    
    print("[TelegramSender] Starting thread...")
    
    def _start_civitai_fetch():
        print("[TelegramSender] Thread is running!")
        try:
            fetch_missing(nsfw=True)
            print("[TelegramSender] fetch_missing completed!")
        except Exception as e:
            print(f"[TelegramSender] Error in fetch_missing: {e}")
            import traceback
            traceback.print_exc()
    
    threading.Thread(target=_start_civitai_fetch, daemon=True).start()
    print("[TelegramSender] Thread started successfully")
    
except Exception as e:
    print(f"[TelegramSender] Exception during init: {e}")
    import traceback
    traceback.print_exc()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']