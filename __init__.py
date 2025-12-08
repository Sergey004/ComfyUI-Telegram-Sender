print("=" * 50)
print("[Telegram Sender] __init__.py IS LOADING NOW!")
print("=" * 50)

from .telegram_sender import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
import execution

try:
    print("[Telegram Sender] Trying to import civitai_fetch...")
    from .telegram_modules.utils.civitai_fetch import fetch_missing
    import threading
    
    print("[Telegram Sender] Starting thread...")
    
    def _start_civitai_fetch():
        print("[Telegram Sender] Thread is running!")
        try:
            fetch_missing(nsfw=True)
            print("[Telegram Sender] fetch_missing completed!")
        except Exception as e:
            print(f"[Telegram Sender] Error in fetch_missing: {e}")
            import traceback
            traceback.print_exc()
    
    threading.Thread(target=_start_civitai_fetch, daemon=True).start()
    print("[Telegram Sender] Thread started successfully")
    
except Exception as e:
    print(f"[Telegram Sender] Exception during init: {e}")
    import traceback
    traceback.print_exc()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']