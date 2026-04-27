from .telegram_sender import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

try:
    print("[Telegram Sender] Trying to import civitai_fetch...")
    from .telegram_modules.utils.civitai_fetch import fetch_missing
    from .telegram_settings import get_civitai_nsfw_level
    import threading

    print("[Telegram Sender] Starting thread...")

    def _start_civitai_fetch():
        print("[Telegram Sender] Thread is running!")
        try:
            nsfw_level = get_civitai_nsfw_level()
            print(f"[Telegram Sender] CivitAI NSFW level from settings: {nsfw_level}")
            fetch_missing(max_nsfw_level=nsfw_level)
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
