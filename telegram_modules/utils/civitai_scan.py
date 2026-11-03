import atexit
import threading
import traceback
from datetime import datetime, timezone

from .log import print_error, print_info


class CivitAIScanManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._cancel_event = None
        self._status = {
            "state": "IDLE",
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }

    def get_status(self):
        with self._lock:
            return dict(self._status)

    def start(self, max_nsfw_level=1, force=False):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return dict(self._status)
            if not force and self._status["state"] != "IDLE":
                return dict(self._status)

            cancel_event = threading.Event()
            self._cancel_event = cancel_event
            self._status = {
                "state": "RUNNING",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "result": None,
                "error": None,
            }
            self._thread = threading.Thread(
                target=self._run,
                args=(max_nsfw_level, cancel_event),
                name="TelegramSender-CivitAIFetch",
                daemon=True,
            )
            try:
                self._thread.start()
            except RuntimeError as error:
                self._status["state"] = "FAILED"
                self._status["error"] = str(error)
                self._status["finished_at"] = datetime.now(timezone.utc).isoformat()
                self._thread = None
                raise
            return dict(self._status)

    def _run(self, max_nsfw_level, cancel_event):
        print_info("Starting background CivitAI model scan")
        try:
            from .civitai_fetch import fetch_missing

            result = fetch_missing(
                max_nsfw_level=max_nsfw_level,
                cancel_event=cancel_event,
            )
        except Exception as error:
            traceback.print_exc()
            with self._lock:
                self._status["state"] = "STOPPING" if cancel_event.is_set() else "FAILED"
                self._status["error"] = str(error)
                self._status["finished_at"] = datetime.now(timezone.utc).isoformat()
            print_error(f"CivitAI scan failed: {error}")
            return

        with self._lock:
            self._status["state"] = (
                "STOPPING"
                if cancel_event.is_set() or result.get("cancelled")
                else "COMPLETED"
            )
            self._status["result"] = result
            self._status["finished_at"] = datetime.now(timezone.utc).isoformat()
        print_info(f"CivitAI model scan finished: {self._status['state']}")

    def stop(self, timeout=5):
        with self._lock:
            thread = self._thread
            cancel_event = self._cancel_event
            if thread and thread.is_alive() and cancel_event:
                cancel_event.set()
                self._status["state"] = "STOPPING"
        if thread and thread.is_alive():
            thread.join(timeout=timeout)


civitai_scan_manager = CivitAIScanManager()
atexit.register(civitai_scan_manager.stop)
