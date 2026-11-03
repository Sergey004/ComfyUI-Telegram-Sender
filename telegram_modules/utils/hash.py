import hashlib
import json
import os
import tempfile
import threading
from collections import OrderedDict

from ..config import NODE_CACHE_DIR
from .log import print_warning, print_error

CACHE_FILE = os.path.join(NODE_CACHE_DIR, "model_hash_cache.json")
CACHE_SIZE_LIMIT = 1000

cache_model_hash = OrderedDict()
cache_model_hash_full = OrderedDict()
_disk_cache = {}
_disk_cache_dirty = False
_cache_lock = threading.Lock()
_cache_write_lock = threading.Lock()

if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as cache_file:
            _disk_cache = json.load(cache_file)
        if not isinstance(_disk_cache, dict):
            raise ValueError("Hash cache root must be a JSON object")
    except (OSError, ValueError) as error:
        print_error(f"Failed to load cache file {CACHE_FILE}: {error}")


def get_file_mod_time(path):
    try:
        return os.stat(path).st_mtime_ns
    except OSError as error:
        print_warning(f"Could not read modification time for {path}: {error}")
        return 0


def _file_signature(path):
    stat = os.stat(path)
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def trim_disk_cache():
    global _disk_cache
    if len(_disk_cache) > CACHE_SIZE_LIMIT:
        _disk_cache = dict(list(_disk_cache.items())[-CACHE_SIZE_LIMIT:])


def save_disk_cache():
    global _disk_cache_dirty
    with _cache_write_lock:
        with _cache_lock:
            if not _disk_cache_dirty:
                return
            trim_disk_cache()
            snapshot = dict(_disk_cache)

        temp_path = None
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=os.path.dirname(CACHE_FILE),
                prefix=".model_hash_cache.",
                suffix=".tmp",
                delete=False,
            ) as cache_file:
                temp_path = cache_file.name
                json.dump(snapshot, cache_file, indent=2)
                cache_file.flush()
                os.fsync(cache_file.fileno())
            os.replace(temp_path, CACHE_FILE)
            with _cache_lock:
                if snapshot == _disk_cache:
                    _disk_cache_dirty = False
        except OSError as error:
            print_error(f"Failed to write cache to {CACHE_FILE}: {error}")
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as cleanup_error:
                    print_warning(f"Failed to remove temporary hash cache {temp_path}: {cleanup_error}")


def _hash_file(filename, full, save_cache, cancel_event):
    global _disk_cache_dirty

    if not filename or not os.path.isfile(filename):
        print_warning(f"Hash requested for missing or invalid path: {filename}")
        return ""

    try:
        resolved_path = os.path.realpath(filename)
        signature = _file_signature(resolved_path)
    except OSError as error:
        print_error(f"Could not inspect model file {filename}: {error}")
        return ""

    memory_cache = cache_model_hash_full if full else cache_model_hash
    hash_field = "sha256_full" if full else "file_hash"
    with _cache_lock:
        cached = memory_cache.get(resolved_path)
        if cached and cached[0] == signature:
            memory_cache.move_to_end(resolved_path)
            return cached[1]

        record = _disk_cache.get(resolved_path)
        if (
            isinstance(record, dict)
            and record
            and record.get("mtime_ns") == signature["mtime_ns"]
            and record.get("size") == signature["size"]
            and record.get(hash_field)
        ):
            result = record[hash_field]
            memory_cache[resolved_path] = (signature, result)
            memory_cache.move_to_end(resolved_path)
            return result

    digest = hashlib.sha256()
    try:
        with open(resolved_path, "rb") as model_file:
            for block in iter(lambda: model_file.read(1024 * 1024), b""):
                if cancel_event is not None and cancel_event.is_set():
                    return ""
                digest.update(block)
        result = digest.hexdigest() if full else digest.hexdigest()[:10]

        with _cache_lock:
            if cancel_event is not None and cancel_event.is_set():
                return ""
            if len(memory_cache) >= CACHE_SIZE_LIMIT:
                memory_cache.popitem(last=False)
            memory_cache[resolved_path] = (signature, result)
            record = _disk_cache.get(resolved_path, {})
            record.update(signature)
            record[hash_field] = result
            if full:
                record["file_hash"] = record.get("file_hash") or result[:10]
            _disk_cache[resolved_path] = record
            _disk_cache_dirty = True
        if save_cache:
            save_disk_cache()
        return result
    except OSError as error:
        print_error(f"Failed to calculate hash for {filename}: {error}")
        return ""


def calc_hash(filename, use_only_filename=True, *, save_cache=True, cancel_event=None):
    return _hash_file(filename, False, save_cache, cancel_event)


def calc_sha256_full(filename, use_only_filename=True, *, save_cache=True, cancel_event=None):
    return _hash_file(filename, True, save_cache, cancel_event)
