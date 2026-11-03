"""
CivitAI metadata and preview fetcher for ComfyUI Telegram Sender.
Fetches model previews and metadata from CivitAI API.

Optimizations:
- os.scandir instead of glob for faster file listing
- mtime-based caching to avoid re-scanning unchanged directories
- ThreadPoolExecutor for parallel API requests
- Skip patterns for excluding unwanted files
"""
import os
import json
import re
import html
import time
import tempfile
import subprocess
import concurrent.futures
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass, field
from datetime import datetime

from tqdm import tqdm
import requests
import folder_paths

from .hash import calc_sha256_full, save_disk_cache
from .civitai_models import (
    CivitVersion,
    CivitModel,
    CivitImage,
    CivitFile,
    CivitStats,
    CivitCreator,
    filter_images_by_nsfw_level,
    get_first_sfw_image_url,
)

try:
    from ..utils.log import print_info, print_warning, print_error
except ImportError as error:
    print(f"[Telegram Sender] CivitAI logging helpers unavailable: {error}")
    def print_info(msg): print(f" {msg}")
    def print_warning(msg): print(f" WARNING: {msg}")
    def print_error(msg): print(f" ERROR: {msg}")


# Configuration constants
API_BASE_URL = os.getenv("CIVITAI_ENDPOINT", "https://civitai.com/api/v1")
RATE_LIMIT_DELAY = 0.3
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0
OVERWRITE_INFO = os.getenv("CIVITAI_OVERWRITE_INFO", "false").lower() in ("1", "true", "yes")
MAX_WORKERS = int(os.getenv("CIVITAI_MAX_WORKERS", "4"))

USER_AGENT = "CivitaiLink:Automatic1111"

TYPES = {
    "Checkpoint": ("checkpoints", ["safetensors", "ckpt"]),
    "LORA": ("loras", ["safetensors", "pt"]),
    "LoCon": ("loras", ["safetensors", "pt"]),
    "DoRA": ("loras", ["safetensors", "pt"]),
    "TextualInversion": ("embeddings", ["safetensors", "pt", "bin"]),
    "VAE": ("vae", ["safetensors", "ckpt", "pt"]),
    "Upscaler": ("upscale_models", ["safetensors", "ckpt", "pt"]),
    "Controlnet": ("controlnet", ["safetensors", "pt"]),
}

PREVIEW_EXTS = [".jpg", ".png", ".jpeg", ".gif"]
PREVIEW_EXTS = PREVIEW_EXTS + [".preview" + x for x in PREVIEW_EXTS]

NSFW_LEVELS = {
    "None": 1,
    "Soft": 2,
    "Mature": 3,
    "X": 4,
    "XXX": 5,
}

SKIP_PATTERNS = [
    r"^\.DS_Store",
    r"^Thumbs\.db",
    r"^\.git",
    r"^__pycache__",
    r"\.cache$",
]


@dataclass
class DirCache:
    """Cache entry for a scanned directory."""
    mtime_ns: int
    files: List[str] = field(default_factory=list)


_dir_cache: Dict[tuple, DirCache] = {}


def has_preview(path: str) -> bool:
    stem = os.path.splitext(path)[0]
    return any(os.path.exists(stem + ext) for ext in PREVIEW_EXTS)


def has_info(path: str) -> bool:
    return os.path.isfile(os.path.splitext(path)[0] + ".json")


def _read_info_json(path: str) -> Optional[dict]:
    try:
        p = Path(path).with_suffix(".json")
        if not p.exists():
            return None
        return json.loads(p.read_text())
    except (OSError, ValueError, TypeError) as error:
        print_warning(f"Could not read model info JSON for {path}: {error}")
        return None


def needs_info_update(path: str) -> bool:
    p = Path(path).with_suffix(".json")
    if not p.exists():
        return True
    if OVERWRITE_INFO:
        data = _read_info_json(path)
        if data is None:
            return True
        desc = data.get("description")
        if not isinstance(desc, str) or desc.strip() == "":
            return True
    return False


def _basename_key(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].strip().lower()


def _should_skip(name: str) -> bool:
    """Check if file/directory matches skip patterns."""
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, name):
            return True
    return False


def _scan_directory_recursive(
    path: str,
    exts: List[str],
    cancel_event=None,
) -> Iterator[str]:
    """
    Fast recursive directory scanner using os.scandir.
    Uses mtime caching to skip unchanged directories.
    """
    if cancel_event is not None and cancel_event.is_set():
        return
    try:
        current_mtime = os.stat(path).st_mtime_ns
    except OSError:
        return
    cache_key = (os.path.abspath(path), tuple(sorted(exts)))
    cached = _dir_cache.get(cache_key)
    files = list(cached.files) if cached and cached.mtime_ns == current_mtime else []
    child_dirs = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if _should_skip(entry.name):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    child_dirs.append(entry.path)
                elif not cached or cached.mtime_ns != current_mtime:
                    if entry.is_file(follow_symlinks=False):
                        lower_name = entry.name.lower()
                        if any(lower_name.endswith(f".{ext}") for ext in exts):
                            files.append(entry.path)
    except OSError as error:
        print_warning(f"Could not scan model directory {path}: {error}")
        return

    _dir_cache[cache_key] = DirCache(mtime_ns=current_mtime, files=files)
    yield from files
    for child_dir in child_dirs:
        yield from _scan_directory_recursive(child_dir, exts, cancel_event)


def iter_files_for_type(folder_type: str, exts: List[str], cancel_event=None) -> Iterator[str]:
    """Iterate over all model files of a given type."""
    for base in folder_paths.get_folder_paths(folder_type):
        if not os.path.isdir(base):
            continue
        yield from _scan_directory_recursive(base, exts, cancel_event)


def collect_paths(narrow_types: Optional[List[str]] = None, cancel_event=None) -> List[str]:
    print_info(" Scanning ComfyUI model folders for resources...")
    all_paths = set()
    for civ_type, (folder_type, exts) in TYPES.items():
        if cancel_event is not None and cancel_event.is_set():
            break
        if narrow_types and civ_type not in narrow_types:
            continue
        print_info(f" Type {civ_type}: folders={folder_paths.get_folder_paths(folder_type)}")
        paths = list(iter_files_for_type(folder_type, exts, cancel_event))
        print_info(f" {civ_type}: total={len(paths)}")
        for p in paths:
            all_paths.add(p)
    print_info(f" Total files discovered: {len(all_paths)}")
    return list(all_paths)


def build_hash_index(paths: List[str], cancel_event=None) -> Dict[str, List[str]]:
    """
    Build hash index for model files.
    Uses ThreadPoolExecutor for parallel hashing.
    """
    mapping: Dict[str, List[str]] = {}
    if not paths:
        return mapping
    
    def hash_file(path: str) -> Optional[tuple]:
        try:
            h = calc_sha256_full(path, save_cache=False, cancel_event=cancel_event)
            if h:
                return (h.lower(), path)
        except Exception as error:
            print_warning(f"Could not hash model file {path}: {error}")
        return None
    
    if len(paths) > 10:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(hash_file, p): p for p in paths}
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Hash files",
                unit="file"
            ):
                if cancel_event is not None and cancel_event.is_set():
                    for pending in futures:
                        pending.cancel()
                    break
                result = future.result()
                if result:
                    h, path = result
                    mapping.setdefault(h, []).append(path)
    else:
        for path in tqdm(paths, desc="Hash files", unit="file"):
            if cancel_event is not None and cancel_event.is_set():
                break
            result = hash_file(path)
            if result:
                h, path = result
                mapping.setdefault(h, []).append(path)

    save_disk_cache()
    
    print_info(f" Hash index entries: {len(mapping)}")
    return mapping


class CivitaiAPIError(Exception):
    """Custom exception for CivitAI API errors."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")


def _make_request(
    endpoint: str,
    method: str = "GET",
    data: Any = None,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    retry_count: int = 0,
    cancel_event=None,
) -> dict:
    """
    Make HTTP request to CivitAI API with retry support.

    Raises:
    CivitaiAPIError: On non-retryable errors or max retries exceeded
    """
    if cancel_event is not None and cancel_event.is_set():
        return {}
    headers = dict(headers or {})

    headers["User-Agent"] = USER_AGENT

    api_key = os.getenv("CIVITAI_API_KEY")
    if not api_key:
        try:
            from ...telegram_settings import get_civitai_api_key
            api_key = get_civitai_api_key()
        except (ImportError, AttributeError) as error:
            print_warning(f"Could not load CivitAI API key from settings: {error}")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    if data is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(data)
    
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    if params is None:
        params = {}
    
    url = API_BASE_URL + endpoint
    
    try:
        resp = requests.request(method, url, data=data, params=params, headers=headers, timeout=(10, 30))
    except requests.exceptions.Timeout:
        if retry_count < MAX_RETRIES:
            delay = RETRY_BASE_DELAY * (2 ** retry_count)
            print_warning(f" Request timeout, retrying in {delay}s... ({retry_count + 1}/{MAX_RETRIES})")
            if cancel_event is not None:
                if cancel_event.wait(delay):
                    return {}
            else:
                time.sleep(delay)
            return _make_request(endpoint, method, data, params, headers, retry_count + 1, cancel_event)
        raise CivitaiAPIError(0, "Request timeout after max retries")
    except requests.exceptions.ConnectionError as e:
        if retry_count < MAX_RETRIES:
            delay = RETRY_BASE_DELAY * (2 ** retry_count)
            print_warning(f" Connection error, retrying in {delay}s... ({retry_count + 1}/{MAX_RETRIES})")
            if cancel_event is not None:
                if cancel_event.wait(delay):
                    return {}
            else:
                time.sleep(delay)
            return _make_request(endpoint, method, data, params, headers, retry_count + 1, cancel_event)
        raise CivitaiAPIError(0, f"Connection error: {e}")
    
    if resp.status_code == 200:
        return resp.json()
    
    if resp.status_code == 404:
        return {}
    
    if resp.status_code in (429, 502, 503, 504):
        if retry_count < MAX_RETRIES:
            delay = RETRY_BASE_DELAY * (2 ** retry_count)
            print_warning(f" Rate limited/server error {resp.status_code}, retrying in {delay}s... ({retry_count + 1}/{MAX_RETRIES})")
            if cancel_event is not None:
                if cancel_event.wait(delay):
                    return {}
            else:
                time.sleep(delay)
            return _make_request(endpoint, method, data, params, headers, retry_count + 1, cancel_event)
    
    print_error(f" HTTP error {resp.status_code}: {resp.text[:200]}")
    raise CivitaiAPIError(resp.status_code, resp.text[:500])


def _fetch_versions_by_hash(hashes: List[str], cancel_event=None) -> List[CivitVersion]:
    """
    Fetch model versions by their SHA256 hashes.
    Uses batch POST endpoint for efficiency.
    """
    if not isinstance(hashes, (list, tuple)):
        hashes = [hashes]
    
    cleaned = []
    for h in hashes:
        if not h:
            continue
        cleaned.append(str(h).strip().lower())
    
    if not cleaned:
        return []
    
    raw_data = _make_request(
        "/model-versions/by-hash",
        method="POST",
        data=cleaned,
        cancel_event=cancel_event,
    )
    
    if not raw_data:
        return []
    
    if isinstance(raw_data, dict) and "id" in raw_data:
        return [CivitVersion.from_dict(raw_data)]
    
    if isinstance(raw_data, list):
        return [CivitVersion.from_dict(item) for item in raw_data if item]
    
    return []


def _fetch_model_and_version_by_name(
    file_basename_key: str,
    cancel_event=None,
) -> Optional[tuple]:
    """
    Search for a model whose version files contain a file matching file_basename_key.

    Iterates all items returned by the search and checks every version/file,
    so we never accidentally pick a wrong model just because it ranked first.

    Returns (CivitModel, CivitVersion) or None.
    """
    raw_data = _make_request(
        "/models",
        params={"query": file_basename_key, "pageSize": 20},
        cancel_event=cancel_event,
    )
    items = raw_data.get("items", []) if isinstance(raw_data, dict) else []
    if not items:
        return None

    for item in items:
        model = CivitModel.from_dict(item)
        for version in model.versions:
            for f in version.files:
                if _basename_key(f.name) == file_basename_key:
                    return (model, version)

    # Second pass: loose match on model name itself (handles cases where the
    # file was renamed locally but still matches the model title).
    for item in items:
        model = CivitModel.from_dict(item)
        if _basename_key(model.name) == file_basename_key:
            # Return the first (latest) version if available
            if model.versions:
                return (model, model.versions[0])

    return None


def _fetch_by_name(
    paths: List[str],
    max_nsfw_level: int = 1,
    cancel_event=None,
    known_hashes: Optional[Dict[str, str]] = None,
) -> tuple:
    """
    Fetch metadata for models using name-based lookup (fallback method).
    Returns (previews_updated, infos_updated) count.
    """
    updated_prev = 0
    updated_info = 0

    for path in tqdm(list(paths), desc="Name lookup", unit="file"):
        if cancel_event is not None and cancel_event.is_set():
            break
        name = _basename_key(path)

        if cancel_event is not None:
            if cancel_event.wait(RATE_LIMIT_DELAY):
                break
        else:
            time.sleep(RATE_LIMIT_DELAY)

        try:
            result = _fetch_model_and_version_by_name(name, cancel_event)
        except CivitaiAPIError as e:
            print_warning(f" Name lookup failed for {name}: {e.status_code}")
            continue
        except Exception as e:
            print_warning(f" Name lookup error for {name}: {e}")
            continue

        if not result:
            print_info(f" No match found for {name}, skipping")
            continue

        model, best_version = result
        best_image_url = get_first_sfw_image_url(best_version.images, max_nsfw_level)

        if best_image_url:
            if save_preview_for(path, best_image_url):
                updated_prev += 1

        save_info_for(
            path,
            best_version,
            model,
            max_nsfw_level=max_nsfw_level,
            sha256=(known_hashes or {}).get(path, ""),
        )
        if not needs_info_update(path):
            updated_info += 1

    return updated_prev, updated_info


def _is_video_url(url: str) -> bool:
    from urllib.parse import urlparse

    lower = urlparse(url).path.lower()
    if lower.endswith((".mp4", ".webm", ".mov", ".mkv")):
        return True
    try:
        head = requests.head(url, allow_redirects=True, timeout=5)
        ctype = head.headers.get("Content-Type", "")
        if ctype.startswith("video/"):
            return True
    except requests.RequestException as error:
        print_warning(f"Could not determine whether preview is video: {error}")
    return False


def _validate_image(path: str) -> None:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()


def _download_file(
    url: str,
    dest: str,
    max_bytes: int = 20 * 1024 * 1024,
    validate_image: bool = True,
):
    """Download and verify an image before atomically replacing its destination."""
    print_info(f" Downloading preview: {url} -> {dest}")
    dest_dir = os.path.dirname(os.path.abspath(dest))
    os.makedirs(dest_dir, exist_ok=True)
    temp_path = None
    try:
        with requests.get(
            url,
            stream=True,
            headers={"User-Agent": USER_AGENT},
            timeout=(10, 60),
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if validate_image and content_type and not (
                content_type.startswith("image/") or content_type == "application/octet-stream"
            ):
                raise ValueError(f"Unexpected preview content type: {content_type}")
            total = int(response.headers.get("content-length", 0))
            if total > max_bytes:
                raise ValueError(f"Preview exceeds the {max_bytes}-byte size limit")
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=dest_dir,
                prefix=".preview.",
                suffix=".part",
                delete=False,
            ) as output:
                temp_path = output.name
                with tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Download {os.path.basename(dest)}",
                ) as bar:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise ValueError(f"Preview exceeds the {max_bytes}-byte size limit")
                        output.write(chunk)
                        bar.update(len(chunk))
        if validate_image:
            _validate_image(temp_path)
        os.replace(temp_path, dest)
        temp_path = None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _extract_gif_frame(url: str, out_png: str) -> bool:
    """Extract first frame from GIF and save as PNG."""
    print_info(f" Extract first GIF frame: {url} -> {out_png}")
    output_dir = os.path.dirname(os.path.abspath(out_png))
    os.makedirs(output_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".gif")
    os.close(fd)
    out_fd, temp_png = tempfile.mkstemp(suffix=".png", dir=output_dir, prefix=".preview.")
    os.close(out_fd)
    try:
        _download_file(url, tmp, max_bytes=100 * 1024 * 1024, validate_image=False)
        from PIL import Image
        with Image.open(tmp) as image:
            image.seek(0)
            image.convert("RGB").save(temp_png, format="PNG")
        _validate_image(temp_png)
        os.replace(temp_png, out_png)
        return True
    except Exception as error:
        print_warning(f"Could not extract GIF preview from {url}: {error}")
        return False
    finally:
        for temp_path in (tmp, temp_png):
            if os.path.exists(temp_path):
                os.remove(temp_path)


def _extract_video_frame(url: str, out_png: str) -> bool:
    """Extract first frame from video and save as PNG."""
    print_info(f" Extract first video frame: {url} -> {out_png}")
    output_dir = os.path.dirname(os.path.abspath(out_png))
    os.makedirs(output_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    out_fd, temp_png = tempfile.mkstemp(suffix=".png", dir=output_dir, prefix=".preview.")
    os.close(out_fd)
    try:
        _download_file(url, tmp, max_bytes=100 * 1024 * 1024, validate_image=False)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp, "-vframes", "1", temp_png],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            _validate_image(temp_png)
            os.replace(temp_png, out_png)
            return True
        except (OSError, subprocess.SubprocessError, ValueError):
            if os.path.exists(temp_png):
                os.remove(temp_png)
        try:
            import imageio.v3 as iio
            from PIL import Image
            arr = iio.imread(tmp, index=0)
            Image.fromarray(arr).save(temp_png, format="PNG")
            _validate_image(temp_png)
            os.replace(temp_png, out_png)
            return True
        except Exception as error:
            print_warning(f"Could not extract video preview from {url}: {error}")
            return False
    except (OSError, requests.RequestException, ValueError) as error:
        print_warning(f"Could not download video preview from {url}: {error}")
        return False
    finally:
        for temp_path in (tmp, temp_png):
            if os.path.exists(temp_path):
                os.remove(temp_path)


def save_preview_for(path: str, image_url: str) -> bool:
    """Save preview image for model file."""
    from urllib.parse import urlparse

    dest = os.path.splitext(path)[0] + ".preview.png"
    lower = urlparse(image_url).path.lower()
    
    if lower.endswith(".gif"):
        return _extract_gif_frame(image_url, dest)
    
    if _is_video_url(image_url):
        return _extract_video_frame(image_url, dest)
    
    try:
        _download_file(image_url, dest)
        return True
    except (OSError, requests.RequestException, ValueError) as error:
        print_warning(f"Could not save preview for {path}: {error}")
        return False


def _detect_sd_version(base_model: str) -> str:
    """Detect Stable Diffusion version from base model string."""
    if not base_model:
        return "Other"
    
    base_lower = base_model.lower()
    
    if "sdxl" in base_lower:
        return "SDXL"
    if "sd 2" in base_lower or "sd2" in base_lower:
        return "SD 2"
    if "sd 1.5" in base_lower or "sd1.5" in base_lower:
        return "SD 1.5"
    if "illustrious" in base_lower:
        return "Illustrious"
    if "pony" in base_lower:
        return "Pony"
    if "noob" in base_lower:
        return "Noob AI"
    if "flux" in base_lower:
        return "Flux"
    if "sd3" in base_lower or "sd 3" in base_lower:
        return "SD 3"
    
    return "Other"


def _clean_html(text: str) -> str:
    """Remove HTML tags and unescape entities."""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def save_info_for(
    path: str,
    version: Optional[CivitVersion],
    model: Optional[CivitModel] = None,
    preferred_weight: float = 0.8,
    max_nsfw_level: int = 1,
    sha256: Optional[str] = None,
):
    """
    Save metadata JSON file for model.
    
    Args:
        path: Path to model file
        version: CivitVersion object (from hash lookup)
        model: Optional CivitModel object (from name lookup)
        preferred_weight: Default weight for LoRA activation
    """
    print_info(f" Writing info JSON for: {path}")
    
    desc = ""
    notes = ""
    trained_words = []
    model_id = 0
    version_id = 0
    base_model = ""
    model_name = ""
    model_type = ""
    creator_name = ""
    tags = []
    stats = None
    preview_url = ""
    allow_commercial = []
    nsfw_level = 0
    
    if version:
        desc = version.description or ""
        trained_words = list(version.trained_words) if version.trained_words else []
        model_id = version.model_id
        version_id = version.id
        base_model = version.base_model
        stats = version.stats
        nsfw_level = version.nsfw_level
        
        if version.images:
            preview_url = get_first_sfw_image_url(version.images, max_nsfw_level) or ""
        
        if not desc and model:
            desc = _clean_html(model.description)
    
    if model:
        model_name = model.name
        model_type = model.model_type
        tags = list(model.tags) if model.tags else []
        if model.creator:
            creator_name = model.creator.username
        if not stats:
            stats = model.stats
        if not model_id:
            model_id = model.id
        allow_commercial = list(model.allow_commercial_use) if model.allow_commercial_use else []
        if not preview_url and model.versions:
            for v in model.versions:
                url = get_first_sfw_image_url(v.images, max_nsfw_level) if v.images else None
                if url:
                    preview_url = url
                    break
    
    if sha256 is None:
        sha256 = calc_sha256_full(path)
    sha_val = sha256.upper() if isinstance(sha256, str) else ""
    
    data = {
        "description": _clean_html(desc),
        "sd version": _detect_sd_version(base_model),
        "activation text": ", ".join(trained_words),
        "preferred weight": preferred_weight,
        "notes": notes,
    }
    
    if model_name:
        data["name"] = model_name
    if model_type:
        data["type"] = model_type
    if base_model:
        data["baseModel"] = base_model
    
    if creator_name:
        data["creator"] = creator_name
    if tags:
        data["tags"] = tags
    
    if model_id:
        data["modelId"] = model_id
    if version_id:
        data["modelVersionId"] = version_id
    if sha_val:
        data["sha256"] = sha_val
    
    if stats:
        data["stats"] = {
            "downloadCount": stats.download_count,
            "favoriteCount": stats.favorite_count,
            "thumbsUpCount": stats.thumbs_up_count,
            "thumbsDownCount": stats.thumbs_down_count,
            "rating": stats.rating,
        }
    
    if preview_url:
        data["previewUrl"] = preview_url
    if allow_commercial:
        data["allowCommercialUse"] = allow_commercial
    if nsfw_level:
        data["nsfwLevel"] = nsfw_level
    
    info_path = Path(path).with_suffix(".json")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=info_path.parent,
            prefix=f".{info_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as info_file:
            temp_path = info_file.name
            json.dump(data, info_file, indent=4, ensure_ascii=False)
            info_file.flush()
            os.fsync(info_file.fileno())
        os.replace(temp_path, info_path)
    except OSError as error:
        print_error(f"Could not write model metadata for {path}: {error}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def fetch_missing(
    max_nsfw_level: int = 1,
    narrow_types: Optional[List[str]] = None,
    batch_size: int = 100,
    cancel_event=None,
):
    """
    Main entry point: fetch missing previews and metadata for all models.
    
    Args:
        max_nsfw_level: Maximum NSFW level for images (1=SFW, 5=XXX)
        narrow_types: Optional list of model types to process
        batch_size: Number of hashes per API request
    
    Returns:
        Dict with counts of updated previews and infos
    """
    print_info(" Start fetching missing previews and info from CivitAI")
    print_info(f" NSFW filter: max_level={max_nsfw_level}")
    
    all_paths = collect_paths(narrow_types, cancel_event)
    if cancel_event is not None and cancel_event.is_set():
        return {"previews_updated": 0, "info_updated": 0, "cancelled": True}

    candidate_paths = [
        path for path in all_paths
        if not has_preview(path) or needs_info_update(path)
    ]
    index = build_hash_index(candidate_paths, cancel_event)

    missing_preview_hashes = {
        model_hash
        for model_hash, paths in index.items()
        if any(not has_preview(path) for path in paths)
    }
    missing_info_hashes = {
        model_hash
        for model_hash, paths in index.items()
        if any(needs_info_update(path) for path in paths)
    }
    required_hashes = sorted(missing_preview_hashes | missing_info_hashes)
    
    print_info(f" Missing previews: {len(missing_preview_hashes)}, missing info: {len(missing_info_hashes)}")
    
    def batch_fetch_hashes(hashes: List[str]) -> List[CivitVersion]:
        results = []
        for i in tqdm(range(0, len(hashes), batch_size), desc="Fetch metadata", unit="batch"):
            if cancel_event is not None and cancel_event.is_set():
                break
            part = hashes[i:i + batch_size]
            try:
                versions = _fetch_versions_by_hash(part, cancel_event)
                results.extend(versions)
            except CivitaiAPIError as e:
                print_warning(f" Batch fetch failed: {e}")
            except Exception as e:
                print_warning(f" Batch fetch error: {e}")
            
            if i + batch_size < len(hashes):
                if cancel_event is not None:
                    if cancel_event.wait(RATE_LIMIT_DELAY):
                        break
                else:
                    time.sleep(RATE_LIMIT_DELAY)
        
        return results
    
    versions = batch_fetch_hashes(required_hashes)
    
    upd_previews = 0
    upd_info = 0
    fallback_paths = set()
    processed_preview_paths = set()
    processed_info_paths = set()

    for version in tqdm(versions, desc="Apply CivitAI data", unit="item"):
        if cancel_event is not None and cancel_event.is_set():
            break
        if not version:
            continue
        
        for f in version.files:
            if not f.hashes.sha256:
                continue
            key = f.hashes.sha256.lower()
            paths = index.get(key, [])
            if not paths:
                continue
            image_url = (
                get_first_sfw_image_url(version.images, max_nsfw_level)
                if key in missing_preview_hashes
                else None
            )
            for path in paths:
                if image_url and path not in processed_preview_paths and not has_preview(path):
                    print_info(f" Update preview for {path}")
                    if save_preview_for(path, image_url):
                        upd_previews += 1
                    processed_preview_paths.add(path)
                if (
                    key in missing_info_hashes
                    and path not in processed_info_paths
                    and needs_info_update(path)
                ):
                    print_info(f" Update info for {path}")
                    save_info_for(
                        path,
                        version,
                        max_nsfw_level=max_nsfw_level,
                        sha256=key,
                    )
                    processed_info_paths.add(path)
                    if not needs_info_update(path):
                        upd_info += 1
                    else:
                        fallback_paths.add(path)

    hashed_paths = {path for paths in index.values() for path in paths}
    hash_by_path = {
        path: model_hash
        for model_hash, paths in index.items()
        for path in paths
    }
    unhashed = [
        path for path in candidate_paths
        if path not in hashed_paths and (not has_preview(path) or needs_info_update(path))
    ]

    if unhashed:
        print_info(f" Fallback name-based matching for {len(unhashed)} unhashed files")
        try:
            prev2, info2 = _fetch_by_name(unhashed, max_nsfw_level, cancel_event)
            upd_previews += prev2
            upd_info += info2
        except Exception as e:
            print_warning(f" Fallback name lookup failed: {e}")

    # Only retry files that failed after hash lookup (e.g. description was empty)
    if fallback_paths and not (cancel_event is not None and cancel_event.is_set()):
        # Exclude files already handled by unhashed pass
        retry_paths = list(fallback_paths - set(unhashed))
        if retry_paths:
            print_info(f" Retry name-based lookup for {len(retry_paths)} files with incomplete info")
            try:
                prev2, info2 = _fetch_by_name(
                    retry_paths,
                    max_nsfw_level,
                    cancel_event,
                    hash_by_path,
                )
                upd_previews += prev2
                upd_info += info2
            except Exception as e:
                print_warning(f" Fallback paths lookup failed: {e}")
    
    result = {
        "previews_updated": upd_previews,
        "info_updated": upd_info,
        "cancelled": bool(cancel_event is not None and cancel_event.is_set()),
    }
    print_info(f" Finished: {result}")
    return result