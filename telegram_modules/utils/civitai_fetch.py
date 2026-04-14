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
from typing import Optional, List, Dict, Any, Iterator, Set
from dataclasses import dataclass, field
from datetime import datetime

from tqdm import tqdm
import requests
import folder_paths

from .hash import calc_sha256_full
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
except Exception:
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
    mtime: float
    files: List[str] = field(default_factory=list)


_dir_cache: Dict[str, DirCache] = {}


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
    except Exception:
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


def _scan_directory_recursive(path: str, exts: List[str]) -> Iterator[str]:
    """
    Fast recursive directory scanner using os.scandir.
    Uses mtime caching to skip unchanged directories.
    """
    try:
        current_mtime = os.path.getmtime(path)
    except OSError:
        return
    
    cached = _dir_cache.get(path)
    if cached and cached.mtime == current_mtime:
        yield from cached.files
        return
    
    files = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if _should_skip(entry.name):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    yield from _scan_directory_recursive(entry.path, exts)
                elif entry.is_file(follow_symlinks=False):
                    lower_name = entry.name.lower()
                    for ext in exts:
                        if lower_name.endswith(f".{ext}"):
                            files.append(entry.path)
                            yield entry.path
                            break
    except PermissionError:
        pass
    except OSError:
        pass
    
    _dir_cache[path] = DirCache(mtime=current_mtime, files=files)


def iter_files_for_type(folder_type: str, exts: List[str]) -> Iterator[str]:
    """Iterate over all model files of a given type."""
    for base in folder_paths.get_folder_paths(folder_type):
        if not os.path.isdir(base):
            continue
        yield from _scan_directory_recursive(base, exts)


def collect_paths(narrow_types: Optional[List[str]] = None) -> List[str]:
    print_info(" Scanning ComfyUI model folders for resources...")
    all_paths = set()
    for civ_type, (folder_type, exts) in TYPES.items():
        if narrow_types and civ_type not in narrow_types:
            continue
        print_info(f" Type {civ_type}: folders={folder_paths.get_folder_paths(folder_type)}")
        paths = list(iter_files_for_type(folder_type, exts))
        print_info(f" {civ_type}: total={len(paths)}")
        for p in paths:
            all_paths.add(p)
    print_info(f" Total files discovered: {len(all_paths)}")
    return list(all_paths)


def build_hash_index(paths: List[str]) -> Dict[str, dict]:
    """
    Build hash index for model files.
    Uses ThreadPoolExecutor for parallel hashing.
    """
    mapping = {}
    if not paths:
        return mapping
    
    def hash_file(path: str) -> Optional[tuple]:
        try:
            h = calc_sha256_full(path)
            if h:
                return (h.lower(), {"path": path})
        except Exception:
            pass
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
                result = future.result()
                if result:
                    h, data = result
                    mapping[h] = data
    else:
        for path in tqdm(paths, desc="Hash files", unit="file"):
            result = hash_file(path)
            if result:
                h, data = result
                mapping[h] = data
    
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
) -> dict:
    """
    Make HTTP request to CivitAI API with retry support.
    
    Raises:
        CivitaiAPIError: On non-retryable errors or max retries exceeded
    """
    if headers is None:
        headers = {}
    
    headers["User-Agent"] = USER_AGENT
    
    api_key = os.getenv("CIVITAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    if data is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(data)
    
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    if params is None:
        params = {}
    
    url = API_BASE_URL + endpoint
    
    try:
        resp = requests.request(method, url, data=data, params=params, headers=headers, timeout=30)
    except requests.exceptions.Timeout:
        if retry_count < MAX_RETRIES:
            delay = RETRY_BASE_DELAY * (2 ** retry_count)
            print_warning(f" Request timeout, retrying in {delay}s... ({retry_count + 1}/{MAX_RETRIES})")
            time.sleep(delay)
            return _make_request(endpoint, method, data, params, headers, retry_count + 1)
        raise CivitaiAPIError(0, "Request timeout after max retries")
    except requests.exceptions.ConnectionError as e:
        if retry_count < MAX_RETRIES:
            delay = RETRY_BASE_DELAY * (2 ** retry_count)
            print_warning(f" Connection error, retrying in {delay}s... ({retry_count + 1}/{MAX_RETRIES})")
            time.sleep(delay)
            return _make_request(endpoint, method, data, params, headers, retry_count + 1)
        raise CivitaiAPIError(0, f"Connection error: {e}")
    
    if resp.status_code == 200:
        return resp.json()
    
    if resp.status_code == 404:
        return {}
    
    if resp.status_code in (429, 502, 503, 504):
        if retry_count < MAX_RETRIES:
            delay = RETRY_BASE_DELAY * (2 ** retry_count)
            print_warning(f" Rate limited/server error {resp.status_code}, retrying in {delay}s... ({retry_count + 1}/{MAX_RETRIES})")
            time.sleep(delay)
            return _make_request(endpoint, method, data, params, headers, retry_count + 1)
    
    print_error(f" HTTP error {resp.status_code}: {resp.text[:200]}")
    raise CivitaiAPIError(resp.status_code, resp.text[:500])


def _fetch_versions_by_hash(hashes: List[str]) -> List[CivitVersion]:
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
        try:
            cleaned.append(str(h).strip().lower())
        except Exception:
            continue
    
    if not cleaned:
        return []
    
    raw_data = _make_request("/model-versions/by-hash", method="POST", data=cleaned)
    
    if not raw_data:
        return []
    
    if isinstance(raw_data, dict) and "id" in raw_data:
        return [CivitVersion.from_dict(raw_data)]
    
    if isinstance(raw_data, list):
        return [CivitVersion.from_dict(item) for item in raw_data if item]
    
    return []


def _fetch_model_by_name(query: str, max_nsfw_level: int = 1) -> Optional[CivitModel]:
    """
    Search for model by name query.
    Returns first matching model or None.
    """
    raw_data = _make_request("/models", params={"query": query, "pageSize": 5})
    
    items = raw_data.get("items", []) if isinstance(raw_data, dict) else []
    if not items:
        return None
    
    return CivitModel.from_dict(items[0]) if items else None


def _fetch_by_name(
    paths: List[str],
    max_nsfw_level: int = 1,
) -> tuple:
    """
    Fetch metadata for models using name-based lookup (fallback method).
    Returns (previews_updated, infos_updated) count.
    """
    updated_prev = 0
    updated_info = 0
    
    for path in tqdm(list(paths), desc="Name lookup", unit="file"):
        name = _basename_key(path)
        
        time.sleep(RATE_LIMIT_DELAY)
        
        model = _fetch_model_by_name(name, max_nsfw_level)
        if not model:
            continue
        
        best_version = None
        best_image_url = None
        
        for version in model.versions:
            for f in version.files:
                if _basename_key(f.name) == name:
                    best_version = version
                    break
            if best_version:
                break
        
        if not best_version and model.versions:
            best_version = model.versions[0]
        
        if best_version:
            best_image_url = get_first_sfw_image_url(best_version.images, max_nsfw_level)
        
        if best_image_url:
            save_preview_for(path, best_image_url)
            updated_prev += 1
        
        if best_version or model:
            save_info_for(path, best_version or model, model)
            updated_info += 1
    
    return updated_prev, updated_info


def _is_video_url(url: str) -> bool:
    lower = url.lower()
    for ext in [".mp4", ".webm", ".mov", ".mkv"]:
        if lower.endswith(ext):
            return True
    try:
        head = requests.head(url, allow_redirects=True, timeout=5)
        ctype = head.headers.get("Content-Type", "")
        if ctype.startswith("video/"):
            return True
    except Exception:
        pass
    return False


def _download_file(url: str, dest: str):
    """Download file from URL to destination with progress bar."""
    print_info(f" Downloading preview: {url} -> {dest}")
    resp = requests.get(url, stream=True, headers={"User-Agent": USER_AGENT}, timeout=60)
    total = int(resp.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    with open(dest, "wb") as f:
        with tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Download {os.path.basename(dest)}"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))


def _extract_gif_frame(url: str, out_png: str) -> bool:
    """Extract first frame from GIF and save as PNG."""
    print_info(f" Extract first GIF frame: {url} -> {out_png}")
    fd, tmp = tempfile.mkstemp(suffix=".gif")
    os.close(fd)
    try:
        _download_file(url, tmp)
        try:
            from PIL import Image
            im = Image.open(tmp)
            im.seek(0)
            im.convert("RGB").save(out_png)
            return True
        except Exception:
            pass
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return False


def _extract_video_frame(url: str, out_png: str) -> bool:
    """Extract first frame from video and save as PNG."""
    print_info(f" Extract first video frame: {url} -> {out_png}")
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        _download_file(url, tmp)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp, "-vframes", "1", out_png],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except Exception:
            pass
        try:
            import imageio.v3 as iio
            from PIL import Image
            arr = iio.imread(tmp, index=0)
            Image.fromarray(arr).save(out_png)
            return True
        except Exception:
            pass
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return False


def save_preview_for(path: str, image_url: str):
    """Save preview image for model file."""
    dest = os.path.splitext(path)[0] + ".preview.png"
    lower = image_url.lower()
    
    if lower.endswith(".gif"):
        _extract_gif_frame(image_url, dest)
        return
    
    if _is_video_url(image_url):
        _extract_video_frame(image_url, dest)
        return
    
    _download_file(image_url, dest)


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
            preview_url = version.images[0].url
        
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
                if v.images:
                    preview_url = v.images[0].url
                    break
    
    try:
        sha_val = calc_sha256_full(path)
        sha_val = sha_val.upper() if isinstance(sha_val, str) else ""
    except Exception:
        sha_val = ""
    
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
    
    Path(path).with_suffix(".json").write_text(json.dumps(data, indent=4, ensure_ascii=False))


def fetch_missing(
    max_nsfw_level: int = 1,
    narrow_types: Optional[List[str]] = None,
    batch_size: int = 100,
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
    
    all_paths = collect_paths(narrow_types)
    index = build_hash_index(all_paths)
    
    missing_preview_hashes = [h for h, v in index.items() if not has_preview(v["path"])]
    missing_info_hashes = [h for h, v in index.items() if needs_info_update(v["path"])]
    
    print_info(f" Missing previews: {len(missing_preview_hashes)}, missing info: {len(missing_info_hashes)}")
    
    def batch_fetch_hashes(hashes: List[str]) -> List[CivitVersion]:
        results = []
        for i in tqdm(range(0, len(hashes), batch_size), desc="Fetch metadata", unit="batch"):
            part = hashes[i:i + batch_size]
            try:
                versions = _fetch_versions_by_hash(part)
                results.extend(versions)
            except CivitaiAPIError as e:
                print_warning(f" Batch fetch failed: {e}")
            except Exception as e:
                print_warning(f" Batch fetch error: {e}")
            
            if i + batch_size < len(hashes):
                time.sleep(RATE_LIMIT_DELAY)
        
        return results
    
    preview_versions = batch_fetch_hashes(missing_preview_hashes)
    info_versions = batch_fetch_hashes(missing_info_hashes)
    
    upd_previews = 0
    for version in tqdm(preview_versions, desc="Previews", unit="item"):
        if not version:
            continue
        
        image_url = get_first_sfw_image_url(version.images, max_nsfw_level)
        if not image_url:
            continue
        
        for f in version.files:
            if not f.hashes.sha256:
                continue
            key = f.hashes.sha256.lower()
            if key in index and not has_preview(index[key]["path"]):
                print_info(f" Update preview for {index[key]['path']}")
                save_preview_for(index[key]["path"], image_url)
                upd_previews += 1
    
    upd_info = 0
    fallback_paths = set()
    
    for version in tqdm(info_versions, desc="Infos", unit="item"):
        if not version:
            continue
        
        for f in version.files:
            if not f.hashes.sha256:
                continue
            key = f.hashes.sha256.lower()
            if key in index and needs_info_update(index[key]["path"]):
                print_info(f" Update info for {index[key]['path']}")
                save_info_for(index[key]["path"], version)
                upd_info += 1
                
                if needs_info_update(index[key]["path"]):
                    fallback_paths.add(index[key]["path"])
    
    hashed_paths = {v["path"] for v in index.values()}
    unhashed = [p for p in all_paths if p not in hashed_paths and (not has_preview(p) or needs_info_update(p))]
    
    if unhashed:
        print_info(f" Fallback name-based matching for {len(unhashed)} files")
        prev2, info2 = _fetch_by_name(unhashed, max_nsfw_level)
        upd_previews += prev2
        upd_info += info2
    
    still_missing = [p for p in all_paths if not has_preview(p) or needs_info_update(p)]
    if still_missing:
        print_info(f" Additional name-based lookup for {len(still_missing)} remaining files")
        prev2, info2 = _fetch_by_name(still_missing, max_nsfw_level)
        upd_previews += prev2
        upd_info += info2
    
    if fallback_paths:
        prev2, info2 = _fetch_by_name(list(fallback_paths), max_nsfw_level)
        upd_previews += prev2
        upd_info += info2
    
    result = {"previews_updated": upd_previews, "info_updated": upd_info}
    print_info(f" Finished: {result}")
    return result
