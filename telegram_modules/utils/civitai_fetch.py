import os
import json
import glob
import tempfile
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm

import requests
import folder_paths

from .hash import calc_sha256_full
try:
    from ..utils.log import print_info, print_warning, print_error
except Exception:
    def print_info(msg):
        print(f"  {msg}")
    def print_warning(msg):
        print(f"  WARNING: {msg}")
    def print_error(msg):
        print(f"  ERROR: {msg}")

TYPES = {
    "Checkpoint":      ("checkpoints", ["safetensors", "ckpt"]),
    "LORA":            ("loras",      ["safetensors", "pt"]),
    "LoCon":           ("loras",      ["safetensors", "pt"]),
    "TextualInversion":("embeddings", ["safetensors", "pt", "bin"]),
    "VAE":             ("vae",        ["safetensors", "ckpt", "pt"]),
    "Upscaler":        ("upscale_models", ["safetensors", "ckpt", "pt"]),
}

PREVIEW_EXTS = [".jpg", ".png", ".jpeg", ".gif"]
PREVIEW_EXTS = PREVIEW_EXTS + [".preview" + x for x in PREVIEW_EXTS]

USER_AGENT = "CivitaiLink:Automatic1111"
BASE_URL = os.getenv("CIVITAI_ENDPOINT", "https://civitai.com/api/v1")

def has_preview(path):
    stem = os.path.splitext(path)[0]
    return any(os.path.exists(stem + ext) for ext in PREVIEW_EXTS)

def has_info(path):
    return os.path.isfile(os.path.splitext(path)[0] + ".json")

def _basename_key(path):
    return os.path.splitext(os.path.basename(path))[0].strip().lower()

def iter_files_for_type(folder_type, exts):
    for base in folder_paths.get_folder_paths(folder_type):
        for ext in exts:
            for p in glob.glob(os.path.join(base, "**", f"*.{ext}"), recursive=True):
                if not os.path.isdir(p):
                    yield p

def collect_paths(narrow_types=None):
    print_info("[Telegram Sender] Scanning ComfyUI model folders for resources...")
    all_paths = set()
    for civ_type, (folder_type, exts) in TYPES.items():
        if narrow_types and civ_type not in narrow_types:
            continue
        print_info(f"  Type {civ_type}: folders={folder_paths.get_folder_paths(folder_type)}")
        paths = list(iter_files_for_type(folder_type, exts))
        print_info(f"  {civ_type}: total={len(paths)}")
        for p in paths:
            all_paths.add(p)
    print_info(f"  Total files discovered: {len(all_paths)}")
    return list(all_paths)

def build_hash_index(paths):
    mapping = {}
    if not paths:
        return mapping
    for path in tqdm(paths, desc="Hash files", unit="file"):
        try:
            full_hash = calc_sha256_full(path)
        except Exception:
            full_hash = ""
        if not full_hash:
            continue
        mapping[full_hash.lower()] = {"path": path}
    print_info(f"  Hash index entries: {len(mapping)}")
    return mapping

def _req(endpoint, method="GET", data=None, params=None, headers=None):
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
    url = BASE_URL + endpoint
    print_info(f"  HTTP {method} {url} params={params} data={'set' if data else 'none'}")
    resp = requests.request(method, url, data=data, params=params, headers=headers)
    if resp.status_code != 200:
        print_error(f"  HTTP error {resp.status_code}: {resp.text[:200]}")
        raise Exception(f"Error: {resp.status_code}")
    return resp.json()

def _get_all_by_hash(hashes):
    # Ensure payload is an array of SHA256 hashes (strings)
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
    return _req("/model-versions/by-hash", method="POST", data=cleaned)

def _fetch_by_name(paths, nsfw=True):
    updated_prev = 0
    updated_info = 0
    for path in tqdm(list(paths), desc="Name lookup", unit="file"):
        name = _basename_key(path)
        try:
            resp = _req("/models", params={"query": name, "pageSize": 5})
        except Exception:
            continue
        items = resp.get("items", []) if isinstance(resp, dict) else []
        if not items:
            continue
        best = None
        for item in items:
            mvs = item.get("modelVersions", []) or []
            for mv in mvs:
                files = mv.get("files", []) or []
                for f in files:
                    fname = f.get("name") or ""
                    if _basename_key(fname) == name:
                        best = {"images": mv.get("images", []) or item.get("images", []), "obj": item}
                        break
                if best:
                    break
            if best:
                break
        if not best:
            continue
        images = best["images"]
        if not nsfw:
            images = [i for i in images if i.get("nsfw") is False]
        if images:
            save_preview_for(path, images[0]["url"])
            updated_prev += 1
        save_info_for(path, best["obj"])
        updated_info += 1
    return updated_prev, updated_info

def _is_video_url(url):
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

def _download_stream(url, dest):
    print_info(f"  Downloading preview: {url} -> {dest}")
    resp = requests.get(url, stream=True, headers={"User-Agent": USER_AGENT})
    total = int(resp.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f, tqdm(total=total or None, unit="B", unit_scale=True, unit_divisor=1024, desc=f"Download {os.path.basename(dest)}") as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            f.write(chunk)
            bar.update(len(chunk))

def _save_first_frame_gif(url, out_png):
    print_info(f"  Extract first GIF frame: {url} -> {out_png}")
    fd, tmp = tempfile.mkstemp(suffix=".gif")
    os.close(fd)
    try:
        _download_stream(url, tmp)
        try:
            from PIL import Image
            im = Image.open(tmp)
            im.seek(0)
            im.convert("RGB").save(out_png)
            return
        except Exception:
            pass
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

def _save_first_frame_video(url, out_png):
    print_info(f"  Extract first video frame: {url} -> {out_png}")
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        _download_stream(url, tmp)
        try:
            subprocess.run(["ffmpeg", "-y", "-i", tmp, "-vframes", "1", out_png], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception:
            pass
        try:
            import imageio.v3 as iio
            from PIL import Image
            arr = iio.imread(tmp, index=0)
            Image.fromarray(arr).save(out_png)
            return
        except Exception:
            pass
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

def save_preview_for(path, image_url):
    dest = os.path.splitext(path)[0] + ".preview.png"
    lower = image_url.lower()
    if lower.endswith(".gif"):
        _save_first_frame_gif(image_url, dest)
        return
    if _is_video_url(image_url):
        _save_first_frame_video(image_url, dest)
        return
    _download_stream(image_url, dest)

def save_info_for(path, civ_obj, preferred_weight=0.8):
    print_info(f"  Writing info JSON for: {path}")
    base = civ_obj.get("baseModel", "") or ""
    if "SDXL" in base:
        sdv = "SDXL"
    elif "SD 2" in base:
        sdv = "SD2"
    elif "SD 1" in base:
        sdv = "SD1"
    else:
        sdv = "unknown"
    data = {
        "description": civ_obj.get("description", ""),
        "sd version": sdv,
        "activation text": ", ".join(civ_obj.get("trainedWords", []) or []),
        "preferred weight": preferred_weight,
        "notes": "",
    }
    Path(path).with_suffix(".json").write_text(json.dumps(data, indent=4))

def fetch_missing(nsfw=True, narrow_types=None, batch=100, api_key_env="CIVITAI_API_KEY"):
    print_info("  Start fetching missing previews and info from Civitai")
    # 1) Собрать ВСЕ файлы и построить индекс хэшей
    all_paths = collect_paths(narrow_types)
    index = build_hash_index(all_paths)
    # 2) Определить, у каких отсутствуют превью/инфо
    missing_preview_hashes = [h for h, v in index.items() if not has_preview(v["path"])]
    missing_info_hashes = [h for h, v in index.items() if not has_info(v["path"])]
    print_info(f"  Missing previews: {len(missing_preview_hashes)}, missing info: {len(missing_info_hashes)}")

    def batched_fetch(hashes):
        results = []
        for i in tqdm(range(0, len(hashes), batch), desc="Fetch metadata", unit="batch"):
            part = hashes[i:i+batch]
            try:
                results.extend(_get_all_by_hash(part))
            except Exception as e:
                print_warning(f"  Batch fetch failed: {e}")
            
        return results

    preview_results = batched_fetch(missing_preview_hashes)
    info_results = batched_fetch(missing_info_hashes)

    upd_previews = 0
    for r in tqdm(preview_results, desc="Previews", unit="item"):
        if not r:
            continue
        imgs = r.get("images", []) or []
        if not nsfw:
            imgs = [i for i in imgs if i.get("nsfw") is False]
        if not imgs:
            continue
        image_url = imgs[0]["url"]
        for f in r.get("files", []):
            h = f.get("hashes", {}).get("SHA256")
            if not h:
                continue
            key = h.lower()
            if key in index and not has_preview(index[key]["path"]):
                print_info(f"  Update preview for {index[key]['path']}")
                save_preview_for(index[key]["path"], image_url)
                upd_previews += 1

    upd_info = 0
    for r in tqdm(info_results, desc="Infos", unit="item"):
        if not r:
            continue
        for f in r.get("files", []):
            h = f.get("hashes", {}).get("SHA256")
            if not h:
                continue
            key = h.lower()
            if key in index and not has_info(index[key]["path"]):
                print_info(f"  Update info for {index[key]['path']}")
                save_info_for(index[key]["path"], r)
                upd_info += 1

    # Fallback: name-based lookup for files we couldn't hash or where API lacks SHA256
    hashed_paths = {v["path"] for v in index.values()}
    fallback_paths = [p for p in all_paths if p not in hashed_paths and (not has_preview(p) or not has_info(p))]
    if fallback_paths:
        print_info(f"  Fallback name-based matching for {len(fallback_paths)} files")
        prev2, info2 = _fetch_by_name(fallback_paths, nsfw=nsfw)
        upd_previews += prev2
        upd_info += info2

    # 4) Фолбэк по имени только для оставшихся после запроса по хэшу
    hashed_paths = {v["path"] for v in index.values()}
    still_missing_paths = [p for p in all_paths if (not has_preview(p) or not has_info(p))]
    if still_missing_paths:
        print_info(f"  Fallback name-based matching for {len(still_missing_paths)} files")
        prev2, info2 = _fetch_by_name(still_missing_paths, nsfw=nsfw)
        upd_previews += prev2
        upd_info += info2
    result = {"previews_updated": upd_previews, "info_updated": upd_info}
    print_info(f"  Finished: {result}")
    return result
