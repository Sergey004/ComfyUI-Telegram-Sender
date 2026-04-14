"""
CivitAI API response models for type-safe parsing.
Designed for ComfyUI Telegram Sender - minimal models for metadata fetching.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CivitImage:
    """Image attached to a model version."""
    url: str = ""
    width: int = 0
    height: int = 0
    nsfw_level: int = 0
    image_type: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "CivitImage":
        if not data:
            return cls()
        return cls(
            url=data.get("url", ""),
            width=data.get("width", 0) or 0,
            height=data.get("height", 0) or 0,
            nsfw_level=data.get("nsfwLevel", data.get("nsfw_level", 0)) or 0,
            image_type=data.get("type", ""),
        )


@dataclass
class CivitFileHashes:
    """Hash values for a model file."""
    sha256: Optional[str] = None
    autov1: Optional[str] = None
    autov2: Optional[str] = None
    blake3: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "CivitFileHashes":
        if not data:
            return cls()
        return cls(
            sha256=data.get("SHA256", data.get("sha256")),
            autov1=data.get("AutoV1", data.get("autov1")),
            autov2=data.get("AutoV2", data.get("autov2")),
            blake3=data.get("BLAKE3", data.get("blake3")),
        )


@dataclass
class CivitFile:
    """File in a model version."""
    name: str = ""
    file_type: str = ""
    size_kb: float = 0.0
    hashes: CivitFileHashes = field(default_factory=CivitFileHashes)
    download_url: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "CivitFile":
        if not data:
            return cls()
        return cls(
            name=data.get("name", ""),
            file_type=data.get("type", ""),
            size_kb=data.get("sizeKB", data.get("size_kb", 0)) or 0,
            hashes=CivitFileHashes.from_dict(data.get("hashes", {})),
            download_url=data.get("downloadUrl", data.get("download_url", "")),
        )


@dataclass
class CivitStats:
    """Statistics for a model or version."""
    download_count: int = 0
    favorite_count: int = 0
    thumbs_up_count: int = 0
    thumbs_down_count: int = 0
    rating: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "CivitStats":
        if not data:
            return cls()
        return cls(
            download_count=data.get("downloadCount", data.get("download_count", 0)) or 0,
            favorite_count=data.get("favoriteCount", data.get("favorite_count", 0)) or 0,
            thumbs_up_count=data.get("thumbsUpCount", data.get("thumbs_up_count", 0)) or 0,
            thumbs_down_count=data.get("thumbsDownCount", data.get("thumbs_down_count", 0)) or 0,
            rating=data.get("rating", 0.0) or 0.0,
        )


@dataclass
class CivitCreator:
    """Creator/author of a model."""
    username: str = ""
    image: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "CivitCreator":
        if not data:
            return cls()
        return cls(
            username=data.get("username", ""),
            image=data.get("image", ""),
        )


@dataclass
class CivitVersion:
    """
    Model version - response from /model-versions/by-hash endpoint.
    This is the primary data structure returned when looking up by hash.
    """
    id: int = 0
    model_id: int = 0
    name: str = ""
    base_model: str = ""
    description: str = ""
    trained_words: list = field(default_factory=list)
    files: list = field(default_factory=list)
    images: list = field(default_factory=list)
    stats: CivitStats = field(default_factory=CivitStats)
    nsfw_level: int = 0
    download_url: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "CivitVersion":
        if not data:
            return cls()
        files_data = data.get("files", []) or []
        images_data = data.get("images", []) or []
        trained = data.get("trainedWords", data.get("trained_words", [])) or []
        if isinstance(trained, str):
            trained = [trained]
        
        return cls(
            id=data.get("id", 0) or 0,
            model_id=data.get("modelId", data.get("model_id", 0)) or 0,
            name=data.get("name", ""),
            base_model=data.get("baseModel", data.get("base_model", "")),
            description=data.get("description", "") or "",
            trained_words=list(trained),
            files=[CivitFile.from_dict(f) for f in files_data],
            images=[CivitImage.from_dict(i) for i in images_data],
            stats=CivitStats.from_dict(data.get("stats", {})),
            nsfw_level=data.get("nsfwLevel", data.get("nsfw_level", 0)) or 0,
            download_url=data.get("downloadUrl", data.get("download_url", "")),
        )


@dataclass
class CivitModel:
    """
    Full model data - response from /models endpoint.
    Used for name-based fallback lookups.
    """
    id: int = 0
    name: str = ""
    model_type: str = ""
    description: str = ""
    tags: list = field(default_factory=list)
    creator: CivitCreator = field(default_factory=CivitCreator)
    versions: list = field(default_factory=list)
    stats: CivitStats = field(default_factory=CivitStats)
    nsfw_level: int = 0
    allow_commercial_use: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "CivitModel":
        if not data:
            return cls()
        versions_data = data.get("modelVersions", data.get("versions", [])) or []
        commercial = data.get("allowCommercialUse", data.get("allow_commercial_use", [])) or []
        if isinstance(commercial, str):
            commercial = [commercial] if commercial else []
        
        return cls(
            id=data.get("id", 0) or 0,
            name=data.get("name", ""),
            model_type=data.get("type", ""),
            description=data.get("description", "") or "",
            tags=data.get("tags", []) or [],
            creator=CivitCreator.from_dict(data.get("creator", {})),
            versions=[CivitVersion.from_dict(v) for v in versions_data],
            stats=CivitStats.from_dict(data.get("stats", {})),
            nsfw_level=data.get("nsfwLevel", data.get("nsfw_level", 0)) or 0,
            allow_commercial_use=list(commercial),
        )


def filter_images_by_nsfw_level(images: list, max_level: int = 1) -> list:
    """
    Filter images by NSFW level.
    
    NSFW Levels:
        1 = None (SFW)
        2 = Soft
        3 = Mature
        4 = X
        5 = XXX
    
    Args:
        images: List of CivitImage objects
        max_level: Maximum allowed NSFW level (default 1 = SFW only)
    
    Returns:
        Filtered list of images
    """
    return [img for img in images if img.nsfw_level <= max_level]


def get_first_sfw_image_url(images: list, max_level: int = 1) -> Optional[str]:
    """Get URL of first image within NSFW threshold."""
    filtered = filter_images_by_nsfw_level(images, max_level)
    return filtered[0].url if filtered else None
