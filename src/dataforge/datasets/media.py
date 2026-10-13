"""Resolve local image references and adapt them to API image payloads."""

import base64
import hashlib
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MIME_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}


def resolve_image(uri: str, root: str | Path) -> Path:
    root = Path(root).resolve()
    relative_path = Path(uri)
    if relative_path.is_absolute() or "://" in uri:
        raise ValueError(f"image reference must be a local relative path: {uri!r}")
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"image reference escapes media_root: {uri!r}")
    try:
        with Image.open(path) as image:
            if image.format not in MIME_TYPES:
                raise ValueError(f"unsupported image format {image.format!r}: {uri!r}")
            image.verify()
        with Image.open(path) as image:
            image.load()
    except (OSError, UnidentifiedImageError, SyntaxError) as error:
        raise ValueError(f"cannot read image {uri!r}: {error}") from error
    return path


def image_data(uri: str, root: str | Path) -> tuple[str, str]:
    path = resolve_image(uri, root)
    with Image.open(path) as image:
        mime_type = MIME_TYPES[image.format]
    return mime_type, base64.b64encode(path.read_bytes()).decode("ascii")


def image_sha256(uri: str, root: str | Path) -> str:
    path = resolve_image(uri, root)
    return hashlib.sha256(path.read_bytes()).hexdigest()
