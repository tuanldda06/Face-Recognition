from io import BytesIO
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings
from .errors import ApplicationError


def validate_rgb(image, settings: Settings):
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.dtype != np.uint8):
        raise ApplicationError("invalid_image", "Cần ảnh RGB uint8 (H,W,3).")
    height, width = image.shape[:2]
    validate_size(width, height, settings)
    return image


def validate_size(width, height, settings):
    if min(width, height) < 20:
        raise ApplicationError("image_too_small", "Ảnh phải có chiều rộng và cao ít nhất 20 pixel.")
    if max(width, height) > settings.max_dimension or width * height > settings.max_pixels:
        raise ApplicationError("image_too_large", "Ảnh vượt giới hạn kích thước/pixel.", 413)


def decode_image(data: bytes, settings: Settings):
    if len(data) > settings.max_file_bytes:
        raise ApplicationError("file_too_large", "Mỗi ảnh tối đa 8 MiB.", 413)
    if not data:
        raise ApplicationError("invalid_image", "File ảnh rỗng.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"} or getattr(source, "n_frames", 1) != 1:
                    raise ApplicationError("unsupported_image", "Chỉ nhận JPEG, PNG hoặc WebP tĩnh.", 415)
                validate_size(*source.size, settings)
                source.load()
                rgb = np.array(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8)
    except ApplicationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ApplicationError("invalid_image", "Không giải mã được ảnh hợp lệ.") from error
    return validate_rgb(rgb, settings)
