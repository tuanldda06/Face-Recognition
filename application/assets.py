"""Verify only the artifact identified by the repository's provenance record."""
import hashlib
from pathlib import Path

from .config import MODEL_PATH, MODEL_SHA256, MODEL_SIZE
from .errors import ApplicationError


def verify_model(path: Path = MODEL_PATH):
    if not path.is_file():
        raise ApplicationError("model_missing", "Thiếu FaceNet ONNX. Xem application/README.md.", 503)
    with path.open("rb") as stream:
        prefix = stream.read(200)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ApplicationError("model_lfs_pointer", "FaceNet đang là Git LFS pointer; cần khôi phục model gốc.", 503)
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != MODEL_SIZE or digest != MODEL_SHA256:
        raise ApplicationError("model_checksum", "FaceNet không khớp dung lượng/SHA-256 bàn giao.", 503)
    return {"sha256": digest, "bytes": path.stat().st_size}
