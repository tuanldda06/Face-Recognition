

from __future__ import annotations

import numpy as np


def representative_from_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Nhận float32 (N,512) đơn vị, trả float32 (512,) từ mean float64 rồi L2.

    Giữ thứ tự validation và exception gốc. Không chuẩn hóa lại từng hàng hoặc
    thay đổi input. Sai kiểu/dtype gây TypeError; shape, finite, norm sai gây
    ValueError. Mean có norm <= 1e-12 bị từ chối vì không ổn định để chuẩn hóa.
    Hàm chỉ trả vector; ID, num_images và model_version do bên gọi quản lý.
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("Embeddings phải là numpy.ndarray.")
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] != 512:
        raise ValueError("Embeddings phải có shape (N,512), N > 0.")
    if embeddings.dtype != np.float32:
        raise TypeError("Embeddings phải là float32.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings chứa NaN hoặc Inf.")
    row_norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.allclose(row_norms, 1.0, atol=1e-6, rtol=0):
        raise ValueError("Embedding từng ảnh chưa có norm xấp xỉ 1.")
    mean_embedding = np.mean(embeddings, axis=0, dtype=np.float64)
    if not np.isfinite(mean_embedding).all():
        raise ValueError("Embedding trung bình chứa NaN hoặc Inf.")
    norm = float(np.linalg.norm(mean_embedding))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Norm embedding trung bình bằng 0, quá nhỏ hoặc không hữu hạn.")
    representative = (mean_embedding / norm).astype(np.float32)
    if (representative.shape != (512,) or not np.isfinite(representative).all()
            or not np.isclose(np.linalg.norm(representative), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("Embedding đại diện không hợp lệ.")
    return representative
