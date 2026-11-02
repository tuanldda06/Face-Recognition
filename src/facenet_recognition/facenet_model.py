
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .contracts import validate_face


def load_facenet_model(model_path: str | Path) -> cv2.dnn.Net:
    """Nạp ONNX CPU từ path tuyệt đối; thiếu file hoặc model lỗi được báo rõ.

    Path tương đối bị từ chối bằng ValueError để cùng lời gọi không phụ thuộc
    current working directory. Không tự chọn hoặc thay model.
    """
    model_path = Path(model_path)
    if not model_path.is_absolute():
        raise ValueError("Đường dẫn model phải là đường dẫn tuyệt đối.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model FaceNet: {model_path}")
    try:
        net = cv2.dnn.readNetFromONNX(str(model_path))
        if net.empty():
            raise RuntimeError("OpenCV trả về model rỗng.")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except cv2.error as error:
        raise RuntimeError(f"Không nạp được FaceNet ONNX: {model_path}") from error
    return net


def preprocess_face(face_rgb: np.ndarray) -> np.ndarray:
    """Trả float32 contiguous (1,3,160,160), giữ nguyên RGB uint8 đầu vào."""
    validate_face(face_rgb)
    image = (face_rgb.astype(np.float32) - np.float32(127.5)) / np.float32(128.0)
    batch = image.transpose(2, 0, 1)[np.newaxis, ...]
    batch = np.ascontiguousarray(batch, dtype=np.float32)
    if not np.isfinite(batch).all():
        raise ValueError("Tensor preprocessing chứa NaN hoặc Inf.")
    return batch


def extract_embedding(face_rgb: np.ndarray, net: cv2.dnn.Net) -> np.ndarray:
    """Nhận RGB uint8 (160,160,3), trả float32 (512,) chuẩn hóa L2."""
    batch = preprocess_face(face_rgb)
    try:
        net.setInput(batch, "in:0")
        output = net.forward("out:0")
    except cv2.error as error:
        raise RuntimeError("FaceNet inference thất bại trên OpenCV DNN CPU.") from error

    if not isinstance(output, np.ndarray):
        raise TypeError("Output FaceNet phải là numpy.ndarray.")
    if output.shape != (1, 512):
        raise ValueError(f"Output phải có shape (1, 512), nhận được {output.shape}.")
    if output.dtype != np.float32:
        raise TypeError(f"Output phải là float32, nhận được {output.dtype}.")
    if not np.isfinite(output).all():
        raise ValueError("Output FaceNet chứa NaN hoặc Inf.")

    # Tính norm bằng float64 như notebook, chỉ ép float32 sau L2.
    embedding = output[0].astype(np.float64)
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Embedding có L2 norm bằng 0, quá nhỏ hoặc không hữu hạn.")

    embedding = (embedding / norm).astype(np.float32)
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding sau L2 normalization không hợp lệ.")
    return embedding


# Giữ alias notebook 02; extract_embeddings tra alias này khi chạy.
get_face_embedding = extract_embedding


def extract_embeddings(
    face_images: list[np.ndarray] | tuple[np.ndarray, ...], net: cv2.dnn.Net
) -> np.ndarray:
    """Trả float32 (N,512) theo thứ tự; [] trả (0,512), lỗi ghi chỉ số ảnh."""
    if not isinstance(face_images, (list, tuple)):
        raise TypeError("face_images phải là list hoặc tuple có thứ tự các ảnh RGB.")
    if len(face_images) == 0:
        return np.empty((0, 512), dtype=np.float32)

    embeddings = []
    for index, face_rgb in enumerate(face_images):
        try:
            embedding = get_face_embedding(face_rgb, net)
        except (TypeError, ValueError, RuntimeError) as error:
            raise type(error)(
                f"Ảnh tại chỉ số {index} (đếm từ 0): {error}"
            ) from error
        embeddings.append(embedding)
    return np.stack(embeddings, axis=0)
