"""Điều phối nhận diện ảnh đã alignment, giữ luồng của notebook 10.

Caller cung cấp model đã nạp và registry trong RAM. Nếu dùng SQLite, caller gọi
database.load_registered_embeddings(db_path, expected_model_version) trước;
reader kiểm tra version và chỉ đọc. Registry truyền trực tiếp không mang metadata
version: caller chịu trách nhiệm về cấu hình embedding/alignment tương thích.
Không mặc định registry IMM/ASF tương thích MTCNN. Threshold luôn tường minh.
Module không tự nạp model, đọc database, sửa ảnh hoặc thực hiện I/O nghiệp vụ.
"""

from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np

from .contracts import validate_aligned_face, validate_mtcnn_output
from .facenet_model import extract_embedding
from .matching import detect_unknown, validate_threshold


def recognize_aligned_face(
    face_rgb: np.ndarray,
    net: cv2.dnn.Net,
    registered_embeddings: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, object]:
    """Nhận RGB uint8 (160,160,3), trả sáu trường quyết định của detect_unknown.

    Giữ thứ tự gốc: ảnh → threshold → inference → matching. Registry không được
    kiểm tra sớm hơn notebook. Không đổi màu, resize hoặc alignment đầu vào.
    """
    validate_aligned_face(face_rgb)
    validate_threshold(threshold)
    embedding = extract_embedding(face_rgb, net)
    return detect_unknown(embedding, registered_embeddings, threshold)


def recognize_detected_faces(
    detections: list[dict[str, object]],
    net: cv2.dnn.Net,
    registered_embeddings: dict[str, np.ndarray],
    threshold: float,
) -> list[dict[str, object]]:
    """Validate cả list trước inference; trả kết quả đúng thứ tự, metadata deepcopy.

    [] trả [] trước khi kiểm tra threshold/registry như notebook 10. Với list có
    mặt, giữ TypeError/ValueError/RuntimeError và tiền tố face_index của hàm gốc.
    Không thêm metadata bị thiếu; kết quả không chứa face_rgb.
    """
    validate_mtcnn_output(detections)
    if not detections:
        return []
    validate_threshold(threshold)
    results = []
    for index, detection in enumerate(detections):
        try:
            decision = recognize_aligned_face(
                detection["face_rgb"], net, registered_embeddings, threshold)
        except (TypeError, ValueError, RuntimeError) as error:
            raise type(error)(f"face_index {index}: {error}") from error
        result = {"face_index": index, **decision}
        for name in ("bbox", "confidence", "landmarks"):
            if name in detection:
                result[name] = deepcopy(detection[name])
        results.append(result)
    return results
