
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
