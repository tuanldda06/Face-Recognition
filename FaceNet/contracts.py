"""Hợp đồng giữ nguyên từ notebook 01 và 10, không tự sửa ảnh/metadata.

Validator chỉ kiểm tra cấu trúc/số học. Bên gọi chịu trách nhiệm về màu RGB,
alignment và biên ảnh gốc; module không đổi màu, resize, clip hoặc clamp.
"""

from __future__ import annotations

import numpy as np


LANDMARK_NAMES = {"left_eye", "right_eye", "nose", "mouth_left", "mouth_right"}
DETECTION_KEYS = {"face_rgb", "bbox", "confidence", "landmarks"}
NUMERIC_TYPES = (int, float, np.integer, np.floating)


def validate_face(face_rgb: np.ndarray) -> None:
    """Kiểm tra RGB uint8 (160,160,3); lỗi kiểu dùng TypeError, hình dạng dùng ValueError."""
    if not isinstance(face_rgb, np.ndarray):
        raise TypeError("Ảnh phải là numpy.ndarray.")
    if face_rgb.size == 0:
        raise ValueError("Ảnh đầu vào rỗng.")
    if face_rgb.shape != (160, 160, 3):
        raise ValueError(
            f"Ảnh phải có shape (160, 160, 3), nhận được {face_rgb.shape}."
        )
    if face_rgb.dtype != np.uint8:
        raise TypeError(
            f"Ảnh phải có dtype uint8, nhận được {face_rgb.dtype}."
        )
    # uint8 đã giới hạn giá trị; giữ kiểm tra tường minh của notebook gốc.
    if face_rgb.min() < 0 or face_rgb.max() > 255:
        raise ValueError("Giá trị pixel phải nằm trong 0–255.")


def validate_aligned_face(face_rgb: np.ndarray) -> None:
    """Giữ wrapper notebook 10: None là ValueError, khác validate_face(None)."""
    if face_rgb is None:
        raise ValueError("face_rgb không được là None.")
    validate_face(face_rgb)


def validate_coordinate_sequence(values: object, size: int, label: str) -> None:
    """Kiểm tra list/tuple gồm size số hữu hạn, không âm; không nhận bool."""
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ValueError(f"{label} phải là list/tuple gồm {size} tọa độ.")
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, NUMERIC_TYPES):
            raise TypeError(f"{label} phải chứa số thực, không phải bool.")
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{label} phải hữu hạn và không âm.")


def validate_detection_metadata(detection: dict[str, object]) -> None:
    """Kiểm tra metadata được cung cấp; trường không có thì bỏ qua, không tự sửa."""
    if "bbox" in detection:
        bbox = detection["bbox"]
        validate_coordinate_sequence(bbox, 4, "bbox")
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox cần x2>x1 và y2>y1, không nhận vùng rỗng.")
    if "confidence" in detection:
        confidence = detection["confidence"]
        if isinstance(confidence, (bool, np.bool_)) or not isinstance(confidence, NUMERIC_TYPES):
            raise TypeError("confidence phải là số thực, không phải bool.")
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence phải hữu hạn trong [0,1].")
    if "landmarks" in detection:
        landmarks = detection["landmarks"]
        if not isinstance(landmarks, dict) or set(landmarks) != LANDMARK_NAMES:
            raise ValueError("landmarks phải là dict đúng 5 tên điểm đã đề xuất.")
        for name, point in landmarks.items():
            validate_coordinate_sequence(point, 2, f"landmarks.{name}")


def validate_mtcnn_output(detections: list[dict[str, object]]) -> None:
    """Kiểm tra toàn bộ list theo notebook 10; lỗi có face_index, [] hợp lệ.

    Giữ tên API cũ để tương thích. Dữ liệu có thể đến từ detector bất kỳ đáp ứng
    hợp đồng; không import hoặc gọi MTCNN trong module này.
    """
    if not isinstance(detections, list):
        raise TypeError("detections phải là list các dictionary.")
    for index, detection in enumerate(detections):
        try:
            if not isinstance(detection, dict):
                raise TypeError("Mỗi detection phải là dictionary.")
            if "face_rgb" not in detection:
                raise ValueError("Detection thiếu face_rgb.")
            if set(detection) - DETECTION_KEYS:
                raise ValueError("Detection có khóa ngoài hợp đồng đề xuất.")
            validate_aligned_face(detection["face_rgb"])
            validate_detection_metadata(detection)
        except (TypeError, ValueError) as error:
            raise type(error)(f"face_index {index}: {error}") from error
