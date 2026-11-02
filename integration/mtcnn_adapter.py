
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from facenet_recognition.contracts import validate_mtcnn_output

if TYPE_CHECKING:
    from MTCNN.mtcnn import DetectionResult


LANDMARK_ORDER = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")


def _metadata_array(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Nhận mảng số thực; float32 là dtype MTCNN, cũng giữ được int/float khác."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} phải là numpy.ndarray.")
    if value.shape != shape:
        raise ValueError(f"{name} phải có shape {shape}, nhận được {value.shape}.")
    if value.dtype.kind not in "iuf":
        raise TypeError(f"{name} phải chứa số thực, không phải bool/object/complex.")
    return value


def mtcnn_to_facenet_detections(
    image_rgb: np.ndarray,
    detection_result: DetectionResult,
    faces_rgb: np.ndarray,
) -> list[dict[str, object]]:
    """Validate và ánh xạ boxes[i]/scores[i]/landmarks[i]/faces_rgb[i] theo 1:1.

    DetectionResult được kiểm tra theo cấu trúc ba thuộc tính NumPy, không import
    Torch để kiểm tra kiểu class. Sai kiểu/dtype: TypeError; shape, số lượng,
    finite, score hoặc bbox sai: ValueError. Lỗi từng hàng chứa face_index.
    Không tự detect/align/resize/normalize, lọc face hoặc thực hiện I/O.
    """
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb phải là numpy.ndarray.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or min(image_rgb.shape[:2]) < 1:
        raise ValueError("image_rgb phải có shape (H,W,3), H/W > 0.")
    if image_rgb.dtype != np.uint8:
        raise TypeError("image_rgb phải có dtype uint8.")

    try:
        boxes = detection_result.boxes
        scores = detection_result.scores
        landmarks = detection_result.landmarks
    except AttributeError as error:
        raise TypeError("detection_result phải có boxes, scores và landmarks.") from error

    if not isinstance(boxes, np.ndarray):
        raise TypeError("boxes phải là numpy.ndarray.")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes phải có shape (N,4).")
    count = boxes.shape[0]
    boxes = _metadata_array(boxes, "boxes", (count, 4))
    scores = _metadata_array(scores, "scores", (count,))
    landmarks = _metadata_array(landmarks, "landmarks", (count, 5, 2))
    if not isinstance(faces_rgb, np.ndarray):
        raise TypeError("faces_rgb phải là numpy.ndarray.")
    if faces_rgb.shape != (count, 160, 160, 3):
        raise ValueError("faces_rgb phải có shape (N,160,160,3), N khớp detections.")
    if faces_rgb.dtype != np.uint8:
        raise TypeError("faces_rgb phải có dtype uint8.")

    height, width = image_rgb.shape[:2]
    metadata = []
    for index in range(count):
        box, score, points = boxes[index], scores[index], landmarks[index]
        if not (np.isfinite(box).all() and np.isfinite(score) and np.isfinite(points).all()):
            raise ValueError(f"face_index {index}: metadata chứa NaN hoặc Inf.")
        if np.any(box[2:] <= box[:2]):
            raise ValueError(f"face_index {index}: bbox gốc cần x2>x1 và y2>y1.")
        if not 0 <= score <= 1:
            raise ValueError(f"face_index {index}: confidence phải nằm trong [0,1].")
        clipped = np.clip(box.copy(), [0, 0, 0, 0], [width, height, width, height])
        if np.any(clipped[2:] <= clipped[:2]):
            raise ValueError(f"face_index {index}: bbox sau clip rỗng hoặc không hợp lệ.")
        item = {"bbox": clipped.tolist(), "confidence": float(score)}
        # Âm nhưng hữu hạn/đúng shape: chỉ lược metadata, vẫn giữ ảnh và vị trí.
        if not np.any(points < 0):
            item["landmarks"] = {name: point.tolist()
                                 for name, point in zip(LANDMARK_ORDER, points)}
        metadata.append(item)

    results = [{"face_rgb": faces_rgb[index].copy(), **item}
               for index, item in enumerate(metadata)]
    validate_mtcnn_output(results)
    return results
