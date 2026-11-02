

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from facenet_recognition import database as db
from facenet_recognition import recognition as core
from facenet_recognition.contracts import validate_mtcnn_output
from facenet_recognition.facenet_model import extract_embedding
from facenet_recognition.matching import validate_database_embedding, validate_threshold
from .enrollment_pipeline import build_model_version
from .mtcnn_adapter import mtcnn_to_facenet_detections


def _load_registry(db_path: str | Path, model_version: str) -> dict[str, np.ndarray]:
    """Đọc representative qua list/get, không áp chính sách N == 3 của reader cũ."""
    path = db.validate_database_path(db_path)
    if not path.is_file():
        raise FileNotFoundError("Không tìm thấy SQLite registry; recognition không tự tạo.")
    registered = {}
    for metadata in db.list_people(path):
        person_id = metadata["person_id"]
        if person_id == "Unknown":
            raise ValueError("person_id 'Unknown' được dành riêng cho kết quả từ chối.")
        person = db.get_person(path, person_id, expected_model_version=model_version)
        if person is None:
            raise RuntimeError("Registry thay đổi trong khi đọc; không bỏ qua bản ghi.")
        validate_database_embedding(person["embedding"])
        registered[person_id] = person["embedding"]
    return registered


def recognize_faces_from_image(
    image_rgb: np.ndarray,
    detector,
    facenet_net,
    db_path: str | Path,
    model_version: str,
    threshold: float,
) -> list[dict[str, object]]:
    """Trả kết quả theo thứ tự detector, không chứa ảnh/embedding/ranking.

    Preflight luôn kiểm tra ảnh, threshold, profile và registry, kể cả ảnh không
    có mặt. Registry rỗng hợp lệ -> unknown với score/candidate None, nhưng vẫn
    extract/validate từng embedding. Lỗi bất kỳ face làm cả lời gọi raise.
    Registry phải ổn định trong lúc đọc; không hỗ trợ concurrent writers.
    """
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb phải là numpy.ndarray.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or min(image_rgb.shape[:2]) < 1:
        raise ValueError("image_rgb phải có shape (H,W,3), H/W > 0.")
    if image_rgb.dtype != np.uint8:
        raise TypeError("image_rgb phải có dtype uint8.")
    threshold = validate_threshold(threshold)
    if not isinstance(model_version, str):
        raise TypeError("model_version phải là chuỗi.")
    if not model_version.strip() or model_version != model_version.strip():
        raise ValueError("model_version không được rỗng hoặc có khoảng trắng đầu/cuối.")
    if model_version != build_model_version(detector):
        raise ValueError("model_version không khớp pipeline MTCNN hiện tại.")
    registered = _load_registry(db_path, model_version)

    stage = "MTCNN detection"
    try:
        detected = detector.detect(image_rgb)
        stage = "MTCNN alignment"
        faces = detector.extract_faces_rgb(image_rgb, detected, size=160, align=True)
        stage = "MTCNN adapter"
        mapped = mtcnn_to_facenet_detections(image_rgb, detected, faces)
        validate_mtcnn_output(mapped)
        if len(mapped) != len(detected.boxes):
            raise ValueError("Số output adapter không khớp detections.")
        for index, face in enumerate(mapped):
            if "bbox" not in face or "confidence" not in face:
                raise ValueError(f"face_index {index}: adapter thiếu bbox/confidence.")
    except (TypeError, ValueError, RuntimeError, AttributeError) as error:
        raise type(error)(f"{stage}: {error}") from error

    if not mapped:
        return []
    if registered:
        # Giữ validation, cosine, ranking/tie-break, >= và face_index của core.
        decisions = core.recognize_detected_faces(mapped, facenet_net, registered, threshold)
    else:
        # Core cũ từ chối registry rỗng. Nhánh mới chỉ biểu diễn không có ứng viên,
        # không viết lại matching hoặc gán similarity giả bằng 0.
        decisions = []
        for index, face in enumerate(mapped):
            try:
                embedding = extract_embedding(face["face_rgb"], facenet_net)
                validate_database_embedding(embedding)
            except (TypeError, ValueError, RuntimeError) as error:
                raise type(error)(f"face_index {index}: {error}") from error
            decisions.append({"is_known": False, "predicted_id": "Unknown",
                              "best_match_id": None, "similarity": None})
    if len(decisions) != len(mapped):
        raise RuntimeError("Core recognition phải trả một kết quả cho mỗi face.")
    results = []
    for index, (face, decision) in enumerate(zip(mapped, decisions)):
        result = {
            "face_index": index,
            "bbox": deepcopy(face["bbox"]),
            "confidence": float(face["confidence"]),
            "status": "recognized" if decision["is_known"] else "unknown",
            "person_id": decision["predicted_id"] if decision["is_known"] else None,
            "best_match_id": decision["best_match_id"],
            "similarity": decision["similarity"],
            "threshold": threshold,
            "model_version": model_version,
        }
        if "landmarks" in face:
            result["landmarks"] = deepcopy(face["landmarks"])
        results.append(result)
    return results
