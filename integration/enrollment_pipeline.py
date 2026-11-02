
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from facenet_recognition import database as db
from facenet_recognition.facenet_model import extract_embedding
from facenet_recognition.enrollment import representative_from_embeddings
from facenet_recognition.matching import validate_database_embedding
from .mtcnn_adapter import mtcnn_to_facenet_detections


def _detector_config(detector) -> dict:
    """Ghi đúng tham số đang dùng; không tự sửa cấu hình detector."""
    try:
        minimum = detector.min_face_size
        factor = detector.factor
        thresholds = detector.thresholds
        batch = detector.batch_size
        device = str(detector.device)
    except AttributeError as error:
        raise TypeError("detector thiếu cấu hình MTCNN.") from error
    numeric = (int, float, np.integer, np.floating)
    values = [minimum, factor]
    if not isinstance(thresholds, (list, tuple)) or len(thresholds) != 3:
        raise ValueError("MTCNN thresholds phải có 3 giá trị.")
    values.extend(thresholds)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numeric)
           or not np.isfinite(v) for v in values):
        raise ValueError("Cấu hình MTCNN phải là số hữu hạn.")
    if (minimum <= 0 or not 0 < factor < 1
            or any(not 0 <= value <= 1 for value in thresholds)):
        raise ValueError("Cấu hình MTCNN ngoài khoảng hợp lệ.")
    if isinstance(batch, (bool, np.bool_)) or not isinstance(batch, (int, np.integer)) or batch < 1:
        raise ValueError("MTCNN batch_size phải nguyên dương.")
    if device != "cpu":
        raise ValueError("Phase này chỉ hỗ trợ MTCNN CPU.")
    return {"min_face_size": float(minimum), "factor": float(factor),
            "thresholds": [float(v) for v in thresholds],
            "batch_size": int(batch), "device": device}


def build_model_version(detector) -> str:
    """Tạo JSON canonical cho assets bundled và cấu hình thực tế, chỉ khi gọi.

    Không nạp model. Hash ONNX, P/R/O weights và source ảnh hưởng embedding;
    mô tả alignment ArcFace112 scale160, preprocessing, aggregation và runtime.
    Caller phải dùng model loader/weights tương ứng, không dùng net tùy biến.
    """
    root = Path(__file__).resolve().parents[1]
    configuration = _detector_config(detector)
    files = [
        "models/facenet.onnx",
        "MTCNN/weights/pnet.pt", "MTCNN/weights/rnet.pt", "MTCNN/weights/onet.pt",
        "MTCNN/mtcnn.py", "MTCNN/model_loader.py",
        "MTCNN/models/pnet.py", "MTCNN/models/rnet.py", "MTCNN/models/onet.py",
        "MTCNN/detection/bounding_box.py", "MTCNN/detection/image_pyramid.py",
        "MTCNN/preprocessing/image_processing.py", "MTCNN/preprocessing/face_alignment.py",
        "src/facenet_recognition/contracts.py", "src/facenet_recognition/facenet_model.py",
        "src/facenet_recognition/enrollment.py", "src/facenet_recognition/matching.py",
        "integration/mtcnn_adapter.py", "integration/enrollment_pipeline.py",
    ]
    hashes = {}
    for name in files:
        digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    profile = {
        "schema": "facenet-mtcnn-enrollment/1",
        "input": "RGB_uint8_HWC",
        "aligned_face": [160, 160, 3],
        "alignment": "MTCNN_5pt_similarity_ArcFace112_scaled160_alignTrue",
        "facenet_preprocessing": "(pixel-127.5)/128.0;NCHW;RGB",
        "embedding": "512_float32_L2_float64_norm",
        "aggregation": "mean_float64_then_L2_to_float32",
        "execution": "MTCNN_CPU;OpenCV_DNN_BACKEND_OPENCV_TARGET_CPU",
        "detector": configuration,
        "assets_and_source_sha256": hashes,
        "runtime": {name: importlib.metadata.version(name) for name in
                    ("numpy", "opencv-python", "Pillow", "torch")},
    }
    return json.dumps(profile, sort_keys=True, separators=(",", ":"))


def _registry_path(db_path: str | Path) -> Path:
    """Chỉ ghi registry trong project tích hợp hoặc vùng tạm; cấm source/alias."""
    path = db.validate_database_path(db_path, for_write=True)
    root = Path(__file__).resolve().parents[1]
    sources = (Path("D:/MTCNN").resolve(), Path("D:/Facenet-Recognition").resolve())
    if any(path.is_relative_to(source) for source in sources):
        raise PermissionError("Không được ghi registry trong project nguồn.")
    if not (path.is_relative_to(root) or path.is_relative_to(Path(tempfile.gettempdir()).resolve())):
        raise PermissionError("Registry phải nằm trong project tích hợp hoặc thư mục tạm.")
    # Hardlink có thể trỏ tới baseline/source dù path nằm trong project mới.
    if path.exists() and path.stat().st_nlink > 1:
        raise PermissionError("Không ghi registry qua hardlink.")
    return path


def initialize_enrollment_registry(db_path: str | Path) -> Path:
    """Yêu cầu tạo tường minh; file đã tồn tại luôn bị từ chối, không ghi đè."""
    path = _registry_path(db_path)
    if path.exists():
        raise FileExistsError("Registry đã tồn tại; không khởi tạo hoặc ghi đè.")
    db.initialize_database(path)
    return path


def _check_registry(path: Path, person_id: str, model_version: str) -> None:
    # list/get kiểm tra schema, metadata, BLOB/checksum và version; không dùng
    # load_registered_embeddings vì reader nghiên cứu đó còn yêu cầu N == 3.
    people = db.list_people(path)
    duplicate = False
    for person in people:
        db.get_person(path, person["person_id"], expected_model_version=model_version)
        duplicate = duplicate or person["person_id"] == person_id
    if duplicate:
        raise ValueError("person_id đã tồn tại; không ghi đè enrollment.")


def enroll_person_from_images(
    person_id: str,
    images_rgb: list[np.ndarray] | tuple[np.ndarray, ...],
    detector,
    facenet_net,
    db_path: str | Path,
    model_version: str,
) -> dict[str, object]:
    """N > 0 ảnh cùng người, đúng 1 face/ảnh; INSERT sau khi tất cả thành công.

    Registry phải có sẵn, được caller khởi tạo tường minh. Không tự initialize,
    update, migrate, chọn mặt hoặc bỏ ảnh lỗi. Caller giữ quyền sở hữu input.
    Không hỗ trợ concurrent writers đổi model_version giữa kiểm tra và INSERT.
    """
    if not isinstance(images_rgb, (list, tuple)):
        raise TypeError("images_rgb phải là list/tuple ảnh RGB.")
    if not images_rgb:
        raise ValueError("Cần ít nhất một ảnh enrollment.")
    db.validate_record_metadata(person_id, len(images_rgb), model_version)
    # Validate cả danh sách trước khi bất kỳ ảnh nào được detection/inference.
    for index, image in enumerate(images_rgb):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image_index {index}: ảnh phải là numpy.ndarray.")
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError(f"image_index {index}: cần RGB shape (H,W,3), H/W > 0.")
        if image.dtype != np.uint8:
            raise TypeError(f"image_index {index}: ảnh phải là uint8.")

    # Chỉ nhận profile MTCNN kiểm chứng tại thời điểm gọi, từ chối IMM/ASF/v1.
    try:
        profile = json.loads(model_version)
    except (ValueError, TypeError) as error:
        raise ValueError("model_version phải do build_model_version(detector) tạo.") from error
    if (not isinstance(profile, dict) or profile.get("schema") != "facenet-mtcnn-enrollment/1"
            or model_version != build_model_version(detector)):
        raise ValueError("model_version không khớp pipeline MTCNN hiện tại.")
    path = _registry_path(db_path)
    if not path.is_file():
        raise FileNotFoundError("Hãy khởi tạo registry tường minh trước enrollment.")
    _check_registry(path, person_id, model_version)

    embeddings = []
    for index, image in enumerate(images_rgb):
        try:
            result = detector.detect(image)
            if not isinstance(result.boxes, np.ndarray) or result.boxes.ndim != 2 or result.boxes.shape[1] != 4:
                raise ValueError("DetectionResult.boxes phải có shape (N,4).")
            count = len(result.boxes)
            if count != 1:
                raise ValueError(f"Mỗi ảnh phải có đúng 1 face, phát hiện {count}.")
            faces = detector.extract_faces_rgb(image, result, size=160, align=True)
            mapped = mtcnn_to_facenet_detections(image, result, faces)
            if len(mapped) != 1:
                raise ValueError("Adapter phải trả đúng một face.")
            embedding = extract_embedding(mapped[0]["face_rgb"], facenet_net)
            validate_database_embedding(embedding)
            embeddings.append(embedding)
        except (TypeError, ValueError, RuntimeError, AttributeError) as error:
            raise type(error)(f"image_index {index}: {error}") from error
    representative = representative_from_embeddings(np.stack(embeddings))
    validate_database_embedding(representative)
    # Recheck ngay trước ghi; add_person tự transaction/rollback, không upsert.
    _check_registry(path, person_id, model_version)
    db.add_person(path, person_id, representative, len(images_rgb), model_version)
    return {"person_id": person_id, "num_images": len(images_rgb),
            "model_version": model_version, "embedding": representative.copy(),
            "status": "enrolled"}
