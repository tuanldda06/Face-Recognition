"""Enrollment MTCNN -> FaceNet -> SQLite, không tự nạp model hoặc tạo registry.

Caller xác nhận ảnh được phép sử dụng và cùng danh tính. Chỉ dùng assets bundled
đã được build_model_version() fingerprint; caller phải nạp đúng các assets đó.
Hàm không thể suy ra danh tính hoặc kiểm chứng weights đang nằm trong cv2.Net.
Không ghi ảnh, embedding vào log; không có recognition hoặc threshold.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from FaceNet import database as db
from FaceNet.facenet_model import extract_embedding
from FaceNet.enrollment import representative_from_embeddings
from FaceNet.matching import validate_database_embedding
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
        "FaceNet/models/facenet.onnx",
        "MTCNN/weights/pnet.pt", "MTCNN/weights/rnet.pt", "MTCNN/weights/onet.pt",
        "MTCNN/mtcnn.py", "MTCNN/model_loader.py",
        "MTCNN/models/pnet.py", "MTCNN/models/rnet.py", "MTCNN/models/onet.py",
        "MTCNN/detection/bounding_box.py", "MTCNN/detection/image_pyramid.py",
        "MTCNN/preprocessing/image_processing.py", "MTCNN/preprocessing/face_alignment.py",
        "FaceNet/contracts.py", "FaceNet/facenet_model.py",
        "FaceNet/enrollment.py", "FaceNet/matching.py",
        "integration/mtcnn_adapter.py", "integration/enrollment_pipeline.py",
        "FaceNet/database.py", "FaceNet/recognition.py", "FaceNet/__init__.py",
        "integration/recognition_pipeline.py", "integration/recognition_runtime.py",
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

# SQLITE_RELOCATION_COMPATIBILITY
_MIGRATION_BASE_SHA256 = "75aa60e24dcda29e58886dedf0ac7999bc41f5cdf204a7d7e3ad84fe7bbdbbef"
# Current bytes verified against the archived algorithm before this relocation.
# These are NEW source hashes, not historical hashes assigned to changed files.
_MIGRATION_SOURCE_SHA256 = {
    "FaceNet/__init__.py": "52789f71cdbe76891c52ca48def569fb4eb7ebecedc75ed385a38ccaa6938405",
    "integration/mtcnn_adapter.py": "9a37737898f392f86a0e226180a21ab7f14b5447ef8e63d53256c4f436151c1c",
    "integration/recognition_pipeline.py": "4e1bd402427eee9c7730ca8de407d363f8a05011e73dc6e9662745e3cd6c6523",
    "integration/recognition_runtime.py": "8623ff49fdce2b9f319a1c852067f6bf138f18ce58ed9a34dd81f156d0826af4",
    "integration/__init__.py": "14c2a9ce97a9d3039f2b89e558755fb650a83999ad583d5f1593411a5bf64b29",
    "MTCNN/model_loader.py": "a4950fb34e906a39209bc8b39470cb2d170c602d40bf4859ace160ffbad2a8b8",
    "MTCNN/mtcnn.py": "25e4ea8ec74853c2c640d42953529d377449469172d0c2262c5c621a33c8cd87",
    "MTCNN/__init__.py": "30eac9735e84988967128ac67c34a7aed2c9ffb9442a9b6eee2e65e360f4d399",
    "MTCNN/detection/bounding_box.py": "bb1d63c42fec67f79b81e7a3c79ea3c7f63f523e891252ac27e24089e821a079",
    "MTCNN/detection/image_pyramid.py": "785e711dec10d4ee2332d9a633fc726d7c09d52767cad1aea15c4a69240df047",
    "MTCNN/models/onet.py": "1e2af79f85c54919ab5d06853b33f3ae1163b5fd6e067edcc24e6f2255284fc4",
    "MTCNN/models/pnet.py": "b5afc63ebce752c04fc185db1b69ac197badf6147f7c2e6a4218d07fe0880835",
    "MTCNN/models/rnet.py": "ebff53e9a15caf16e469e6bfa9a51de90e120a18637271e8fe9094107a0c45d5",
    "MTCNN/preprocessing/face_alignment.py": "e1f3d740b1270e49d039a65fb7dd2195e9d521dd6c230dcb54dddda27a3def54",
    "MTCNN/preprocessing/image_processing.py": "8072f6047aa68a12650c05b27ee4eb67df880701e486108763f692bade023541",
    "FaceNet/contracts.py": "76180ff655f813d056986d734b06da3e7e1b24c4bec5ad50a676dc55370da0c6",
    "FaceNet/database.py": "815833ab427cd267ab93f7efbaaf20a8f364653c7436b07e6fd1fd5952081679",
    "FaceNet/enrollment.py": "c5d0f17b30ffd253565f81ef6e7532af8ab56a8895265f6b660b9137f7115801",
    "FaceNet/facenet_model.py": "714532b51b3701d3b82e25b6286d6bfe6a4032638ce10aa99331d06db9a9620e",
    "FaceNet/matching.py": "ef5579f23ca4675a8a5cd6d502ce9de54400b08f5b3e349324103eca1afba84c",
    "FaceNet/recognition.py": "f5fb12b44919032766a081b16f3c8a1a8c1e7d59f7652e296d2f960224474ecc"
}
_LEGACY_PROFILE_SHA256 = "9a42c0f3c8d4261e54304123746441254e8fea225abdc2a1e8c6084dbb3d9f1f"
_LEGACY_DESCRIPTOR_SHA256 = "26f1432e29721a4d1e881164a62db603a968a5aa3c0e505afcba29c59668d99b"


def _migration_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _migration_versions(detector, model_version: str) -> tuple[str, str]:
    """Allow only the reviewed relocation, identical weights/config/runtime."""
    root = Path(__file__).resolve().parents[1]
    own_source = Path(__file__).read_bytes().split(b"\n# SQLITE_RELOCATION_COMPATIBILITY\n", 1)[0]
    if hashlib.sha256(own_source).hexdigest() != _MIGRATION_BASE_SHA256:
        raise ValueError("Enrollment source differs from the verified relocation.")
    profile_path = root / "integration/reports/evaluation_pipeline_profile.json"
    if _migration_sha256(profile_path) != _LEGACY_PROFILE_SHA256:
        raise ValueError("Frozen legacy profile checksum mismatch.")
    legacy = json.loads(profile_path.read_text(encoding="utf-8"))["model_version"]
    if hashlib.sha256(legacy.encode()).hexdigest() != _LEGACY_DESCRIPTOR_SHA256:
        raise ValueError("Untrusted legacy descriptor.")
    current = build_model_version(detector)
    if not isinstance(model_version, str) or model_version != current:
        raise ValueError("Migration requires the exact current model_version.")
    for relative, expected in _MIGRATION_SOURCE_SHA256.items():
        if _migration_sha256(root / relative) != expected:
            raise ValueError("Source differs from the verified relocation: " + relative)
    old = json.loads(legacy)
    new = json.loads(current)
    old_assets = old.pop("assets_and_source_sha256")
    new_assets = new.pop("assets_and_source_sha256")
    if old != new:
        raise ValueError("Runtime/detector/embedding profile differs; re-enrollment required.")
    for old_name in ("models/facenet.onnx", "MTCNN/weights/pnet.pt",
                     "MTCNN/weights/rnet.pt", "MTCNN/weights/onet.pt"):
        new_name = "FaceNet/" + old_name if old_name.startswith("models/") else old_name
        if new_assets.get(new_name) != old_assets[old_name]:
            raise ValueError("Weights differ; migration refused.")
    return legacy, current


def migrate_legacy_registry(
    source_path: str | Path,
    destination_path: str | Path,
    detector,
    *,
    model_version: str,
) -> Path:
    """Copy a frozen-profile registry to a NEW file with current metadata.

    Explicit opt-in only. Embedding bytes, IDs, counts, checksums and schema
    remain identical. No inference, threshold changes, upsert or in-place update.
    Source must be closed/stable without SQLite sidecars. Unknown versions,
    mixed versions, changed assets/runtime/source or an existing target fail.
    """
    from contextlib import closing
    import os
    import sqlite3

    legacy, current = _migration_versions(detector, model_version)
    source = db.validate_database_path(source_path)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    destination = _registry_path(destination_path)
    if source == destination:
        raise ValueError("Migration source and destination must differ.")
    if destination.exists():
        raise FileExistsError(str(destination))
    if not destination.parent.is_dir():
        raise FileNotFoundError(str(destination.parent))
    if any(Path(str(source) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("Close the source registry and its SQLite sidecars first.")
    source_sha = _migration_sha256(source)
    with closing(db.connect_database(source, readonly=True)) as connection:
        connection.execute("BEGIN")
        rows = connection.execute(
            "SELECT person_id,embedding,num_images,model_version,embedding_sha256 "
            "FROM FaceDatabase ORDER BY person_id"
        ).fetchall()
        if not rows:
            raise ValueError("Empty registry has no legacy version to verify.")
        for row in rows:
            db.row_to_person(row, expected_model_version=legacy)
        connection.rollback()
    if _migration_sha256(source) != source_sha:
        raise RuntimeError("Source registry changed during validation.")
    # Reserve exclusively; this function owns only this newly created file.
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    file_id = os.fstat(descriptor).st_ino
    os.close(descriptor)
    try:
        with closing(sqlite3.connect(destination.as_uri() + "?mode=rw", uri=True)) as connection:
            with connection:
                connection.execute(db.CREATE_TABLE_SQL)
                connection.executemany(
                    "INSERT INTO FaceDatabase VALUES (?,?,?,?,?)",
                    [(person, blob, count, current, checksum)
                     for person, blob, count, _, checksum in rows],
                )
                db.check_schema(connection)
                actual = connection.execute(
                    "SELECT person_id,embedding,num_images,model_version,embedding_sha256 "
                    "FROM FaceDatabase ORDER BY person_id"
                ).fetchall()
                expected = [(p, b, n, current, h) for p, b, n, _, h in rows]
                if actual != expected:
                    raise RuntimeError("Migration changed data beyond model_version.")
                if (_migration_sha256(source) != source_sha
                        or _migration_versions(detector, model_version) != (legacy, current)):
                    raise RuntimeError("Source or runtime changed during migration.")
        return destination
    except BaseException:
        if destination.exists():
            stat = destination.stat()
            if (destination.resolve() != destination or stat.st_nlink != 1
                    or stat.st_ino != file_id):
                raise RuntimeError("Migration target ownership changed; no cleanup attempted.")
            destination.unlink()
        raise
