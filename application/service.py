"""One application service shared by API/UI; existing core owns all inference/math."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import tempfile
from threading import RLock
from time import perf_counter

from filelock import FileLock, Timeout
import numpy as np

from FaceNet import (build_model_version, enroll_person_from_images,
                     load_facenet_model, recognize_image)
from FaceNet import database as db
from FaceNet.matching import validate_person_id

from .assets import verify_model
from .config import MODEL_PATH, ROOT, Settings
from .errors import ApplicationError
from .images import validate_rgb

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    detector: object
    net: object
    version: str


def load_runtime(settings: Settings):
    verify_model()
    import cv2
    import torch
    from MTCNN.mtcnn import MTCNN
    torch.set_num_threads(settings.cpu_threads)
    cv2.setNumThreads(settings.cpu_threads)
    detector = MTCNN(device="cpu")
    net = load_facenet_model(MODEL_PATH)
    return Runtime(detector, net, build_model_version(detector))


class BoundedDetector:
    """Delegate unchanged detection/alignment; reject count limits before embedding."""

    def __init__(self, detector, maximum, enrollment=False):
        self.detector, self.maximum, self.enrollment = detector, maximum, enrollment

    def __getattr__(self, name):
        return getattr(self.detector, name)

    def detect(self, image):
        result = self.detector.detect(image)
        from MTCNN.mtcnn import validate_detection_result
        validate_detection_result(result)
        count = len(result.boxes)
        if self.enrollment and count != 1:
            raise ApplicationError("enrollment_face_count", f"Mỗi ảnh đăng ký cần đúng 1 khuôn mặt; phát hiện {count}.")
        if count > self.maximum:
            raise ApplicationError("too_many_faces", f"Ảnh có {count} khuôn mặt, giới hạn {self.maximum}.")
        return result


FACE_FIELDS = ("face_index", "bbox", "confidence", "person_id", "status", "similarity", "threshold")


def public_faces(results):
    # Explicit allowlist: no embeddings, candidate identity, crops or rankings.
    return [{key: row[key] for key in FACE_FIELDS} for row in results]


class ApplicationService:
    def __init__(self, settings=None, runtime_loader=load_runtime):
        self.settings = settings or Settings.from_env()
        self._loader = runtime_loader
        self._mutex = RLock()
        self.runtime = None
        self.model_error = None
        self._started = False
        from .webcam_enrollment import WebcamEnrollment
        self.webcam_enrollment = WebcamEnrollment(self)
        path = db.validate_database_path(self.settings.registry, for_write=True)
        if not (path.is_relative_to(ROOT) or path.is_relative_to(Path(tempfile.gettempdir()).resolve())):
            raise ValueError("Registry must be within the project or temporary directory.")
        self.path = path
        # Shared across application processes, without creating the registry directory.
        lock_key = hashlib.sha256(os.path.normcase(str(path)).encode()).hexdigest()
        self._file_lock = FileLock(Path(tempfile.gettempdir()) / f"face_application_{lock_key}.lock",
                                   timeout=self.settings.lock_timeout)

    def start(self):
        with self._mutex:
            if self._started:
                return
            self._started = True
            try:
                self.runtime = self._loader(self.settings)
            except ApplicationError as error:
                self.model_error = error
            except Exception as error:
                logger.error("Model startup failed (%s)", type(error).__name__)
                self.model_error = ApplicationError("model_load_failed", "Không nạp được MTCNN/FaceNet. Kiểm tra assets và runtime.", 503)

    @contextmanager
    def guard(self):
        with self._mutex:
            try:
                with self._file_lock:
                    yield
            except Timeout as error:
                raise ApplicationError("service_busy", "Registry đang bận; thử lại sau.", 503) from error

    def _require_model(self):
        if self.runtime is None:
            raise self.model_error or ApplicationError("not_started", "Model chưa được khởi động.", 503)

    def _registry(self):
        if not self.path.is_file():
            raise ApplicationError("registry_missing", "Registry chưa tạo. Chạy python -m application init-registry.", 503)
        try:
            db.validate_database_path(self.path, for_write=True)
            if self.path.stat().st_nlink > 1:
                raise ValueError("Hardlinked registry")
            people = db.list_people(self.path)
            for person in people:
                if person["person_id"] == "Unknown":
                    raise ValueError("Reserved identity")
                version = self.runtime.version if self.runtime else person["model_version"]
                if db.get_person(self.path, person["person_id"], version) is None:
                    raise ValueError("Registry changed during read")
            return people
        except Exception as error:
            logger.error("Registry validation failed (%s)", type(error).__name__)
            raise ApplicationError("registry_invalid", "Registry lỗi hoặc model_version không tương thích; không tự migrate.", 503) from error

    def health(self):
        with self.guard():
            problems = []
            if self.runtime is None:
                problems.append((self.model_error or ApplicationError("not_started", "Chưa khởi động.", 503)).payload()["error"])
            count = None
            try:
                count = len(self._registry())
                registry_state = "empty" if count == 0 else "ready"
            except ApplicationError as error:
                registry_state = "missing" if error.code == "registry_missing" else "invalid"
                problems.append(error.payload()["error"])
            return {"status": "ok" if not problems else "degraded", "backend": "running",
                    "models": "ready" if self.runtime else "unavailable",
                    "registry": registry_state, "person_count": count,
                    "registry_version_checked": self.runtime is not None and count is not None,
                    "threshold_profile": "v2", "issues": problems}

    @staticmethod
    def check_id(person_id):
        try:
            validate_person_id(person_id)
        except (ValueError, TypeError) as error:
            raise ApplicationError("invalid_person_id", "person_id không được rỗng hoặc có khoảng trắng hai đầu.") from error
        if (person_id == "Unknown" or len(person_id) > 100
                or any(ord(c) < 32 or c in "/\\" for c in person_id)):
            raise ApplicationError("invalid_person_id", "ID tối đa 100 ký tự, không dùng Unknown, dấu /, \\ hoặc ký tự điều khiển.")

    @contextmanager
    def _operation(self):
        try:
            yield
        except ApplicationError:
            raise
        except Exception as error:
            logger.error("Pipeline operation failed (%s)", type(error).__name__)
            raise ApplicationError("pipeline_failed", "Detection/alignment/inference hoặc SQLite thất bại; không có kết quả nhận diện hợp lệ.", 500) from error

    def enroll(self, person_id, images):
        self.check_id(person_id)
        if not isinstance(images, (list, tuple)) or not 1 <= len(images) <= self.settings.max_images:
            raise ApplicationError("image_count", f"Cần từ 1 đến {self.settings.max_images} ảnh.")
        for image in images:
            validate_rgb(image, self.settings)
        with self.guard(), self._operation():
            self._require_model()
            people = self._registry()
            if any(p["person_id"] == person_id for p in people):
                raise ApplicationError("duplicate_person", "person_id đã tồn tại; không ghi đè.", 409)
            rt = self.runtime
            detector = BoundedDetector(rt.detector, self.settings.max_faces, enrollment=True)
            result = enroll_person_from_images(person_id, images, detector, rt.net, self.path, rt.version)
            return {key: result[key] for key in ("person_id", "num_images", "status")}

    def recognize(self, image):
        started = perf_counter()
        validate_rgb(image, self.settings)
        with self.guard(), self._operation():
            self._require_model()
            self._registry()
            rt = self.runtime
            rows = recognize_image(image, BoundedDetector(rt.detector, self.settings.max_faces),
                                   rt.net, self.path, rt.version)
            faces = public_faces(rows)
        elapsed = perf_counter() - started
        return {"faces": faces, "face_count": len(faces),
                "image": {"width": image.shape[1], "height": image.shape[0]},
                "latency_ms": elapsed * 1000, "processing_fps": 1 / elapsed}

    def persons(self):
        from .webcam_registry import read_names
        with self.guard(), self._operation():
            people = self._registry()
            names = read_names(self.path)
            return [{"person_id": p["person_id"], "num_images": p["num_images"],
                     **({"display_name": names[p["person_id"]]} if p["person_id"] in names else {})}
                    for p in people]

    def delete(self, person_id):
        self.check_id(person_id)
        with self.guard(), self._operation():
            self._registry()
            try:
                from .webcam_registry import delete_identity
                delete_identity(self.path, person_id)
            except KeyError as error:
                raise ApplicationError("person_not_found", "Không tìm thấy person_id.", 404) from error
            return {"person_id": person_id, "status": "deleted"}
