from dataclasses import dataclass, field, asdict
import hashlib
from pathlib import Path
import secrets
import tempfile
from time import monotonic

import cv2
import numpy as np

from FaceNet import initialize_enrollment_registry, enroll_person_from_images
from integration.webcam_identity import validate_embedding_consistency
from MTCNN.mtcnn import validate_detection_result
from .errors import ApplicationError
from .head_pose import estimate_head_pose, classify_pose
from .images import validate_rgb
from .webcam_registry import commit_identity


POSES = ("front", "left", "right", "up")
POSE_COUNT = len(POSES)
LABELS = dict(zip(POSES, ("Nhìn thẳng", "Quay sang trái của bạn", "Quay sang phải của bạn", "Ngẩng đầu")))


@dataclass(frozen=True)
class EnrollmentPolicy:
    stable_frames: int = 4
    hold_seconds: float = 0.8
    max_gap: float = 2.0
    max_angle_spread: float = 6.0
    min_face_pixels: int = 80
    min_brightness: float = 45
    max_brightness: float = 220
    min_sharpness: float = 15
    min_similarity: float = 0.65
    ttl_seconds: float = 1200
    max_sessions: int = 8


@dataclass
class EnrollmentSession:
    person_id: str
    display_name: str
    created: float
    version: str
    captures: dict = field(default_factory=dict)
    stable: list = field(default_factory=list)
    stable_since: float | None = None
    last_frame: float | None = None
    last_digest: str | None = None
    revision: int = 0
    reviewed_revision: int | None = None
    paused: bool = False
    message: str = "Bật camera và giữ tư thế nhìn thẳng."


class RecordingNet:
    """Observe the exact pipeline forwards; leave all core math untouched."""
    def __init__(self, net):
        self.net, self.outputs = net, []

    def setInput(self, *args):
        return self.net.setInput(*args)

    def forward(self, *args):
        result = self.net.forward(*args)
        self.outputs.append(result.copy() if isinstance(result, np.ndarray) else result)
        return result


class WebcamEnrollment:
    def __init__(self, service, policy=None, clock=monotonic):
        self.service = service
        self.policy = policy or EnrollmentPolicy()
        self.clock = clock
        self.sessions = {}

    def _expire(self):
        now = self.clock()
        for token in list(self.sessions):
            if now - self.sessions[token].created > self.policy.ttl_seconds:
                del self.sessions[token]

    def _get(self, token):
        self._expire()
        if not isinstance(token, str) or token not in self.sessions:
            raise ApplicationError("session_missing", "Phiên không tồn tại hoặc đã hết hạn; hãy bắt đầu lại.", 404)
        return self.sessions[token]

    @staticmethod
    def _reset(session):
        session.stable.clear()
        session.stable_since = None

    @staticmethod
    def _target(session):
        return next((pose for pose in POSES if pose not in session.captures), None)

    def start(self, person_id, display_name):
        self.service.check_id(person_id)
        if (not isinstance(display_name, str) or not display_name.strip()
                or len(display_name.strip()) > 150 or any(ord(c) < 32 for c in display_name)):
            raise ApplicationError("invalid_name", "Nhập tên từ 1–150 ký tự, không có ký tự điều khiển.")
        with self.service.guard(), self.service._operation():
            self.service._require_model()
            people = self.service._registry()
            if any(p["person_id"] == person_id for p in people):
                raise ApplicationError("duplicate_person", "person_id đã tồn tại; không ghi đè.", 409)
            self._expire()
            if len(self.sessions) >= self.policy.max_sessions:
                raise ApplicationError("sessions_full", "Đã đạt giới hạn phiên camera; hủy phiên cũ hoặc chờ hết hạn.", 503)
            token = secrets.token_urlsafe(32)
            self.sessions[token] = EnrollmentSession(person_id, display_name.strip(), self.clock(), self.service.runtime.version)
            return self._snapshot(token, self.sessions[token])

    def _snapshot(self, token, session):
        target = self._target(session)
        return {"session_id": token, "person_id": session.person_id, "display_name": session.display_name,
                "revision": session.revision, "progress": len(session.captures), "total": POSE_COUNT,
                "captured_poses": [p for p in POSES if p in session.captures],
                "next_pose": target, "instruction": LABELS.get(target, f"Xem lại {POSE_COUNT} ảnh rồi xác nhận lưu."),
                "status": "review" if target is None else ("paused" if session.paused else "capturing"),
                "stable_frames": len(session.stable), "required_stable_frames": self.policy.stable_frames,
                "message": session.message, "expires_in_seconds": max(0, self.policy.ttl_seconds - (self.clock() - session.created))}

    def status(self, token):
        with self.service._mutex:
            session = self._get(token)
            if (self._target(session) and session.last_frame is not None
                    and self.clock() - session.last_frame > self.policy.max_gap):
                self._reset(session)
                session.message = "Không nhận được frame camera. Kiểm tra kết nối và bật lại ghi hình."
            return self._snapshot(token, session)

    def review(self, token):
        return self.review_snapshot(token)["images"]

    def review_snapshot(self, token):
        with self.service._mutex:
            session = self._get(token)
            if set(session.captures) == set(POSES):
                session.reviewed_revision = session.revision
            return {"revision": session.revision,
                    "images": [(pose, session.captures[pose].copy()) for pose in POSES if pose in session.captures]}

    def cancel(self, token):
        with self.service._mutex:
            removed = self.sessions.pop(token, None)
            return {"status": "cancelled" if removed is not None else "session_ended", "progress": 0, "total": POSE_COUNT}

    def pause(self, token):
        with self.service._mutex:
            session = self._get(token)
            session.paused = True
            session.revision += 1
            self._reset(session)
            session.message = "Camera đã dừng/ngắt. Bật lại camera rồi bấm Tiếp tục."
            return self._snapshot(token, session)

    def resume(self, token):
        with self.service._mutex:
            session = self._get(token)
            session.paused = False
            session.revision += 1
            self._reset(session)
            session.last_frame = None
            session.last_digest = None
            session.message = "Đã tiếp tục; giữ tư thế theo hướng dẫn."
            return self._snapshot(token, session)

    def retake(self, token, pose):
        with self.service._mutex:
            session = self._get(token)
            if pose not in POSES or pose not in session.captures:
                raise ApplicationError("invalid_retake", "Chọn tư thế đã chụp để chụp lại.")
            del session.captures[pose]
            session.revision += 1
            session.reviewed_revision = None
            self._reset(session)
            session.message = f"Chụp lại: {LABELS[pose]}. Các góc khác được giữ trong phiên."
            return self._snapshot(token, session)

    def _prepare(self, frame):
        validate_rgb(frame, self.service.settings)
        h, w = frame.shape[:2]
        bound = self.service.settings.webcam_width
        if max(h, w) > bound:
            scale = bound / max(h, w)
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame).copy()

    def _inspect(self, frame):
        result = self.service.runtime.detector.detect(frame)
        validate_detection_result(result)
        count = len(result.boxes)
        if count != 1:
            raise ApplicationError("enrollment_face_count", f"Cần đúng 1 khuôn mặt; phát hiện {count}.")
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = result.boxes[0]
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            raise ApplicationError("face_clipped", "Đưa toàn bộ khuôn mặt vào khung hình.")
        if min(x2 - x1, y2 - y1) < self.policy.min_face_pixels:
            raise ApplicationError("face_too_small", "Đưa khuôn mặt gần camera hơn.")
        crop = frame[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        # Fixed scale keeps the blur threshold comparable across camera distances.
        gray = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
        brightness = float(gray.mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if not self.policy.min_brightness <= brightness <= self.policy.max_brightness:
            raise ApplicationError("poor_lighting", "Ánh sáng chưa đạt; chiếu sáng đều vào mặt, tránh ngược sáng.")
        if sharpness < self.policy.min_sharpness:
            raise ApplicationError("blurry_image", "Ảnh mờ; giữ yên và kiểm tra lấy nét camera.")
        pose = estimate_head_pose(result.landmarks[0], frame.shape)
        if pose is None:
            raise ApplicationError("pose_unavailable", "Landmarks chưa đủ tin cậy để đo hướng đầu; nhìn rõ vào camera.")
        return pose, {"brightness": brightness, "sharpness": sharpness}

    def frame(self, token, frame, revision):
        with self.service.guard(), self.service._operation():
            session = self._get(token)
            if revision != session.revision:
                raise ApplicationError("stale_frame", "Frame thuộc bước cũ; gửi revision hiện tại.", 409)
            if session.paused or self._target(session) is None:
                return self._snapshot(token, session)
            self.service._require_model()
            now = self.clock()
            if session.last_frame is not None and now - session.last_frame > self.policy.max_gap:
                self._reset(session)
            session.last_frame = now
            try:
                if frame is None:
                    raise ApplicationError("camera_unavailable", "Không nhận được ảnh camera; kiểm tra quyền và kết nối.")
                image = self._prepare(frame)
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                if digest == session.last_digest:
                    raise ApplicationError("repeated_frame", "Đang chờ frame camera mới; không dùng lại một ảnh tĩnh.")
                session.last_digest = digest
                pose, quality = self._inspect(image)
                expected = self._target(session)
                if classify_pose(pose) != expected:
                    raise ApplicationError("wrong_pose", f"Tư thế chưa đạt. Hãy {LABELS[expected].lower()}.")
                angles = np.array([pose.yaw, pose.pitch, pose.roll])
                if session.stable and np.ptp(np.vstack([*session.stable, angles]), axis=0).max() > self.policy.max_angle_spread:
                    self._reset(session)
                if not session.stable:
                    session.stable_since = now
                session.stable.append(angles)
                session.stable = session.stable[-self.policy.stable_frames:]
                captured = False
                if (len(session.stable) >= self.policy.stable_frames
                        and now - session.stable_since >= self.policy.hold_seconds):
                    # Store only server-validated images, immutable to caller mutation.
                    image.setflags(write=False)
                    session.captures[expected] = image
                    session.revision += 1
                    self._reset(session)
                    captured = True
                    session.message = f"Đã chụp: {LABELS[expected]}."
                else:
                    session.message = "Đúng tư thế; giữ yên cho đến khi tự chụp."
                return {**self._snapshot(token, session), "pose": asdict(pose), "quality": quality, "captured": captured}
            except ApplicationError as error:
                self._reset(session)
                session.message = error.message
                return {**self._snapshot(token, session), **error.payload(), "captured": False}
            except Exception:
                self._reset(session)
                session.message = "Lỗi model khi xử lý camera; chưa lưu danh tính."
                raise

    def confirm(self, token, confirmed=False, revision=None):
        with self.service.guard(), self.service._operation():
            session = self._get(token)
            if confirmed is not True:
                raise ApplicationError("confirmation_required", f"Xem lại đủ {POSE_COUNT} ảnh và xác nhận trước khi lưu.")
            if revision != session.revision:
                raise ApplicationError("stale_review", "Ảnh đã thay đổi; xem lại và xác nhận lần nữa.", 409)
            if set(session.captures) != set(POSES):
                raise ApplicationError("incomplete_poses", f"Cần đủ {POSE_COUNT} tư thế khác nhau trước khi lưu.")
            if session.reviewed_revision != session.revision:
                raise ApplicationError("review_required", f"Hãy xem lại bộ {POSE_COUNT} ảnh mới nhất trước khi xác nhận lưu.")
            self.service._require_model()
            if session.version != self.service.runtime.version:
                raise ApplicationError("model_changed", "Model đã thay đổi; bắt đầu phiên đăng ký mới.", 409)
            people = self.service._registry()
            if any(p["person_id"] == session.person_id for p in people):
                raise ApplicationError("duplicate_person", "person_id đã tồn tại; không ghi đè.", 409)
            images = [session.captures[p].copy() for p in POSES]
            for expected, image in zip(POSES, images):
                validate_rgb(image, self.service.settings)
                pose, _ = self._inspect(image)
                if classify_pose(pose) != expected:
                    raise ApplicationError("pose_changed", "Ảnh không đạt tư thế khi kiểm tra lại; hãy chụp lại.")
            rt = self.service.runtime
            observed = RecordingNet(rt.net)
            with tempfile.TemporaryDirectory(prefix="face_enrollment_") as folder:
                staging = Path(folder) / "staging.sqlite"
                initialize_enrollment_registry(staging)
                result = enroll_person_from_images(session.person_id, images, rt.detector, observed, staging, rt.version)
                # Pipeline has already validated all raw outputs. Check those exact
                # forwards rather than embedding a second, possibly different set.
                try:
                    validate_embedding_consistency(
                        [output[0] for output in observed.outputs], self.policy.min_similarity, expected_count=POSE_COUNT,
                    )
                except ValueError as error:
                    raise ApplicationError("identity_inconsistent", f"{POSE_COUNT} ảnh chưa nhất quán danh tính. Hãy chụp lại hoặc bắt đầu phiên mới.") from error
                commit_identity(
                    self.service.path, session.person_id, session.display_name, result["embedding"], rt.version,
                    num_images=result["num_images"],
                )
            response = {"person_id": session.person_id, "display_name": session.display_name,
                        "num_images": result["num_images"], "status": "enrolled"}
            del self.sessions[token]
            return response
