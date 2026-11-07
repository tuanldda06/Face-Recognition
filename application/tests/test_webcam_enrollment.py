"""Orchestration tests with synthetic landmarks/net, real core pipeline + SQLite.

These do not claim real webcam accuracy. See test_real_smoke for real ML assets.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from io import BytesIO
import sqlite3
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from FaceNet import build_model_version, initialize_enrollment_registry
from FaceNet import database as db
from application.api import create_app
from application.config import Settings
from application.errors import ApplicationError
from application.head_pose import _MODEL_POINTS
from application.service import ApplicationService, Runtime
from application.webcam_registry import commit_identity, metadata_path, read_names


class Clock:
    def __init__(self):
        self.now = 100.0
    def __call__(self):
        return self.now
    def advance(self, seconds=.3):
        self.now += seconds


class Detector:
    min_face_size, factor, thresholds, batch_size, device = 20, .709, (.6, .7, .7), 256, "cpu"
    fail = False

    def detect(self, image):
        if self.fail:
            raise RuntimeError("model failed")
        n, pose = map(int, image[0, 0, :2])
        yaw, pitch = [(0, 0), (-25, 0), (25, 0), (0, -25), (0, 25)][pose]
        y, p = np.deg2rad([yaw, pitch])
        ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
        rotation, _ = cv2.Rodrigues(ry @ rx)
        points, _ = cv2.projectPoints(_MODEL_POINTS, rotation, np.array([0., 0., 180.]),
                                     np.array([[320., 0, 160.], [0, 320., 120.], [0, 0, 1.]]), None)
        return SimpleNamespace(boxes=np.tile(np.array([[50, 15, 270, 225]], np.float32), (n, 1)),
                               scores=np.full(n, .99, np.float32),
                               landmarks=np.tile(points.reshape(1, 5, 2).astype(np.float32), (n, 1, 1)))

    def extract_faces_rgb(self, image, result, size, align):
        return np.full((len(result.boxes), 160, 160, 3), int(image[0, 0, 2]), np.uint8)


class Net:
    def __init__(self):
        self.fail = False
        self.calls = 0
    def setInput(self, batch, name):
        self.index = round(float(batch[0, 0, 0, 0]) * 128 + 127.5)
    def forward(self, name):
        self.calls += 1
        if self.fail:
            raise RuntimeError("inference failure")
        output = np.zeros((1, 512), np.float32)
        output[0, self.index] = 1
        return output


def frame(pose=0, count=1, identity=0, sequence=0):
    image = np.random.default_rng(42).integers(65, 190, (240, 320, 3), dtype=np.uint8)
    image[0, 0] = (count, pose, identity)
    image[1, 1, 0] = sequence
    return image


def png(image):
    data = BytesIO()
    Image.fromarray(image).save(data, format="PNG")
    return data.getvalue()


@pytest.fixture
def service(tmp_path):
    path = tmp_path / "registry.sqlite"
    initialize_enrollment_registry(path)
    detector = Detector()
    rt = Runtime(detector, Net(), build_model_version(detector))
    result = ApplicationService(Settings(registry=path), lambda _: rt)
    result.start()
    result.webcam_enrollment.clock = Clock()
    return result


def capture(service, state, pose, identity=0):
    manager = service.webcam_enrollment
    for i in range(4):
        manager.clock.advance()
        state = manager.frame(state["session_id"], frame(pose, identity=identity, sequence=i), state["revision"])
    assert state.get("captured"), state
    return state


def complete(service, identity=0):
    state = service.webcam_enrollment.start("alice", "Nguyễn An")
    for i in range(4):
        state = capture(service, state, i, identity)
    service.webcam_enrollment.review(state["session_id"])
    return state


def test_four_poses_confirm_atomic_name_and_recognition(service):
    before = service.path.read_bytes()
    state = complete(service)
    manager = service.webcam_enrollment
    assert state["progress"] == state["total"] == 4 and state["status"] == "review"
    assert service.path.read_bytes() == before and not metadata_path(service.path).exists()
    assert len(manager.review(state["session_id"])) == 4
    result = manager.confirm(state["session_id"], True, state["revision"])
    assert result == {"person_id": "alice", "display_name": "Nguyễn An", "num_images": 4, "status": "enrolled"}
    assert db.get_person(service.path, "alice", service.runtime.version)["num_images"] == 4
    assert service.persons() == [{"person_id": "alice", "num_images": 4, "display_name": "Nguyễn An"}]
    assert service.recognize(frame())["faces"][0]["person_id"] == "alice"
    assert service.recognize(frame(identity=1))["faces"][0]["status"] == "unknown"
    with pytest.raises(ApplicationError, match="Phiên"):
        manager.confirm(state["session_id"], True, state["revision"])
    service.delete("alice")
    assert service.persons() == [] and read_names(service.path) == {}


@pytest.mark.parametrize("count", [0, 2])
def test_face_count_breaks_stability_and_never_writes(service, count):
    manager = service.webcam_enrollment
    state = manager.start("alice", "An")
    manager.frame(state["session_id"], frame(), 0)
    result = manager.frame(state["session_id"], frame(count=count, sequence=1), 0)
    assert result["error"]["code"] == "enrollment_face_count"
    assert result["stable_frames"] == 0 and result["progress"] == 0
    assert db.list_people(service.path) == []


def test_wrong_pose_unstable_and_no_duplicate_pose(service):
    manager = service.webcam_enrollment
    state = manager.start("alice", "An")
    bad = manager.frame(state["session_id"], frame(1), 0)
    assert bad["error"]["code"] == "wrong_pose"
    for i in range(3):
        manager.clock.advance()
        state = manager.frame(state["session_id"], frame(sequence=i), 0)
        assert state["progress"] == 0
    manager.clock.advance()
    state = manager.frame(state["session_id"], frame(sequence=3), 0)
    assert state["progress"] == 1
    for i in range(8):
        manager.clock.advance()
        state = manager.frame(state["session_id"], frame(sequence=i), state["revision"])
    assert state["progress"] == 1 and state["next_pose"] == "left"
    assert state["stable_frames"] == 0


def test_repeated_static_frame_cannot_supply_stable_frames(service):
    manager = service.webcam_enrollment
    state = manager.start("alice", "An")
    for _ in range(6):
        manager.clock.advance()
        result = manager.frame(state["session_id"], frame(), 0)
    assert result["error"]["code"] == "repeated_frame" and result["progress"] == 0


@pytest.mark.parametrize("mode,code", [("dark", "poor_lighting"), ("bright", "poor_lighting"), ("blur", "blurry_image")])
def test_quality_on_face_region(service, mode, code):
    state = service.webcam_enrollment.start("alice", "An")
    image = frame()
    image[15:225, 50:270] = {"dark": 5, "bright": 250, "blur": 125}[mode]
    result = service.webcam_enrollment.frame(state["session_id"], image, 0)
    assert result["error"]["code"] == code and result["progress"] == 0


def test_retake_one_pose_preserves_others_and_invalidates_review(service):
    state = complete(service)
    manager = service.webcam_enrollment
    old = dict(manager.review(state["session_id"]))
    retake = manager.retake(state["session_id"], "left")
    assert retake["progress"] == 3 and retake["next_pose"] == "left"
    with pytest.raises(ApplicationError) as caught:
        manager.confirm(state["session_id"], True, state["revision"])
    assert caught.value.code == "stale_review"
    with pytest.raises(ApplicationError) as caught:
        manager.frame(state["session_id"], frame(1), state["revision"])
    assert caught.value.code == "stale_frame"
    new = capture(service, retake, 1)
    assert new["progress"] == 4
    assert all(np.array_equal(image, old[pose]) for pose, image in manager.review(state["session_id"]) if pose != "left")


def test_cancel_destroys_images_and_cross_session_isolation(service):
    manager = service.webcam_enrollment
    first = capture(service, manager.start("alice", "An"), 0)
    second = manager.start("bob", "Bình")
    assert len(manager.review(first["session_id"])) == 1 and manager.review(second["session_id"]) == []
    manager.cancel(first["session_id"])
    for operation in (lambda: manager.status(first["session_id"]),
                      lambda: manager.frame(first["session_id"], frame(), 0),
                      lambda: manager.confirm(first["session_id"], True, first["revision"])):
        with pytest.raises(ApplicationError) as caught:
            operation()
        assert caught.value.code == "session_missing"
    assert db.list_people(service.path) == []


def test_invalid_camera_disconnect_gap_and_resume(service):
    manager = service.webcam_enrollment
    state = manager.start("alice", "An")
    state = manager.frame(state["session_id"], frame(), 0)
    assert state["stable_frames"] == 1
    result = manager.frame(state["session_id"], None, 0)
    assert result["error"]["code"] == "camera_unavailable" and result["stable_frames"] == 0
    manager.frame(state["session_id"], frame(sequence=1), 0)
    manager.clock.advance(3)
    assert manager.status(state["session_id"])["stable_frames"] == 0
    paused = manager.pause(state["session_id"])
    assert manager.frame(state["session_id"], frame(sequence=2), paused["revision"])["status"] == "paused"
    resumed = manager.resume(state["session_id"])
    assert capture(service, resumed, 0)["progress"] == 1


def test_session_expiry_and_capacity(service):
    manager = service.webcam_enrollment
    manager.policy = replace(manager.policy, max_sessions=1, ttl_seconds=10)
    state = manager.start("alice", "An")
    with pytest.raises(ApplicationError) as caught:
        manager.start("bob", "Bình")
    assert caught.value.code == "sessions_full"
    manager.clock.advance(11)
    with pytest.raises(ApplicationError) as caught:
        manager.status(state["session_id"])
    assert caught.value.code == "session_missing" and not manager.sessions
    assert manager.start("bob", "Bình")["progress"] == 0


def test_confirmation_required_and_missing_poses(service):
    manager = service.webcam_enrollment
    state = manager.start("alice", "An")
    for accepted, code in [(False, "confirmation_required"), (True, "incomplete_poses")]:
        with pytest.raises(ApplicationError) as caught:
            manager.confirm(state["session_id"], accepted, state["revision"])
        assert caught.value.code == code
    assert db.list_people(service.path) == []


def test_mixed_identity_rejected_before_destination_write(service):
    manager = service.webcam_enrollment
    state = manager.start("alice", "An")
    for pose in range(4):
        state = capture(service, state, pose, identity=int(pose == 3))
    manager.review(state["session_id"])
    before = service.path.read_bytes()
    with pytest.raises(ApplicationError) as caught:
        manager.confirm(state["session_id"], True, state["revision"])
    assert caught.value.code == "identity_inconsistent"
    assert service.path.read_bytes() == before and not metadata_path(service.path).exists()


@pytest.mark.parametrize("component", ["detector", "net"])
def test_model_failure_no_partial_enrollment(service, component):
    state = complete(service)
    getattr(service.runtime, component).fail = True
    with pytest.raises(ApplicationError) as caught:
        service.webcam_enrollment.confirm(state["session_id"], True, state["revision"])
    assert caught.value.code == "pipeline_failed" and db.list_people(service.path) == []
    assert not metadata_path(service.path).exists()


def test_duplicate_after_session_start_preserves_existing_record(service):
    state = complete(service)
    service.enroll("alice", [frame(identity=1)])
    before = service.path.read_bytes()
    with pytest.raises(ApplicationError) as caught:
        service.webcam_enrollment.confirm(state["session_id"], True, state["revision"])
    assert caught.value.code == "duplicate_person" and service.path.read_bytes() == before
    with pytest.raises(ApplicationError) as caught:
        service.webcam_enrollment.start("alice", "Name")
    assert caught.value.code == "duplicate_person"


def test_sidecar_failure_rolls_back_main_insert_and_existing_data(service, monkeypatch):
    service.enroll("existing", [frame(identity=1)])
    state = complete(service)
    # Existing stale metadata deliberately causes the SECOND insert to fail.
    names = metadata_path(service.path)
    from application.webcam_registry import NAME_SQL
    with closing(sqlite3.connect(names)) as conn:
        conn.execute(NAME_SQL)
        conn.execute("INSERT INTO EnrollmentNames VALUES ('alice','old')")
        conn.commit()
    before_main, before_names = service.path.read_bytes(), names.read_bytes()
    with pytest.raises(ApplicationError) as caught:
        service.webcam_enrollment.confirm(state["session_id"], True, state["revision"])
    assert caught.value.code == "pipeline_failed"
    assert service.path.read_bytes() == before_main and names.read_bytes() == before_names
    assert [p["person_id"] for p in db.list_people(service.path)] == ["existing"]


def test_corrupt_database_at_confirmation_no_repair(service):
    state = complete(service)
    service.path.write_bytes(b"corrupt")
    with pytest.raises(ApplicationError) as caught:
        service.webcam_enrollment.confirm(state["session_id"], True, state["revision"])
    assert caught.value.code == "registry_invalid" and service.path.read_bytes() == b"corrupt"


def test_transaction_rechecks_duplicate_and_reserved_rows(service):
    embedding = np.eye(1, 512, dtype=np.float32)[0]
    commit_identity(service.path, "alice", "An", embedding, service.runtime.version, num_images=4)
    with pytest.raises(ApplicationError) as caught:
        commit_identity(service.path, "alice", "Changed", embedding, service.runtime.version, num_images=4)
    assert caught.value.code == "duplicate_person" and read_names(service.path)["alice"] == "An"
    db.add_person(service.path, "Unknown", embedding, 1, service.runtime.version)
    with pytest.raises(ValueError, match="reserved"):
        commit_identity(service.path, "bob", "Bình", embedding, service.runtime.version, num_images=4)
    assert db.get_person(service.path, "bob", service.runtime.version) is None


def test_wal_registry_refused_without_identity_write(service):
    with closing(db.connect_database(service.path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
    embedding = np.eye(1, 512, dtype=np.float32)[0]
    with pytest.raises(ValueError, match="DELETE"):
        commit_identity(service.path, "alice", "An", embedding, service.runtime.version, num_images=4)
    assert db.list_people(service.path) == []


def test_concurrent_confirm_and_cancel_serialize(service):
    state = complete(service)
    manager = service.webcam_enrollment
    def confirm(_):
        try:
            return manager.confirm(state["session_id"], True, state["revision"])["status"]
        except ApplicationError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(confirm, range(2)))
    assert sorted(outcomes) == ["enrolled", "session_missing"]
    assert len(db.list_people(service.path)) == 1


def test_api_full_session_review_retake_confirm_and_limits(service):
    manager = service.webcam_enrollment
    with TestClient(create_app(service)) as client:
        state = client.post("/enrollment-sessions", json={"person_id": "alice", "display_name": "An"}).json()
        token = state["session_id"]
        root = f"/enrollment-sessions/{token}"
        for pose in range(4):
            for seq in range(4):
                manager.clock.advance()
                response = client.post(root + "/frames", data={"revision": state["revision"]},
                    files={"image": ("frame.png", png(frame(pose, sequence=seq)), "image/png")})
                assert response.status_code == 200, response.text
                state = response.json()
        assert state["progress"] == state["total"] == 4
        assert state["next_pose"] is None and state["status"] == "review"
        review = client.get(root + "/review").json()["images"]
        assert [item["pose"] for item in review] == ["front", "left", "right", "up"]
        assert all(item["image"].startswith("data:image/jpeg;base64,") for item in review)
        assert client.post(root + "/retake/down").status_code == 422
        assert client.post(root + "/confirm", json={"confirmed": False, "revision": state["revision"]}).status_code == 422
        assert client.post(root + "/confirm", json={"confirmed": True, "revision": state["revision"]}).status_code == 201
        assert client.get(root).status_code == 404
        assert client.get("/persons").json()["persons"][0]["display_name"] == "An"


def test_api_invalid_frame_breaks_stability_cancel_and_missing_token(service):
    with TestClient(create_app(service)) as client:
        state = client.post("/enrollment-sessions", json={"person_id": "alice", "display_name": "An"}).json()
        root = f'/enrollment-sessions/{state["session_id"]}'
        service.webcam_enrollment.frame(state["session_id"], frame(), 0)
        response = client.post(root + "/frames", data={"revision": 0}, files={"image": ("bad.png", b"bad", "image/png")})
        assert response.status_code == 422 and client.get(root).json()["stable_frames"] == 0
        assert client.post(root + "/pause").json()["status"] == "paused"
        assert client.post(root + "/resume").json()["status"] == "capturing"
        assert client.delete(root).json()["status"] == "cancelled"
        assert client.post(root + "/frames", data={"revision": 0}, files={"image": ("frame.png", png(frame()), "image/png")}).status_code == 404


def test_ui_enrollment_callbacks_review_confirmation_and_cancel(service):
    from application.ui import build_ui
    demo = build_ui(service)
    callbacks = {f.fn.__name__: f.fn for f in demo.fns.values() if hasattr(f.fn, "__name__")}
    state, message, _, _, _, save, _ = callbacks["start_enrollment"]("alice", "An", {})
    assert "0/4" in message and save["interactive"] is False
    for pose in range(4):
        for seq in range(4):
            service.webcam_enrollment.clock.advance()
            output = callbacks["enrollment_stream"](frame(pose, sequence=seq), state)
            state = output[0]
    assert "4/4" in output[1] and output[3].count("data:image/jpeg;base64,") == 4
    assert output[5]["interactive"] is True
    assert callbacks["save_enrollment"](state, False)[2]["error"]["code"] == "confirmation_required"
    assert callbacks["cancel_enrollment"](state)[0] == {} and db.list_people(service.path) == []


def test_latest_review_required_after_retake(service):
    manager = service.webcam_enrollment
    state = complete(service)
    old_review = manager.review_snapshot(state["session_id"])
    state = manager.retake(state["session_id"], "left")
    state = capture(service, state, 1)
    with pytest.raises(ApplicationError) as caught:
        manager.confirm(state["session_id"], True, state["revision"])
    assert caught.value.code == "review_required"
    new_review = manager.review_snapshot(state["session_id"])
    assert new_review["revision"] > old_review["revision"]
    assert manager.confirm(state["session_id"], True, new_review["revision"])["status"] == "enrolled"
    assert manager.cancel(state["session_id"])["status"] == "session_ended"


def test_failed_first_commit_empty_sidecar_does_not_poison_registry(service):
    embedding = np.eye(1, 512, dtype=np.float32)[0]
    # Directly provoke final-transaction revalidation AFTER ATTACH created the file.
    db.add_person(service.path, "existing", embedding, 1, "changed-version")
    with pytest.raises(ValueError, match="model_version"):
        commit_identity(service.path, "alice", "An", embedding, service.runtime.version, num_images=4)
    assert metadata_path(service.path).exists() and read_names(service.path) == {}
    from application.webcam_registry import delete_identity
    delete_identity(service.path, "existing")
    assert service.persons() == []
    commit_identity(service.path, "alice", "An", embedding, service.runtime.version, num_images=4)
    assert read_names(service.path) == {"alice": "An"}


def test_stable_frame_count_alone_does_not_bypass_hold_time(service):
    manager = service.webcam_enrollment
    state = manager.start("alice", "An")
    for seq in range(8):
        manager.clock.advance(.01)
        result = manager.frame(state["session_id"], frame(sequence=seq), 0)
    assert result["stable_frames"] == 4 and result["progress"] == 0


def test_head_jitter_resets_stability(service, monkeypatch):
    from application.head_pose import HeadPose
    import application.webcam_enrollment as module
    manager = service.webcam_enrollment
    state = manager.start("alice", "An")
    for seq, yaw in enumerate([-4, 4, -4, 4, -4, 4]):
        monkeypatch.setattr(module, "estimate_head_pose", lambda *_, angle=yaw: HeadPose(angle, 0, 0))
        manager.clock.advance()
        result = manager.frame(state["session_id"], frame(sequence=seq), 0)
    assert result["stable_frames"] == 1 and result["progress"] == 0


def test_invalid_model_detection_is_infrastructure_error(service, monkeypatch):
    state = service.webcam_enrollment.start("alice", "An")
    monkeypatch.setattr(service.runtime.detector, "detect", lambda image: SimpleNamespace(boxes=None))
    with pytest.raises(ApplicationError) as caught:
        service.webcam_enrollment.frame(state["session_id"], frame(), 0)
    assert caught.value.code == "pipeline_failed"


def test_ui_stream_and_controls_have_independent_queue_slots(service):
    from application.ui import build_ui
    demo = build_ui(service)
    functions = {getattr(f.fn, "__name__", ""): f for f in demo.fns.values()}
    assert functions["enrollment_stream"].concurrency_id != functions["save_enrollment"].concurrency_id
    assert functions["enrollment_stream"].concurrency_id != functions["cancel_enrollment"].concurrency_id


def test_ui_recovers_latest_revision_for_stale_inflight_frame(service):
    from application.ui import build_ui
    demo = build_ui(service)
    callbacks = {getattr(f.fn, "__name__", ""): f.fn for f in demo.fns.values()}
    state = service.webcam_enrollment.start("alice", "An")
    old = {"session_id": state["session_id"], "revision": state["revision"]}
    state = capture(service, state, 0)
    result = callbacks["enrollment_stream"](frame(1), old)
    assert result[0]["revision"] == state["revision"] and result[2]["progress"] == 1


def test_ui_late_stream_does_not_overwrite_completed_control_message(service):
    from application.ui import build_ui
    demo = build_ui(service)
    callback = next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "enrollment_stream")
    state = service.webcam_enrollment.start("alice", "An")
    service.webcam_enrollment.cancel(state["session_id"])
    result = callback(frame(), {"session_id": state["session_id"], "revision": 0})
    assert all(output == {"__type__": "update"} for output in result)
