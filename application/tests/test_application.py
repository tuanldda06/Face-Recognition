from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
import hashlib
from io import BytesIO
from pathlib import Path
import sqlite3
from time import sleep
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from FaceNet import build_model_version, initialize_enrollment_registry, recognize_image
from FaceNet import database as db
from application.api import create_app
from application.assets import verify_model
from application.config import ROOT, Settings
from application.errors import ApplicationError
from application.images import decode_image
from application.service import ApplicationService, Runtime, public_faces
from application.webcam import process_frame


class Detector:
    """Synthetic metadata for transport/orchestration tests, never accuracy evidence."""
    min_face_size, factor, thresholds, batch_size, device = 20, .709, (.6, .7, .7), 256, "cpu"

    def detect(self, image):
        self.image = image.copy()
        count = int(image[0, 0, 0])
        return SimpleNamespace(boxes=np.array([[10 + i * 5, 10, 60 + i * 5, 90]
                                               for i in range(count)], np.float32).reshape(count, 4),
                               scores=np.full(count, .95, np.float32),
                               landmarks=np.tile(np.array([[[20, 30], [40, 30], [30, 40], [22, 60], [38, 60]]], np.float32), (count, 1, 1)))

    def extract_faces_rgb(self, image, result, size, align):
        assert size == 160 and align is True
        return np.array([np.full((160, 160, 3), int(image[0, 0, 1]) + i, np.uint8)
                         for i in range(len(result.boxes))], np.uint8).reshape(-1, 160, 160, 3)


class Net:
    def __init__(self):
        self.fail = False
        self.pending = False
        self.calls = 0

    def setInput(self, batch, name):
        assert not self.pending, "Concurrent setInput/forward"
        assert name == "in:0" and batch.shape == (1, 3, 160, 160) and batch.dtype == np.float32
        self.pending = True
        self.index = round(float(batch[0, 0, 0, 0]) * 128 + 127.5)
        sleep(.001)

    def forward(self, name):
        self.pending = False
        self.calls += 1
        if self.fail:
            raise RuntimeError("forced inference failure")
        assert name == "out:0"
        result = np.zeros((1, 512), np.float32)
        result[0, self.index] = 1
        return result


def picture(count=1, index=5):
    result = np.full((240, 320, 3), (count, index, 150), np.uint8)
    return result


def encoded(image=None, format="PNG"):
    data = BytesIO()
    Image.fromarray(picture() if image is None else image).save(data, format=format)
    return data.getvalue()


@pytest.fixture
def service(tmp_path):
    settings = Settings(registry=tmp_path / "registry.sqlite")
    initialize_enrollment_registry(settings.registry)
    detector, net = Detector(), Net()
    runtime = Runtime(detector, net, build_model_version(detector))
    loader = Mock(return_value=runtime)
    service = ApplicationService(settings, runtime_loader=loader)
    service.start()
    service.test_loader = loader
    return service


@pytest.fixture
def client(service):
    with TestClient(create_app(service)) as client:
        yield client


def start_webcam_enrollment(client, person="alice"):
    return client.post("/enrollment-sessions", json={"person_id": person, "display_name": "Alice"})


def post_recognize(client, image=None):
    return client.post("/recognize", files={"image": ("face.png", encoded(image), "image/png")})


def test_health_startup_loads_once_no_registry_reset(client, service):
    assert client.get("/health").status_code == 200
    assert client.get("/health").json()["registry"] == "empty"
    service.enroll("alice", [picture()])
    service.start()
    service.test_loader.assert_called_once()
    assert len(db.list_people(service.path)) == 1
    assert client.get("/docs").status_code == 200


def test_internal_enroll_multiple_images_actual_sqlite_and_no_vectors(client, service):
    result = service.enroll("alice", [picture(), picture()])
    assert result == {"person_id": "alice", "num_images": 2, "status": "enrolled"}
    record = db.get_person(service.path, "alice", service.runtime.version)
    assert record["embedding"].shape == (512,) and record["num_images"] == 2
    assert record["embedding"][5] == 1
    assert client.get("/persons").json() == {"persons": [{"person_id": "alice", "num_images": 2}]}


def test_photo_enrollment_route_removed(client, service):
    before = service.path.read_bytes()
    result = client.post("/enroll", data={"person_id": "alice"},
                         files={"images": ("face.png", encoded(), "image/png")})
    assert result.status_code == 404
    assert "/enroll" not in client.get("/openapi.json").json()["paths"]
    assert service.path.read_bytes() == before


def test_duplicate_does_not_overwrite(client, service):
    service.enroll("alice", [picture()])
    before = service.path.read_bytes()
    assert start_webcam_enrollment(client).status_code == 409
    assert before == service.path.read_bytes()


@pytest.mark.parametrize("person", ["Unknown", "", " alice", "alice ", "a/b", "a\\b", "x" * 101])
def test_invalid_ids_rejected(client, person):
    assert start_webcam_enrollment(client, person).status_code == 422
    assert client.get("/persons").json()["persons"] == []


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\ntruncated"])
def test_invalid_upload(client, data):
    response = client.post("/recognize", files={"image": ("valid.jpg", data, "image/jpeg")})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_image"


@pytest.mark.parametrize("count", [0, 2])
def test_enrollment_face_count_no_partial_write(client, service, count):
    before = service.path.read_bytes()
    with pytest.raises(ApplicationError) as caught:
        service.enroll("alice", [picture(), picture(count)])
    assert caught.value.status == 422
    assert caught.value.code == "enrollment_face_count"
    assert before == service.path.read_bytes()


def test_no_face_known_unknown_order_and_core_consistency(client, service):
    service.enroll("alice", [picture()])
    assert post_recognize(client, picture(0)).json()["faces"] == []
    image = picture(3)
    response = post_recognize(client, image)
    assert response.status_code == 200
    faces = response.json()["faces"]
    assert [f["person_id"] for f in faces] == ["alice", None, None]
    assert [f["face_index"] for f in faces] == [0, 1, 2]
    rt = service.runtime
    expected = recognize_image(image, rt.detector, rt.net, service.path, rt.version)
    assert faces == public_faces(expected)
    assert all(f["threshold"] == 0.665513450766159 for f in faces)
    assert "best_match_id" not in response.text and "embedding" not in response.text


def test_empty_registry_unknown_without_fake_score(client):
    result = post_recognize(client).json()["faces"][0]
    assert result["status"] == "unknown" and result["person_id"] is None and result["similarity"] is None


def test_missing_registry_not_created(tmp_path):
    path = tmp_path / "never_created.sqlite"
    detector = Detector()
    service = ApplicationService(Settings(registry=path), lambda s: Runtime(detector, Net(), build_model_version(detector)))
    with TestClient(create_app(service)) as client:
        assert client.get("/health").status_code == 503
        assert post_recognize(client).status_code == 503
        assert start_webcam_enrollment(client).status_code == 503
    assert not path.exists()


def test_delete_and_missing_id(client, service):
    service.enroll("alice", [picture()])
    assert client.delete("/persons/alice").status_code == 200
    assert client.delete("/persons/alice").status_code == 404
    assert db.list_people(service.path) == []
    assert post_recognize(client).json()["faces"][0]["similarity"] is None


def test_inference_failure_not_unknown_or_partial_enrollment(client, service):
    service.runtime.net.fail = True
    response = post_recognize(client)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "pipeline_failed"
    assert "faces" not in response.json()
    with pytest.raises(ApplicationError) as caught:
        service.enroll("alice", [picture()])
    assert caught.value.code == "pipeline_failed"
    assert db.list_people(service.path) == []


def test_missing_model_health_stays_available(tmp_path):
    def loader(settings):
        raise ApplicationError("model_missing", "missing", 503)
    service = ApplicationService(Settings(registry=tmp_path / "missing.sqlite"), loader)
    with TestClient(create_app(service)) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["models"] == "unavailable"
        assert post_recognize(client).json()["error"]["code"] == "model_missing"


@pytest.mark.parametrize("content,code", [(b"version https://git-lfs.github.com/spec/v1\n", "model_lfs_pointer"), (b"wrong-model", "model_checksum")])
def test_model_asset_gate(tmp_path, content, code):
    path = tmp_path / "model.onnx"
    path.write_bytes(content)
    with pytest.raises(ApplicationError) as caught:
        verify_model(path)
    assert caught.value.code == code


def test_corrupt_registry_and_version_not_repaired(client, service):
    service.enroll("alice", [picture()])
    with closing(db.connect_database(service.path)) as connection:
        with connection:
            connection.execute("UPDATE FaceDatabase SET model_version='wrong'")
    before = service.path.read_bytes()
    assert client.get("/health").status_code == 503
    assert post_recognize(client).status_code == 503
    assert client.delete("/persons/alice").status_code == 503
    assert before == service.path.read_bytes()


def test_upload_limits_and_wrong_format(service):
    small = replace(service.settings, max_file_bytes=1024, max_request_bytes=2048)
    service.settings = small
    with TestClient(create_app(service)) as client:
        response = client.post("/recognize", content=b"x" * 2049, headers={"content-type": "application/octet-stream"})
        assert response.status_code == 413
        response = client.post("/recognize", files={"image": ("x.png", b"x" * 1025, "image/png")})
        assert response.status_code == 413
        response = client.post("/recognize", files={"image": ("x.gif", encoded(format="GIF"), "image/gif")})
        assert response.status_code == 415


def test_decoding_rgb_exif_and_dimensions():
    image = np.full((40, 60, 3), [222, 13, 55], np.uint8)
    assert np.array_equal(decode_image(encoded(image), Settings()), image)
    with pytest.raises(ApplicationError) as caught:
        decode_image(encoded(np.zeros((20, 4097, 3), np.uint8)), Settings())
    assert caught.value.status == 413
    with pytest.raises(ApplicationError):
        decode_image(encoded(np.zeros((10, 30, 3), np.uint8)), Settings())
    exif = Image.Exif()
    exif[274] = 6
    data = BytesIO()
    Image.fromarray(image).save(data, format="JPEG", exif=exif)
    assert decode_image(data.getvalue(), Settings()).shape == (60, 40, 3)


def test_max_faces_and_max_images(client, service):
    with pytest.raises(ApplicationError) as caught:
        service.enroll("alice", [picture()] * 11)
    assert caught.value.status == 422
    assert post_recognize(client, picture(21)).status_code == 422
    assert service.runtime.net.calls == 0


def test_webcam_sequential_frames_and_state_isolation(service):
    state = {}
    for index in range(3):
        annotated, result, state = process_frame(service, picture(), state)
        assert annotated.shape == picture().shape
        assert result["frames_processed"] == index + 1
        assert result["frame_latency_ms"] > 0 and result["stream_fps"] > 0
    assert process_frame(service, picture(), {})[1]["frames_processed"] == 1
    with pytest.raises(ApplicationError) as caught:
        process_frame(service, None)
    assert caught.value.code == "camera_unavailable"


def test_concurrent_requests_serialize_model_and_duplicate_writes(service):
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(service.recognize, [picture()] * 6))
    assert all(r["face_count"] == 1 for r in results)
    def enroll(_):
        try:
            return service.enroll("racing_id", [picture()])["status"]
        except ApplicationError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(enroll, range(2)))
    assert sorted(results) == ["duplicate_person", "enrolled"]


def test_two_service_instances_share_registry_lock(service):
    second = ApplicationService(replace(service.settings, lock_timeout=.05), service._loader)
    with service.guard(), ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(second.persons)
        with pytest.raises(ApplicationError) as caught:
            future.result()
    assert caught.value.code == "service_busy"


def test_benchmark_summary():
    from application.benchmark import summarize
    assert summarize([])["status"] == "not_measured"
    assert summarize([10, 20, 30])["p50_ms"] == 20


def test_ui_stop_skips_inference_and_start_resets_session(service):
    from application.ui import build_ui
    demo = build_ui(service)
    stream = next(fn.fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "stream")
    before = service.runtime.net.calls
    _, _, state, message = stream(picture(), False, {})
    assert service.runtime.net.calls == before and state == {}
    assert "dừng" in message
    _, result, state, _ = stream(picture(), True, {})
    assert result["frames_processed"] == 1
    _, _, state, _ = stream(picture(), True, state)
    assert state["frames"] == 2


def test_wrong_schema_and_corrupt_checksum_are_unavailable(service):
    service.enroll("alice", [picture()])
    with closing(db.connect_database(service.path)) as connection:
        with connection:
            connection.execute("UPDATE FaceDatabase SET embedding_sha256=?", ("0" * 64,))
    before = service.path.read_bytes()
    assert service.health()["registry"] == "invalid"
    with pytest.raises(ApplicationError) as caught:
        service.recognize(picture())
    assert caught.value.status == 503 and service.path.read_bytes() == before


def test_late_inference_failure_rolls_back_entire_enrollment(service, monkeypatch):
    original = service.runtime.net.forward
    def forward(name):
        if service.runtime.net.calls == 1:
            raise RuntimeError("second image failed")
        return original(name)
    monkeypatch.setattr(service.runtime.net, "forward", forward)
    before = service.path.read_bytes()
    with pytest.raises(ApplicationError) as caught:
        service.enroll("alice", [picture(), picture()])
    assert caught.value.status == 500 and service.path.read_bytes() == before


def test_ui_mount_webcam_only_enrollment_and_continuous_streams(service):
    with TestClient(create_app(service, with_ui=True)) as client:
        assert client.get("/ui/").status_code == 200
        config = client.get("/ui/config").json()
    tabs = [part["props"]["label"] for part in config["components"] if part["type"] == "tabitem"]
    assert tabs == ["1 · Đăng ký webcam · 4 góc", "2 · Nhận diện ảnh", "3 · Webcam trực tiếp", "4 · Danh sách đăng ký"]
    streams = [item for item in config["dependencies"] if any(target[1] == "stream" for target in item["targets"])]
    assert len(streams) == 2 and all(item["trigger_mode"] == "multiple" for item in streams)
    assert "#enrollment-camera-status" in config["css"]


def test_ui_registration_list_confirms_one_deletion(service):
    from application.ui import build_ui
    from application.webcam_registry import commit_identity, read_names

    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1
    commit_identity(service.path, "alice", "Nguyễn An", embedding, service.runtime.version, num_images=4)
    service.enroll("legacy", [picture()])
    demo = build_ui(service)
    callbacks = {f.fn.__name__: f.fn for f in demo.fns.values() if hasattr(f.fn, "__name__")}

    rows, choice, *_ = callbacks["refresh_registrations"]()
    assert rows == [["alice", "Nguyễn An", 4], ["legacy", "—", 1]]
    assert choice["choices"] == [("Nguyễn An · alice", "alice"), ("legacy", "legacy")]
    pending, prompt, question, _ = callbacks["request_registration_delete"]("alice")
    assert prompt["visible"] and "alice" in question
    assert len(service.persons()) == 2

    callbacks["confirm_registration_delete"](None)
    assert len(service.persons()) == 2
    rows, _, _, pending_after, prompt_after, message = callbacks["confirm_registration_delete"](pending)
    assert rows == [["legacy", "—", 1]]
    assert pending_after is None and prompt_after["visible"] is False
    assert "Đã xóa" in message and read_names(service.path) == {}
