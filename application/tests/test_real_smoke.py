from contextlib import closing
import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from FaceNet import initialize_enrollment_registry, recognize_image
from FaceNet import database as db
from application.api import create_app
from application.assets import verify_model
from application.config import ROOT, Settings
from application.errors import ApplicationError
from application.images import decode_image
from application.service import ApplicationService, load_runtime, public_faces
from application.webcam import process_frame


@pytest.fixture(scope="module")
def runtime():
    try:
        verify_model()
    except ApplicationError as error:
        if error.code in {"model_missing", "model_lfs_pointer"}:
            pytest.skip("BLOCKED real-model smoke: " + error.code)
        raise  # Wrong hash is a failure, never silently skipped.
    return load_runtime(Settings())


def test_real_models_api_sqlite_multi_face_webcam_and_core_consistency(runtime, tmp_path):
    path = tmp_path / "real.sqlite"
    initialize_enrollment_registry(path)
    service = ApplicationService(Settings(registry=path), lambda settings: runtime)
    single_bytes = (ROOT / "integration/tests/fixtures/mtcnn_single.png").read_bytes()
    group_bytes = (ROOT / "integration/tests/fixtures/mtcnn_multi.jpg").read_bytes()
    single = decode_image(single_bytes, service.settings)
    group = decode_image(group_bytes, service.settings)
    with TestClient(create_app(service)) as client:
        assert client.get("/health").status_code == 200
        empty = client.post("/recognize", files={"image": ("single.png", single_bytes, "image/png")})
        assert empty.status_code == 200
        assert empty.json()["face_count"] == 1
        assert empty.json()["faces"][0]["similarity"] is None
        enrolled = service.enroll("fixture_smoke", [single])
        assert enrolled["status"] == "enrolled"
        assert "embedding" not in enrolled
        record = db.get_person(path, "fixture_smoke", runtime.version)
        assert record["embedding"].shape == (512,) and record["embedding"].dtype == np.float32
        assert np.isclose(np.linalg.norm(record["embedding"]), 1)
        known = client.post("/recognize", files={"image": ("single.png", single_bytes, "image/png")})
        assert known.status_code == 200 and known.json()["faces"][0]["person_id"] == "fixture_smoke"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        response = client.post("/recognize", files={"image": ("group.jpg", group_bytes, "image/jpeg")})
        assert response.status_code == 200
        assert response.json()["face_count"] == 5
        direct = recognize_image(group, runtime.detector, runtime.net, path, runtime.version)
        assert response.json()["faces"] == public_faces(direct)
        assert any(row["status"] == "unknown" for row in direct)
        assert before == hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ApplicationError) as invalid:
            service.enroll("group_rejected", [group])
        assert invalid.value.status == 422 and len(service.persons()) == 1
        assert service.recognize(np.zeros((240, 320, 3), np.uint8))["face_count"] == 0
        state = {}
        for i in range(3):
            annotated, result, state = process_frame(service, group, state)
            assert result["frames_processed"] == i + 1 and result["face_count"] > 0
            assert annotated.shape[1] <= 640
        assert client.delete("/persons/fixture_smoke").status_code == 200
        assert db.list_people(path) == []


def test_real_benchmark_stage_instrumentation(runtime, tmp_path):
    from application.benchmark import run_case
    path = tmp_path / "benchmark.sqlite"
    initialize_enrollment_registry(path)
    service = ApplicationService(Settings(registry=path), lambda settings: runtime)
    service.start()
    fixture = ROOT / "integration/tests/fixtures/mtcnn_single.png"
    service.enroll("benchmark", [decode_image(fixture.read_bytes(), service.settings)])
    detector = runtime.detector
    result = run_case(service, fixture, repetitions=2, warmup=1)
    assert runtime.detector is detector
    assert result["detected_faces_each_run"] == [1, 1]
    assert all(item["status"] == "measured" for item in result["latency"].values())
    assert result["fps"] > 0
