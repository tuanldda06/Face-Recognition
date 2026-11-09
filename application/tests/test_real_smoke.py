# Kiểm thử tích hợp bằng model thật: API, lưu SQLite, nhận diện webcam và đo thời gian.
# Lấy closing từ contextlib để quản lý tài nguyên bằng with, yield và thao tác dọn tự động.
from contextlib import closing
# Nạp hashlib để tính mã băm để kiểm tra nội dung file/ảnh/vector.
import hashlib

# Nạp numpy dưới tên ngắn np để tính toán mảng ảnh, vector và ma trận.
import numpy as np
# Nạp pytest để viết và chạy bài kiểm thử tự động.
import pytest
# Lấy TestClient từ fastapi.testclient để gọi thử API mà không cần mở cổng máy chủ thật.
from fastapi.testclient import TestClient

# Lấy initialize_enrollment_registry, recognize_image từ FaceNet.
from FaceNet import initialize_enrollment_registry, recognize_image
# Lấy database với bí danh db từ FaceNet.
from FaceNet import database as db
# Lấy create_app từ application.api.
from application.api import create_app
# Lấy verify_model từ application.assets.
from application.assets import verify_model
# Lấy ROOT, Settings từ application.config.
from application.config import ROOT, Settings
# Lấy ApplicationError từ application.errors.
from application.errors import ApplicationError
# Lấy decode_image từ application.images.
from application.images import decode_image
# Lấy ApplicationService, load_runtime, public_faces từ application.service.
from application.service import ApplicationService, load_runtime, public_faces
# Lấy process_frame từ application.webcam.
from application.webcam import process_frame


# Nạp model thật một lần cho module; chỉ bỏ qua khi thiếu file hoặc còn con trỏ LFS.
# Decorator fixture: pytest tự tạo dữ liệu/phụ thuộc này khi bài test yêu cầu tên hàm.
@pytest.fixture(scope="module")
# Định nghĩa hàm runtime().
def runtime():
    # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
    try:
        # Xác minh file model bằng dung lượng và SHA-256 bằng verify_model(...).
        verify_model()
    # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
    except ApplicationError as error:
        # Chỉ chạy khối bên dưới khi error.code in {'model_missing', 'model_lfs_pointer'} cho kết quả đúng.
        if error.code in {"model_missing", "model_lfs_pointer"}:
            # Đánh dấu bỏ qua bài test với lý do cụ thể bằng pytest.skip(...); dữ liệu/tham số: 'BLOCKED real-model smoke: ' + error.code.
            pytest.skip("BLOCKED real-model smoke: " + error.code)
        # Ném lại lỗi đang xử lý để phía gọi biết thao tác chưa thành công.
        raise  # Sai mã băm là lỗi kiểm thử, không được âm thầm bỏ qua.
    # Kết thúc hàm và trả nạp detector và FaceNet theo cấu hình bằng load_runtime(...); dữ liệu/tham số: Settings() (tạo cấu hình bằng Settings(...)).
    return load_runtime(Settings())


# Kiểm tra: Model thật nhận diện qua API/webcam và cho kết quả khớp pipeline gốc.
# Định nghĩa hàm test_real_models_api_sqlite_multi_face_webcam_and_core_consistency(runtime, tmp_path).
def test_real_models_api_sqlite_multi_face_webcam_and_core_consistency(runtime, tmp_path):
    # Gán cho path: ghép đường dẫn bằng toán tử / của pathlib trong tmp_path / 'real.sqlite'. Ở đây / nối thư mục với tên file, không phải phép chia số.
    path = tmp_path / "real.sqlite"
    # Khởi tạo kho SQLite theo cấu trúc FaceNet bằng initialize_enrollment_registry(...); dữ liệu/tham số: path.
    initialize_enrollment_registry(path)
    # Gán cho service: tạo dịch vụ dùng chung cho nhận diện và quản lý hồ sơ bằng ApplicationService(...); dữ liệu/tham số: Settings(registry=path) (tạo cấu hình bằng Settings(...)), lambda settings: runtime.
    service = ApplicationService(Settings(registry=path), lambda settings: runtime)
    # Gán cho single_bytes: đọc toàn bộ file dưới dạng byte qua (ROOT / 'integration/tests/fixtures/mtcnn_single.png').read_bytes(...).
    single_bytes = (ROOT / "integration/tests/fixtures/mtcnn_single.png").read_bytes()
    # Gán cho group_bytes: đọc toàn bộ file dưới dạng byte qua (ROOT / 'integration/tests/fixtures/mtcnn_multi.jpg').read_bytes(...).
    group_bytes = (ROOT / "integration/tests/fixtures/mtcnn_multi.jpg").read_bytes()
    # Gán cho single: giải mã byte thành ảnh RGB đã kiểm tra bằng decode_image(...); dữ liệu/tham số: single_bytes, service.settings.
    single = decode_image(single_bytes, service.settings)
    # Gán cho group: giải mã byte thành ảnh RGB đã kiểm tra bằng decode_image(...); dữ liệu/tham số: group_bytes, service.settings.
    group = decode_image(group_bytes, service.settings)
    # Mở khối with để tạo trình gọi API nội bộ cho kiểm thử bằng TestClient(...); dữ liệu/tham số: create_app(service) (tạo API và tùy chọn gắn giao diện bằng create_app(...)); gọi tài nguyên là client; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with TestClient(create_app(service)) as client:
        # Kiểm chứng điều kiện client.get('/health').status_code == 200; sai thì test dừng bằng AssertionError.
        assert client.get("/health").status_code == 200
        # Gán cho empty: gửi yêu cầu HTTP POST qua client.post(...); dữ liệu/tham số: '/recognize', files={'image': ('single.png', single_bytes, 'image/png')}.
        empty = client.post("/recognize", files={"image": ("single.png", single_bytes, "image/png")})
        # Kiểm chứng điều kiện empty.status_code == 200; sai thì test dừng bằng AssertionError.
        assert empty.status_code == 200
        # Kiểm chứng điều kiện empty.json()['face_count'] == 1; sai thì test dừng bằng AssertionError.
        assert empty.json()["face_count"] == 1
        # Kiểm chứng điều kiện empty.json()['faces'][0]['similarity'] is None; sai thì test dừng bằng AssertionError.
        assert empty.json()["faces"][0]["similarity"] is None
        # Gán cho enrolled: đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'fixture_smoke', [single].
        enrolled = service.enroll("fixture_smoke", [single])
        # Kiểm chứng điều kiện enrolled['status'] == 'enrolled'; sai thì test dừng bằng AssertionError.
        assert enrolled["status"] == "enrolled"
        # Kiểm chứng điều kiện 'embedding' not in enrolled; sai thì test dừng bằng AssertionError.
        assert "embedding" not in enrolled
        # Gán cho record: gọi db.get_person(...) với path, 'fixture_smoke', runtime.version; dữ liệu/tham số: path, 'fixture_smoke', runtime.version.
        record = db.get_person(path, "fixture_smoke", runtime.version)
        # Kiểm chứng điều kiện record['embedding'].shape == (512,) and record['embedding'].dtype == np.float32; sai thì test dừng bằng AssertionError.
        assert record["embedding"].shape == (512,) and record["embedding"].dtype == np.float32
        # Kiểm chứng điều kiện np.isclose(np.linalg.norm(record['embedding']), 1); sai thì test dừng bằng AssertionError.
        assert np.isclose(np.linalg.norm(record["embedding"]), 1)
        # Gán cho known: gửi yêu cầu HTTP POST qua client.post(...); dữ liệu/tham số: '/recognize', files={'image': ('single.png', single_bytes, 'image/png')}.
        known = client.post("/recognize", files={"image": ("single.png", single_bytes, "image/png")})
        # Kiểm chứng điều kiện known.status_code == 200 and known.json()['faces'][0]['person_id'] == 'fixture_smoke'; sai thì test dừng bằng AssertionError.
        assert known.status_code == 200 and known.json()["faces"][0]["person_id"] == "fixture_smoke"
        # Gán cho before: lấy mã băm dạng chữ hệ 16 qua hashlib.sha256(path.read_bytes()).hexdigest(...).
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        # Gán cho response: gửi yêu cầu HTTP POST qua client.post(...); dữ liệu/tham số: '/recognize', files={'image': ('group.jpg', group_bytes, 'image/jpeg')}.
        response = client.post("/recognize", files={"image": ("group.jpg", group_bytes, "image/jpeg")})
        # Kiểm chứng điều kiện response.status_code == 200; sai thì test dừng bằng AssertionError.
        assert response.status_code == 200
        # Kiểm chứng điều kiện response.json()['face_count'] == 5; sai thì test dừng bằng AssertionError.
        assert response.json()["face_count"] == 5
        # Gán cho direct: chạy pipeline nhận diện khuôn mặt trong ảnh bằng recognize_image(...); dữ liệu/tham số: group, runtime.detector, runtime.net.
        direct = recognize_image(group, runtime.detector, runtime.net, path, runtime.version)
        # Kiểm chứng điều kiện response.json()['faces'] == public_faces(direct); sai thì test dừng bằng AssertionError.
        assert response.json()["faces"] == public_faces(direct)
        # Kiểm chứng điều kiện any((row['status'] == 'unknown' for row in direct)); sai thì test dừng bằng AssertionError.
        assert any(row["status"] == "unknown" for row in direct)
        # Kiểm chứng điều kiện before == hashlib.sha256(path.read_bytes()).hexdigest(); sai thì test dừng bằng AssertionError.
        assert before == hashlib.sha256(path.read_bytes()).hexdigest()
        # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là invalid; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with pytest.raises(ApplicationError) as invalid:
            # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'group_rejected', [group].
            service.enroll("group_rejected", [group])
        # Kiểm chứng điều kiện invalid.value.status == 422 and len(service.persons()) == 1; sai thì test dừng bằng AssertionError.
        assert invalid.value.status == 422 and len(service.persons()) == 1
        # Kiểm chứng điều kiện service.recognize(np.zeros((240, 320, 3), np.uint8))['face_count'] == 0; sai thì test dừng bằng AssertionError.
        assert service.recognize(np.zeros((240, 320, 3), np.uint8))["face_count"] == 0
        # Gán cho state: từ điển {} (ánh xạ khóa → giá trị).
        state = {}
        # Lặp qua range(3); mỗi lượt gán phần tử cho i.
        for i in range(3):
            # Tách kết quả theo thứ tự và gán cho (annotated, result, state): xử lý một frame và cập nhật thống kê phiên webcam bằng process_frame(...); dữ liệu/tham số: service, group, state.
            annotated, result, state = process_frame(service, group, state)
            # Kiểm chứng điều kiện result['frames_processed'] == i + 1 and result['face_count'] > 0; sai thì test dừng bằng AssertionError.
            assert result["frames_processed"] == i + 1 and result["face_count"] > 0
            # Kiểm chứng điều kiện annotated.shape[1] <= 640; sai thì test dừng bằng AssertionError.
            assert annotated.shape[1] <= 640
        # Kiểm chứng điều kiện client.delete('/persons/fixture_smoke').status_code == 200; sai thì test dừng bằng AssertionError.
        assert client.delete("/persons/fixture_smoke").status_code == 200
        # Kiểm chứng điều kiện db.list_people(path) == []; sai thì test dừng bằng AssertionError.
        assert db.list_people(path) == []


# Kiểm tra: Đo được từng công đoạn với model thật và khôi phục detector sau phép đo.
# Định nghĩa hàm test_real_benchmark_stage_instrumentation(runtime, tmp_path).
def test_real_benchmark_stage_instrumentation(runtime, tmp_path):
    # Lấy run_case từ application.benchmark.
    from application.benchmark import run_case
    # Gán cho path: ghép đường dẫn bằng toán tử / của pathlib trong tmp_path / 'benchmark.sqlite'. Ở đây / nối thư mục với tên file, không phải phép chia số.
    path = tmp_path / "benchmark.sqlite"
    # Khởi tạo kho SQLite theo cấu trúc FaceNet bằng initialize_enrollment_registry(...); dữ liệu/tham số: path.
    initialize_enrollment_registry(path)
    # Gán cho service: tạo dịch vụ dùng chung cho nhận diện và quản lý hồ sơ bằng ApplicationService(...); dữ liệu/tham số: Settings(registry=path) (tạo cấu hình bằng Settings(...)), lambda settings: runtime.
    service = ApplicationService(Settings(registry=path), lambda settings: runtime)
    # Khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua service.start(...).
    service.start()
    # Gán cho fixture: ghép đường dẫn bằng toán tử / của pathlib trong ROOT / 'integration/tests/fixtures/mtcnn_single.png'. Ở đây / nối thư mục với tên file, không phải phép chia số.
    fixture = ROOT / "integration/tests/fixtures/mtcnn_single.png"
    # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'benchmark', [decode_image(fixture.read_bytes(), service.settings)].
    service.enroll("benchmark", [decode_image(fixture.read_bytes(), service.settings)])
    # Gán cho detector: runtime.detector (đọc thuộc tính).
    detector = runtime.detector
    # Gán cho result: đo lặp lại thời gian xử lý một ảnh bằng run_case(...); dữ liệu/tham số: service, fixture, repetitions=2, warmup=1.
    result = run_case(service, fixture, repetitions=2, warmup=1)
    # Kiểm chứng điều kiện runtime.detector is detector; sai thì test dừng bằng AssertionError.
    assert runtime.detector is detector
    # Kiểm chứng điều kiện result['detected_faces_each_run'] == [1, 1]; sai thì test dừng bằng AssertionError.
    assert result["detected_faces_each_run"] == [1, 1]
    # Kiểm chứng điều kiện all((item['status'] == 'measured' for item in result['latency'].values())); sai thì test dừng bằng AssertionError.
    assert all(item["status"] == "measured" for item in result["latency"].values())
    # Kiểm chứng điều kiện result['fps'] > 0; sai thì test dừng bằng AssertionError.
    assert result["fps"] > 0
