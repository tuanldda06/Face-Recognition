# Kiểm tra model thật lưu hồ sơ 4 ảnh và vẫn đọc được hồ sơ cũ có 5 ảnh.
# Lấy Path từ pathlib để biểu diễn đường dẫn và thao tác file bằng đối tượng Path.
from pathlib import Path
# Nạp tempfile để tạo file/thư mục tạm có kiểm soát vòng đời.
import tempfile

# Nạp numpy dưới tên ngắn np để tính toán mảng ảnh, vector và ma trận.
import numpy as np

# Lấy enroll_person_from_images, initialize_enrollment_registry từ FaceNet.
from FaceNet import enroll_person_from_images, initialize_enrollment_registry
# Lấy database với bí danh db từ FaceNet.
from FaceNet import database as db
# Lấy ROOT, Settings từ application.config.
from application.config import ROOT, Settings
# Lấy decode_image từ application.images.
from application.images import decode_image
# Lấy ApplicationService, load_runtime từ application.service.
from application.service import ApplicationService, load_runtime
# Lấy RecordingNet từ application.webcam_enrollment.
from application.webcam_enrollment import RecordingNet
# Lấy commit_identity, read_names từ application.webcam_registry.
from application.webcam_registry import commit_identity, read_names
# Lấy validate_embedding_consistency từ integration.webcam_identity.
from integration.webcam_identity import validate_embedding_consistency


# Kiểm tra: Vector từ 4 ảnh được lưu thật; hồ sơ 5 ảnh cũ vẫn dùng được.
# Định nghĩa hàm test_real_four_image_staging_commit_and_legacy_recognition(tmp_path).
def test_real_four_image_staging_commit_and_legacy_recognition(tmp_path):
    # Gán cho settings: tạo cấu hình bằng Settings(...); dữ liệu/tham số: registry=tmp_path / 'real_registry.sqlite'.
    settings = Settings(registry=tmp_path / "real_registry.sqlite")
    # Khởi tạo kho SQLite theo cấu trúc FaceNet bằng initialize_enrollment_registry(...); dữ liệu/tham số: settings.registry.
    initialize_enrollment_registry(settings.registry)
    # Gán cho runtime: nạp detector và FaceNet theo cấu hình bằng load_runtime(...); dữ liệu/tham số: settings.
    runtime = load_runtime(settings)
    # Gán cho service: tạo dịch vụ dùng chung cho nhận diện và quản lý hồ sơ bằng ApplicationService(...); dữ liệu/tham số: settings, lambda _: runtime.
    service = ApplicationService(settings, lambda _: runtime)
    # Khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua service.start(...).
    service.start()
    # Gán cho image: giải mã byte thành ảnh RGB đã kiểm tra bằng decode_image(...); dữ liệu/tham số: (ROOT / 'integration/tests/fixtures/mtcnn_single.png').read_bytes() (đọc toàn bộ file dưới dạng byte qua (ROOT / 'integration/tests/fixtures/mtcnn_single.png').read_bytes(...)), settings.
    image = decode_image((ROOT / "integration/tests/fixtures/mtcnn_single.png").read_bytes(), settings)
    # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'existing_photo_identity', [image.copy() for _ in range(5)].
    service.enroll("existing_photo_identity", [image.copy() for _ in range(5)])
    # Gán cho old: gọi db.get_person(...) với service.path, 'existing_photo_identity', runtime.version; dữ liệu/tham số: service.path, 'existing_photo_identity', runtime.version.
    old = db.get_person(service.path, "existing_photo_identity", runtime.version)
    # Mở khối with để tạo thư mục tạm và tự xóa khi ra khỏi khối with bằng tempfile.TemporaryDirectory(...); dữ liệu/tham số: prefix='real_four_image_'; gọi tài nguyên là folder; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with tempfile.TemporaryDirectory(prefix="real_four_image_") as folder:
        # Gán cho staging: ghép đường dẫn bằng toán tử / của pathlib trong Path(folder) / 'stage.sqlite'. Ở đây / nối thư mục với tên file, không phải phép chia số.
        staging = Path(folder) / "stage.sqlite"
        # Khởi tạo kho SQLite theo cấu trúc FaceNet bằng initialize_enrollment_registry(...); dữ liệu/tham số: staging.
        initialize_enrollment_registry(staging)
        # Gán cho observer: bọc mạng để ghi lại các đầu ra đã suy luận bằng RecordingNet(...); dữ liệu/tham số: runtime.net.
        observer = RecordingNet(runtime.net)
        # Gán cho result: tạo vector đại diện từ các ảnh và ghi hồ sơ bằng enroll_person_from_images(...); dữ liệu/tham số: 'four_image_identity', [image.copy() for _ in range(4)], runtime.detector.
        result = enroll_person_from_images("four_image_identity", [image.copy() for _ in range(4)],
                                           # Tiếp phần biểu thức nhiều dòng: runtime.detector (đọc thuộc tính).
                                           runtime.detector, observer, staging, runtime.version)
        # Gán cho normalized: kiểm tra các vector ảnh có nhất quán danh tính bằng validate_embedding_consistency(...); dữ liệu/tham số: [output[0] for output in observer.outputs], expected_count=4.
        normalized = validate_embedding_consistency([output[0] for output in observer.outputs], expected_count=4)
        # Kiểm chứng điều kiện len(normalized) == 4; sai thì test dừng bằng AssertionError.
        assert len(normalized) == 4
        # Ghi thật vào kho bằng đúng vector đại diện do pipeline gốc tạo ra.
        # Mở khối with để lấy khóa truy cập model và kho hồ sơ qua service.guard(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with service.guard():
            # Lưu vector và tên trong cùng giao dịch bằng commit_identity(...); dữ liệu/tham số: service.path, 'four_image_identity', 'Model smoke', num_images=result['num_images'].
            commit_identity(service.path, "four_image_identity", "Model smoke", result["embedding"],
                            # Truyền num_images (số ảnh tạo nên hồ sơ) = result['num_images'].
                            runtime.version, num_images=result["num_images"])
    # Kiểm chứng điều kiện db.get_person(service.path, 'four_image_identity', runtime.version)['num_images'] == 4; sai thì test dừng bằng AssertionError.
    assert db.get_person(service.path, "four_image_identity", runtime.version)["num_images"] == 4
    # Gán cho current: gọi db.get_person(...) với service.path, 'existing_photo_identity', runtime.version; dữ liệu/tham số: service.path, 'existing_photo_identity', runtime.version.
    current = db.get_person(service.path, "existing_photo_identity", runtime.version)
    # Kiểm chứng điều kiện np.array_equal(current['embedding'], old['embedding']); sai thì test dừng bằng AssertionError.
    assert np.array_equal(current["embedding"], old["embedding"])
    # Kiểm chứng điều kiện current['num_images'] == old['num_images'] and current['model_version'] == old['model_version']; sai thì test dừng bằng AssertionError.
    assert current["num_images"] == old["num_images"] and current["model_version"] == old["model_version"]
    # Kiểm chứng điều kiện read_names(service.path) == {'four_image_identity': 'Model smoke'}; sai thì test dừng bằng AssertionError.
    assert read_names(service.path) == {"four_image_identity": "Model smoke"}
    # Kiểm tra riêng từng ID để tránh hai hồ sơ có điểm khớp bằng nhau hoàn toàn.
    # Gửi lệnh xóa theo ID hoặc URL qua service.delete(...); dữ liệu/tham số: 'existing_photo_identity'.
    service.delete("existing_photo_identity")
    # Kiểm chứng điều kiện service.recognize(image)['faces'][0]['person_id'] == 'four_image_identity'; sai thì test dừng bằng AssertionError.
    assert service.recognize(image)["faces"][0]["person_id"] == "four_image_identity"
    # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'existing_photo_identity', [image.copy() for _ in range(5)].
    service.enroll("existing_photo_identity", [image.copy() for _ in range(5)])
    # Gửi lệnh xóa theo ID hoặc URL qua service.delete(...); dữ liệu/tham số: 'four_image_identity'.
    service.delete("four_image_identity")
    # Kiểm chứng điều kiện service.recognize(image)['faces'][0]['person_id'] == 'existing_photo_identity'; sai thì test dừng bằng AssertionError.
    assert service.recognize(image)["faces"][0]["person_id"] == "existing_photo_identity"
