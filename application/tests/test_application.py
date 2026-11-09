# Kiểm thử tầng ứng dụng với model giả và SQLite thật để kiểm tra luồng xử lý.
# Lấy ThreadPoolExecutor từ concurrent.futures để chạy nhiều công việc qua các luồng và thu kết quả.
from concurrent.futures import ThreadPoolExecutor
# Lấy closing từ contextlib để quản lý tài nguyên bằng with, yield và thao tác dọn tự động.
from contextlib import closing
# Lấy replace từ dataclasses để tạo lớp dữ liệu và sao chép/chuyển đổi dữ liệu của lớp.
from dataclasses import replace
# Nạp hashlib để tính mã băm để kiểm tra nội dung file/ảnh/vector.
import hashlib
# Lấy BytesIO từ io để đọc/ghi byte ảnh trong bộ nhớ bằng BytesIO.
from io import BytesIO
# Lấy Path từ pathlib để biểu diễn đường dẫn và thao tác file bằng đối tượng Path.
from pathlib import Path
# Nạp sqlite3 để đọc, ghi và quản lý giao dịch SQLite.
import sqlite3
# Lấy sleep từ time để đo thời gian xử lý hoặc mô phỏng độ trễ.
from time import sleep
# Lấy SimpleNamespace từ types để tạo đối tượng dữ liệu đơn giản bằng SimpleNamespace.
from types import SimpleNamespace
# Lấy Mock từ unittest.mock để thay tạm hàm/model và theo dõi lời gọi trong kiểm thử.
from unittest.mock import Mock

# Nạp numpy dưới tên ngắn np để tính toán mảng ảnh, vector và ma trận.
import numpy as np
# Lấy Image từ PIL để đọc/ghi ảnh, xử lý hướng EXIF, chọn font và vẽ lên ảnh bằng Pillow.
from PIL import Image
# Nạp pytest để viết và chạy bài kiểm thử tự động.
import pytest
# Lấy TestClient từ fastapi.testclient để gọi thử API mà không cần mở cổng máy chủ thật.
from fastapi.testclient import TestClient

# Lấy build_model_version, initialize_enrollment_registry, recognize_image từ FaceNet.
from FaceNet import build_model_version, initialize_enrollment_registry, recognize_image
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
# Lấy ApplicationService, Runtime, public_faces từ application.service.
from application.service import ApplicationService, Runtime, public_faces
# Lấy process_frame từ application.webcam.
from application.webcam import process_frame


# Detector giả trả dữ liệu được điều khiển bởi ảnh test; không đo độ chính xác model thật.
# Định nghĩa lớp Detector; lớp gom dữ liệu và các hàm liên quan.
class Detector:
    # Chuỗi tài liệu (docstring) mô tả cách dùng; Python lưu nội dung này trong __doc__.
    """Synthetic metadata for transport/orchestration tests, never accuracy evidence."""
    # Tách kết quả theo thứ tự và gán cho (min_face_size, factor, thresholds, batch_size, device): bộ giá trị (20, 0.709, (0.6, 0.7, 0.7), 256, 'cpu') (tuple giữ thứ tự, không gán lại từng phần tử).
    min_face_size, factor, thresholds, batch_size, device = 20, .709, (.6, .7, .7), 256, "cpu"

    # Đọc số mặt từ pixel đầu tiên và dựng khung/điểm mốc giả theo số đó.
    # Định nghĩa hàm detect(self, image). self là đối tượng đang gọi phương thức.
    def detect(self, image):
        # Gán cho self.image: tạo bản sao để thay đổi không tác động dữ liệu gốc qua image.copy(...).
        self.image = image.copy()
        # Gán cho count: đổi sang số nguyên bằng int(...); dữ liệu/tham số: image[0, 0, 0].
        count = int(image[0, 0, 0])
        # Kết thúc hàm và trả tạo đối tượng đơn giản có các thuộc tính từ tham số đặt tên bằng SimpleNamespace(...); dữ liệu/tham số: boxes=np.array([[10 + i * 5, 10, 60 + i * 5, 90] for i in range(count)], np.float32).resha…, scores=np.full(count, 0.95, np.float32), landmarks=np.tile(np.array([[[20, 30], [40, 30], [30, 40], [22, 60], [38, 60]]], np.float32), ….
        return SimpleNamespace(boxes=np.array([[10 + i * 5, 10, 60 + i * 5, 90]
                                               # Phần for của biểu thức rút gọn: duyệt từng phần tử theo thứ tự và tạo kết quả.
                                               for i in range(count)], np.float32).reshape(count, 4),
                               # Truyền scores = np.full(count, 0.95, np.float32).
                               scores=np.full(count, .95, np.float32),
                               # Truyền landmarks = np.tile(np.array([[[20, 30], [40, 30], [30, 40], [22, 60], [38, 60]]], np.float32), (count, 1, 1)).
                               landmarks=np.tile(np.array([[[20, 30], [40, 30], [30, 40], [22, 60], [38, 60]]], np.float32), (count, 1, 1)))

    # Cắt/căn chỉnh các mặt về ảnh RGB có kích thước đầu vào của FaceNet.
    # Định nghĩa hàm extract_faces_rgb(self, image, result, size, align). self là đối tượng đang gọi phương thức.
    def extract_faces_rgb(self, image, result, size, align):
        # Kiểm chứng điều kiện size == 160 and align is True; sai thì test dừng bằng AssertionError.
        assert size == 160 and align is True
        # Kết thúc hàm và trả đổi hình dạng mảng nhưng giữ số phần tử qua np.array([np.full((160, 160, 3), int(image[0, 0, 1]) + i, np.uint8) for i in range(len(re…(...); dữ liệu/tham số: -1, 160, 160.
        return np.array([np.full((160, 160, 3), int(image[0, 0, 1]) + i, np.uint8)
                         # Phần for của biểu thức rút gọn: duyệt từng phần tử theo thứ tự và tạo kết quả.
                         for i in range(len(result.boxes))], np.uint8).reshape(-1, 160, 160, 3)


# Mạng giả tạo vector dự đoán được để kiểm tra nhận diện và lỗi xử lý.
# Định nghĩa lớp Net; lớp gom dữ liệu và các hàm liên quan.
class Net:
    # Khởi tạo đối tượng và giữ các phụ thuộc/trạng thái cần dùng trong những lần gọi sau.
    # Định nghĩa hàm __init__(self). self là đối tượng đang gọi phương thức.
    def __init__(self):
        # Gán cho self.fail: False (sai/tắt).
        self.fail = False
        # Gán cho self.pending: False (sai/tắt).
        self.pending = False
        # Gán cho self.calls: giá trị 0.
        self.calls = 0

    # Chuyển dữ liệu đầu vào cho mạng; giữ nguyên giao diện gọi của OpenCV DNN.
    # Định nghĩa hàm setInput(self, batch, name). self là đối tượng đang gọi phương thức.
    def setInput(self, batch, name):
        # Kiểm chứng điều kiện not self.pending; sai thì test dừng bằng AssertionError.
        assert not self.pending, "Concurrent setInput/forward"
        # Kiểm chứng điều kiện name == 'in:0' and batch.shape == (1, 3, 160, 160) and (batch.dtype == np.float32); sai thì test dừng bằng AssertionError.
        assert name == "in:0" and batch.shape == (1, 3, 160, 160) and batch.dtype == np.float32
        # Gán cho self.pending: True (đúng/bật).
        self.pending = True
        # Gán cho self.index: làm tròn số bằng round(...); dữ liệu/tham số: float(batch[0, 0, 0, 0]) * 128 + 127.5.
        self.index = round(float(batch[0, 0, 0, 0]) * 128 + 127.5)
        # Tạm dừng luồng theo số giây chỉ định bằng sleep(...); dữ liệu/tham số: 0.001.
        sleep(.001)

    # Tạo vector 512 chiều với một phần tử bằng 1; có thể bật cờ để giả lỗi model.
    # Định nghĩa hàm forward(self, name). self là đối tượng đang gọi phương thức.
    def forward(self, name):
        # Gán cho self.pending: False (sai/tắt).
        self.pending = False
        # Cập nhật self.calls: cộng thêm 1 và lưu lại vào chính nó.
        self.calls += 1
        # Chỉ chạy khối bên dưới khi self.fail cho kết quả đúng.
        if self.fail:
            # Phát sinh lỗi: gọi RuntimeError(...) với 'forced inference failure'; dữ liệu/tham số: 'forced inference failure'.
            raise RuntimeError("forced inference failure")
        # Kiểm chứng điều kiện name == 'out:0'; sai thì test dừng bằng AssertionError.
        assert name == "out:0"
        # Gán cho result: tạo mảng gồm các số 0 bằng np.zeros(...); dữ liệu/tham số: (1, 512), np.float32.
        result = np.zeros((1, 512), np.float32)
        # Gán cho result[0, self.index]: giá trị 1.
        result[0, self.index] = 1
        # Kết thúc hàm và trả result (kết quả xử lý).
        return result


# Tạo ảnh tổng hợp; giá trị pixel mã hóa số mặt và chỉ số danh tính cho model giả.
# Định nghĩa hàm picture(count, index). Mặc định: count=1, index=5.
def picture(count=1, index=5):
    # Gán cho result: tạo mảng với mọi phần tử bằng giá trị chỉ định bằng np.full(...); dữ liệu/tham số: (240, 320, 3), (count, index, 150), np.uint8.
    result = np.full((240, 320, 3), (count, index, 150), np.uint8)
    # Kết thúc hàm và trả result (kết quả xử lý).
    return result


# Mã hóa ảnh test thành byte để gửi như một file upload.
# Định nghĩa hàm encoded(image, format). Mặc định: image=None, format='PNG'.
def encoded(image=None, format="PNG"):
    # Gán cho data: tạo luồng byte trong bộ nhớ thay vì file trên ổ đĩa bằng BytesIO(...).
    data = BytesIO()
    # Mã hóa và lưu ảnh qua Image.fromarray(picture() if image is None else image).save(...); dữ liệu/tham số: data, format=format.
    Image.fromarray(picture() if image is None else image).save(data, format=format)
    # Kết thúc hàm và trả lấy toàn bộ byte đang chứa trong bộ đệm qua data.getvalue(...).
    return data.getvalue()


# Fixture tạo kho SQLite riêng trong tmp_path và nạp model giả cho mỗi bài test.
# Decorator fixture: pytest tự tạo dữ liệu/phụ thuộc này khi bài test yêu cầu tên hàm.
@pytest.fixture
# Định nghĩa hàm service(tmp_path).
def service(tmp_path):
    # Gán cho settings: tạo cấu hình bằng Settings(...); dữ liệu/tham số: registry=tmp_path / 'registry.sqlite'.
    settings = Settings(registry=tmp_path / "registry.sqlite")
    # Khởi tạo kho SQLite theo cấu trúc FaceNet bằng initialize_enrollment_registry(...); dữ liệu/tham số: settings.registry.
    initialize_enrollment_registry(settings.registry)
    # Tách kết quả theo thứ tự và gán cho (detector, net): bộ giá trị (Detector(), Net()) (tuple giữ thứ tự, không gán lại từng phần tử).
    detector, net = Detector(), Net()
    # Gán cho runtime: gom detector, mạng và phiên bản model bằng Runtime(...); dữ liệu/tham số: detector, net, build_model_version(detector) (tạo thông tin phiên bản model và môi trường chạy bằng build_model_version(...)).
    runtime = Runtime(detector, net, build_model_version(detector))
    # Gán cho loader: tạo đối tượng giả có thể ghi lại số lần gọi bằng Mock(...); dữ liệu/tham số: return_value=runtime.
    loader = Mock(return_value=runtime)
    # Gán cho service: tạo dịch vụ dùng chung cho nhận diện và quản lý hồ sơ bằng ApplicationService(...); dữ liệu/tham số: settings, runtime_loader=loader.
    service = ApplicationService(settings, runtime_loader=loader)
    # Khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua service.start(...).
    service.start()
    # Gán cho service.test_loader: loader.
    service.test_loader = loader
    # Kết thúc hàm và trả service (dịch vụ điều phối nhận diện).
    return service


# Fixture mở TestClient để gọi API trong tiến trình, kèm vòng đời khởi động ứng dụng.
# Decorator fixture: pytest tự tạo dữ liệu/phụ thuộc này khi bài test yêu cầu tên hàm.
@pytest.fixture
# Định nghĩa hàm client(service).
def client(service):
    # Mở khối with để tạo trình gọi API nội bộ cho kiểm thử bằng TestClient(...); dữ liệu/tham số: create_app(service) (tạo API và tùy chọn gắn giao diện bằng create_app(...)); gọi tài nguyên là client; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with TestClient(create_app(service)) as client:
        # Cung cấp client cho bài test; sau khi test xong, pytest quay lại đây để thoát with và đóng client.
        yield client


# Gửi JSON bắt đầu đăng ký và giữ nguyên phản hồi để bài test kiểm tra.
# Định nghĩa hàm start_webcam_enrollment(client, person). Mặc định: person='alice'.
def start_webcam_enrollment(client, person="alice"):
    # Kết thúc hàm và trả gửi yêu cầu HTTP POST qua client.post(...); dữ liệu/tham số: '/enrollment-sessions', json={'person_id': person, 'display_name': 'Alice'}.
    return client.post("/enrollment-sessions", json={"person_id": person, "display_name": "Alice"})


# Gửi ảnh qua multipart giống thao tác upload trong trình duyệt.
# Định nghĩa hàm post_recognize(client, image). Mặc định: image=None.
def post_recognize(client, image=None):
    # Kết thúc hàm và trả gửi yêu cầu HTTP POST qua client.post(...); dữ liệu/tham số: '/recognize', files={'image': ('face.png', encoded(image), 'image/png')}.
    return client.post("/recognize", files={"image": ("face.png", encoded(image), "image/png")})


# Kiểm tra: Khởi động nhiều lần chỉ nạp model một lần và giữ hồ sơ đang có.
# Định nghĩa hàm test_health_startup_loads_once_no_registry_reset(client, service).
def test_health_startup_loads_once_no_registry_reset(client, service):
    # Kiểm chứng điều kiện client.get('/health').status_code == 200; sai thì test dừng bằng AssertionError.
    assert client.get("/health").status_code == 200
    # Kiểm chứng điều kiện client.get('/health').json()['registry'] == 'empty'; sai thì test dừng bằng AssertionError.
    assert client.get("/health").json()["registry"] == "empty"
    # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'alice', [picture()].
    service.enroll("alice", [picture()])
    # Khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua service.start(...).
    service.start()
    # Xác nhận hàm giả chỉ được gọi đúng một lần qua service.test_loader.assert_called_once(...).
    service.test_loader.assert_called_once()
    # Kiểm chứng điều kiện len(db.list_people(service.path)) == 1; sai thì test dừng bằng AssertionError.
    assert len(db.list_people(service.path)) == 1
    # Kiểm chứng điều kiện client.get('/docs').status_code == 200; sai thì test dừng bằng AssertionError.
    assert client.get("/docs").status_code == 200


# Kiểm tra: Đăng ký nhiều ảnh có ghi SQLite thật nhưng phản hồi không lộ vector.
# Định nghĩa hàm test_internal_enroll_multiple_images_actual_sqlite_and_no_vectors(client, service).
def test_internal_enroll_multiple_images_actual_sqlite_and_no_vectors(client, service):
    # Gán cho result: đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'alice', [picture(), picture()].
    result = service.enroll("alice", [picture(), picture()])
    # Kiểm chứng điều kiện result == {'person_id': 'alice', 'num_images': 2, 'status': 'enrolled'}; sai thì test dừng bằng AssertionError.
    assert result == {"person_id": "alice", "num_images": 2, "status": "enrolled"}
    # Gán cho record: gọi db.get_person(...) với service.path, 'alice', service.runtime.version; dữ liệu/tham số: service.path, 'alice', service.runtime.version.
    record = db.get_person(service.path, "alice", service.runtime.version)
    # Kiểm chứng điều kiện record['embedding'].shape == (512,) and record['num_images'] == 2; sai thì test dừng bằng AssertionError.
    assert record["embedding"].shape == (512,) and record["num_images"] == 2
    # Kiểm chứng điều kiện record['embedding'][5] == 1; sai thì test dừng bằng AssertionError.
    assert record["embedding"][5] == 1
    # Kiểm chứng điều kiện client.get('/persons').json() == {'persons': [{'person_id': 'alice', 'num_images': 2}]}; sai thì test dừng bằng AssertionError.
    assert client.get("/persons").json() == {"persons": [{"person_id": "alice", "num_images": 2}]}


# Kiểm tra: Đường đăng ký bằng ảnh cũ trả 404 và không làm đổi dữ liệu.
# Định nghĩa hàm test_photo_enrollment_route_removed(client, service).
def test_photo_enrollment_route_removed(client, service):
    # Gán cho before: đọc toàn bộ file dưới dạng byte qua service.path.read_bytes(...).
    before = service.path.read_bytes()
    # Gán cho result: gửi yêu cầu HTTP POST qua client.post(...); dữ liệu/tham số: '/enroll', data={'person_id': 'alice'}, files={'images': ('face.png', encoded(), 'image/png')}.
    result = client.post("/enroll", data={"person_id": "alice"},
                         # Truyền files = {'images': ('face.png', encoded(), 'image/png')}.
                         files={"images": ("face.png", encoded(), "image/png")})
    # Kiểm chứng điều kiện result.status_code == 404; sai thì test dừng bằng AssertionError.
    assert result.status_code == 404
    # Kiểm chứng điều kiện '/enroll' not in client.get('/openapi.json').json()['paths']; sai thì test dừng bằng AssertionError.
    assert "/enroll" not in client.get("/openapi.json").json()["paths"]
    # Kiểm chứng điều kiện service.path.read_bytes() == before; sai thì test dừng bằng AssertionError.
    assert service.path.read_bytes() == before


# Kiểm tra: ID trùng bị từ chối và hồ sơ cũ không bị ghi đè.
# Định nghĩa hàm test_duplicate_does_not_overwrite(client, service).
def test_duplicate_does_not_overwrite(client, service):
    # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'alice', [picture()].
    service.enroll("alice", [picture()])
    # Gán cho before: đọc toàn bộ file dưới dạng byte qua service.path.read_bytes(...).
    before = service.path.read_bytes()
    # Kiểm chứng điều kiện start_webcam_enrollment(client).status_code == 409; sai thì test dừng bằng AssertionError.
    assert start_webcam_enrollment(client).status_code == 409
    # Kiểm chứng điều kiện before == service.path.read_bytes(); sai thì test dừng bằng AssertionError.
    assert before == service.path.read_bytes()


# Kiểm tra: Các ID không hợp lệ bị từ chối trước khi tạo hồ sơ.
# Decorator parametrize: chạy cùng một bài test với nhiều bộ dữ liệu liệt kê bên dưới.
@pytest.mark.parametrize("person", ["Unknown", "", " alice", "alice ", "a/b", "a\\b", "x" * 101])
# Định nghĩa hàm test_invalid_ids_rejected(client, person).
def test_invalid_ids_rejected(client, person):
    # Kiểm chứng điều kiện start_webcam_enrollment(client, person).status_code == 422; sai thì test dừng bằng AssertionError.
    assert start_webcam_enrollment(client, person).status_code == 422
    # Kiểm chứng điều kiện client.get('/persons').json()['persons'] == []; sai thì test dừng bằng AssertionError.
    assert client.get("/persons").json()["persons"] == []


# Kiểm tra: File rỗng, giả ảnh hoặc bị cắt hỏng đều trả lỗi giải mã.
# Decorator parametrize: chạy cùng một bài test với nhiều bộ dữ liệu liệt kê bên dưới.
@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\ntruncated"])
# Định nghĩa hàm test_invalid_upload(client, data).
def test_invalid_upload(client, data):
    # Gán cho response: gửi yêu cầu HTTP POST qua client.post(...); dữ liệu/tham số: '/recognize', files={'image': ('valid.jpg', data, 'image/jpeg')}.
    response = client.post("/recognize", files={"image": ("valid.jpg", data, "image/jpeg")})
    # Kiểm chứng điều kiện response.status_code == 422; sai thì test dừng bằng AssertionError.
    assert response.status_code == 422
    # Kiểm chứng điều kiện response.json()['error']['code'] == 'invalid_image'; sai thì test dừng bằng AssertionError.
    assert response.json()["error"]["code"] == "invalid_image"


# Kiểm tra: Ảnh đăng ký sai số mặt khiến cả lượt đăng ký không được ghi dở dang.
# Decorator parametrize: chạy cùng một bài test với nhiều bộ dữ liệu liệt kê bên dưới.
@pytest.mark.parametrize("count", [0, 2])
# Định nghĩa hàm test_enrollment_face_count_no_partial_write(client, service, count).
def test_enrollment_face_count_no_partial_write(client, service, count):
    # Gán cho before: đọc toàn bộ file dưới dạng byte qua service.path.read_bytes(...).
    before = service.path.read_bytes()
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'alice', [picture(), picture(count)].
        service.enroll("alice", [picture(), picture(count)])
    # Kiểm chứng điều kiện caught.value.status == 422; sai thì test dừng bằng AssertionError.
    assert caught.value.status == 422
    # Kiểm chứng điều kiện caught.value.code == 'enrollment_face_count'; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "enrollment_face_count"
    # Kiểm chứng điều kiện before == service.path.read_bytes(); sai thì test dừng bằng AssertionError.
    assert before == service.path.read_bytes()


# Kiểm tra: Kết quả giữ thứ tự mặt, phân biệt đã biết/chưa biết và khớp pipeline gốc.
# Định nghĩa hàm test_no_face_known_unknown_order_and_core_consistency(client, service).
def test_no_face_known_unknown_order_and_core_consistency(client, service):
    # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'alice', [picture()].
    service.enroll("alice", [picture()])
    # Kiểm chứng điều kiện post_recognize(client, picture(0)).json()['faces'] == []; sai thì test dừng bằng AssertionError.
    assert post_recognize(client, picture(0)).json()["faces"] == []
    # Gán cho image: tạo ảnh giả điều khiển kết quả model trong test bằng picture(...); dữ liệu/tham số: 3.
    image = picture(3)
    # Gán cho response: gọi post_recognize(...) với client, image; dữ liệu/tham số: client, image.
    response = post_recognize(client, image)
    # Kiểm chứng điều kiện response.status_code == 200; sai thì test dừng bằng AssertionError.
    assert response.status_code == 200
    # Gán cho faces: lấy phần tử/lát cắt response.json()['faces'] từ response.json().
    faces = response.json()["faces"]
    # Kiểm chứng điều kiện [f['person_id'] for f in faces] == ['alice', None, None]; sai thì test dừng bằng AssertionError.
    assert [f["person_id"] for f in faces] == ["alice", None, None]
    # Kiểm chứng điều kiện [f['face_index'] for f in faces] == [0, 1, 2]; sai thì test dừng bằng AssertionError.
    assert [f["face_index"] for f in faces] == [0, 1, 2]
    # Gán cho rt: service.runtime (đọc thuộc tính).
    rt = service.runtime
    # Gán cho expected: chạy pipeline nhận diện khuôn mặt trong ảnh bằng recognize_image(...); dữ liệu/tham số: image, rt.detector, rt.net.
    expected = recognize_image(image, rt.detector, rt.net, service.path, rt.version)
    # Kiểm chứng điều kiện faces == public_faces(expected); sai thì test dừng bằng AssertionError.
    assert faces == public_faces(expected)
    # Kiểm chứng điều kiện all((f['threshold'] == 0.665513450766159 for f in faces)); sai thì test dừng bằng AssertionError.
    assert all(f["threshold"] == 0.665513450766159 for f in faces)
    # Kiểm chứng điều kiện 'best_match_id' not in response.text and 'embedding' not in response.text; sai thì test dừng bằng AssertionError.
    assert "best_match_id" not in response.text and "embedding" not in response.text


# Kiểm tra: Kho rỗng trả unknown và điểm None thay vì tự tạo điểm so khớp.
# Định nghĩa hàm test_empty_registry_unknown_without_fake_score(client).
def test_empty_registry_unknown_without_fake_score(client):
    # Gán cho result: lấy phần tử/lát cắt post_recognize(client).json()['faces'][0] từ post_recognize(client).json()['faces'].
    result = post_recognize(client).json()["faces"][0]
    # Kiểm chứng điều kiện result['status'] == 'unknown' and result['person_id'] is None and (result['similarity'] is None); sai thì test dừng bằng AssertionError.
    assert result["status"] == "unknown" and result["person_id"] is None and result["similarity"] is None


# Kiểm tra: Kho chưa có phải báo lỗi, không tự tạo file trong thao tác nhận diện.
# Định nghĩa hàm test_missing_registry_not_created(tmp_path).
def test_missing_registry_not_created(tmp_path):
    # Gán cho path: ghép đường dẫn bằng toán tử / của pathlib trong tmp_path / 'never_created.sqlite'. Ở đây / nối thư mục với tên file, không phải phép chia số.
    path = tmp_path / "never_created.sqlite"
    # Gán cho detector: gọi Detector(...).
    detector = Detector()
    # Gán cho service: tạo dịch vụ dùng chung cho nhận diện và quản lý hồ sơ bằng ApplicationService(...); dữ liệu/tham số: Settings(registry=path) (tạo cấu hình bằng Settings(...)), lambda s: Runtime(detector, Net(), build_model_version(detector)).
    service = ApplicationService(Settings(registry=path), lambda s: Runtime(detector, Net(), build_model_version(detector)))
    # Mở khối with để tạo trình gọi API nội bộ cho kiểm thử bằng TestClient(...); dữ liệu/tham số: create_app(service) (tạo API và tùy chọn gắn giao diện bằng create_app(...)); gọi tài nguyên là client; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with TestClient(create_app(service)) as client:
        # Kiểm chứng điều kiện client.get('/health').status_code == 503; sai thì test dừng bằng AssertionError.
        assert client.get("/health").status_code == 503
        # Kiểm chứng điều kiện post_recognize(client).status_code == 503; sai thì test dừng bằng AssertionError.
        assert post_recognize(client).status_code == 503
        # Kiểm chứng điều kiện start_webcam_enrollment(client).status_code == 503; sai thì test dừng bằng AssertionError.
        assert start_webcam_enrollment(client).status_code == 503
    # Kiểm chứng điều kiện not path.exists(); sai thì test dừng bằng AssertionError.
    assert not path.exists()


# Kiểm tra: Xóa hồ sơ có thật thành công; xóa lại trả 404.
# Định nghĩa hàm test_delete_and_missing_id(client, service).
def test_delete_and_missing_id(client, service):
    # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'alice', [picture()].
    service.enroll("alice", [picture()])
    # Kiểm chứng điều kiện client.delete('/persons/alice').status_code == 200; sai thì test dừng bằng AssertionError.
    assert client.delete("/persons/alice").status_code == 200
    # Kiểm chứng điều kiện client.delete('/persons/alice').status_code == 404; sai thì test dừng bằng AssertionError.
    assert client.delete("/persons/alice").status_code == 404
    # Kiểm chứng điều kiện db.list_people(service.path) == []; sai thì test dừng bằng AssertionError.
    assert db.list_people(service.path) == []
    # Kiểm chứng điều kiện post_recognize(client).json()['faces'][0]['similarity'] is None; sai thì test dừng bằng AssertionError.
    assert post_recognize(client).json()["faces"][0]["similarity"] is None


# Kiểm tra: Lỗi chạy model trả lỗi hệ thống và không tạo hồ sơ dở dang.
# Định nghĩa hàm test_inference_failure_not_unknown_or_partial_enrollment(client, service).
def test_inference_failure_not_unknown_or_partial_enrollment(client, service):
    # Gán cho service.runtime.net.fail: True (đúng/bật).
    service.runtime.net.fail = True
    # Gán cho response: gọi post_recognize(...) với client; dữ liệu/tham số: client.
    response = post_recognize(client)
    # Kiểm chứng điều kiện response.status_code == 500; sai thì test dừng bằng AssertionError.
    assert response.status_code == 500
    # Kiểm chứng điều kiện response.json()['error']['code'] == 'pipeline_failed'; sai thì test dừng bằng AssertionError.
    assert response.json()["error"]["code"] == "pipeline_failed"
    # Kiểm chứng điều kiện 'faces' not in response.json(); sai thì test dừng bằng AssertionError.
    assert "faces" not in response.json()
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'alice', [picture()].
        service.enroll("alice", [picture()])
    # Kiểm chứng điều kiện caught.value.code == 'pipeline_failed'; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "pipeline_failed"
    # Kiểm chứng điều kiện db.list_people(service.path) == []; sai thì test dừng bằng AssertionError.
    assert db.list_people(service.path) == []


# Kiểm tra: API health vẫn hoạt động để báo lỗi khi model không nạp được.
# Định nghĩa hàm test_missing_model_health_stays_available(tmp_path).
def test_missing_model_health_stays_available(tmp_path):
    # Giả lập lỗi nạp model để kiểm tra ứng dụng báo đúng tình trạng không sẵn sàng.
    # Định nghĩa hàm loader(settings).
    def loader(settings):
        # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'model_missing', 'missing', 503.
        raise ApplicationError("model_missing", "missing", 503)
    # Gán cho service: tạo dịch vụ dùng chung cho nhận diện và quản lý hồ sơ bằng ApplicationService(...); dữ liệu/tham số: Settings(registry=tmp_path / 'missing.sqlite') (tạo cấu hình bằng Settings(...)), loader.
    service = ApplicationService(Settings(registry=tmp_path / "missing.sqlite"), loader)
    # Mở khối with để tạo trình gọi API nội bộ cho kiểm thử bằng TestClient(...); dữ liệu/tham số: create_app(service) (tạo API và tùy chọn gắn giao diện bằng create_app(...)); gọi tài nguyên là client; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with TestClient(create_app(service)) as client:
        # Gán cho response: đọc giá trị theo khóa, dùng mặc định nếu khóa vắng mặt qua client.get(...); dữ liệu/tham số: '/health'.
        response = client.get("/health")
        # Kiểm chứng điều kiện response.status_code == 503; sai thì test dừng bằng AssertionError.
        assert response.status_code == 503
        # Kiểm chứng điều kiện response.json()['models'] == 'unavailable'; sai thì test dừng bằng AssertionError.
        assert response.json()["models"] == "unavailable"
        # Kiểm chứng điều kiện post_recognize(client).json()['error']['code'] == 'model_missing'; sai thì test dừng bằng AssertionError.
        assert post_recognize(client).json()["error"]["code"] == "model_missing"


# Kiểm tra: Con trỏ LFS và file sai checksum có mã lỗi riêng.
# Decorator parametrize: chạy cùng một bài test với nhiều bộ dữ liệu liệt kê bên dưới.
@pytest.mark.parametrize("content,code", [(b"version https://git-lfs.github.com/spec/v1\n", "model_lfs_pointer"), (b"wrong-model", "model_checksum")])
# Định nghĩa hàm test_model_asset_gate(tmp_path, content, code).
def test_model_asset_gate(tmp_path, content, code):
    # Gán cho path: ghép đường dẫn bằng toán tử / của pathlib trong tmp_path / 'model.onnx'. Ở đây / nối thư mục với tên file, không phải phép chia số.
    path = tmp_path / "model.onnx"
    # Ghi byte xuống file qua path.write_bytes(...); dữ liệu/tham số: content.
    path.write_bytes(content)
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Xác minh file model bằng dung lượng và SHA-256 bằng verify_model(...); dữ liệu/tham số: path.
        verify_model(path)
    # Kiểm chứng điều kiện caught.value.code == code; sai thì test dừng bằng AssertionError.
    assert caught.value.code == code


# Kiểm tra: Kho sai phiên bản bị từ chối và giữ nguyên để tránh sửa dữ liệu ngầm.
# Định nghĩa hàm test_corrupt_registry_and_version_not_repaired(client, service).
def test_corrupt_registry_and_version_not_repaired(client, service):
    # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'alice', [picture()].
    service.enroll("alice", [picture()])
    # Mở khối with để bảo đảm tài nguyên được close khi kết thúc khối with bằng closing(...); dữ liệu/tham số: db.connect_database(service.path) (gọi db.connect_database(...) với service.path); gọi tài nguyên là connection; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with closing(db.connect_database(service.path)) as connection:
        # Mở khối with để connection (kết nối SQLite); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with connection:
            # Thực hiện câu lệnh SQL và truyền các tham số đi kèm qua connection.execute(...); dữ liệu/tham số: "UPDATE FaceDatabase SET model_version='wrong'".
            connection.execute("UPDATE FaceDatabase SET model_version='wrong'")
    # Gán cho before: đọc toàn bộ file dưới dạng byte qua service.path.read_bytes(...).
    before = service.path.read_bytes()
    # Kiểm chứng điều kiện client.get('/health').status_code == 503; sai thì test dừng bằng AssertionError.
    assert client.get("/health").status_code == 503
    # Kiểm chứng điều kiện post_recognize(client).status_code == 503; sai thì test dừng bằng AssertionError.
    assert post_recognize(client).status_code == 503
    # Kiểm chứng điều kiện client.delete('/persons/alice').status_code == 503; sai thì test dừng bằng AssertionError.
    assert client.delete("/persons/alice").status_code == 503
    # Kiểm chứng điều kiện before == service.path.read_bytes(); sai thì test dừng bằng AssertionError.
    assert before == service.path.read_bytes()


# Kiểm tra: Giới hạn tổng request, từng file và định dạng ảnh đều được áp dụng.
# Định nghĩa hàm test_upload_limits_and_wrong_format(service).
def test_upload_limits_and_wrong_format(service):
    # Gán cho small: tạo bản cấu hình sao chép với một số trường thay đổi bằng replace(...); dữ liệu/tham số: service.settings, max_file_bytes=1024, max_request_bytes=2048.
    small = replace(service.settings, max_file_bytes=1024, max_request_bytes=2048)
    # Gán cho service.settings: small.
    service.settings = small
    # Mở khối with để tạo trình gọi API nội bộ cho kiểm thử bằng TestClient(...); dữ liệu/tham số: create_app(service) (tạo API và tùy chọn gắn giao diện bằng create_app(...)); gọi tài nguyên là client; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with TestClient(create_app(service)) as client:
        # Gán cho response: gửi yêu cầu HTTP POST qua client.post(...); dữ liệu/tham số: '/recognize', content=b'x' * 2049, headers={'content-type': 'application/octet-stream'}.
        response = client.post("/recognize", content=b"x" * 2049, headers={"content-type": "application/octet-stream"})
        # Kiểm chứng điều kiện response.status_code == 413; sai thì test dừng bằng AssertionError.
        assert response.status_code == 413
        # Gán cho response: gửi yêu cầu HTTP POST qua client.post(...); dữ liệu/tham số: '/recognize', files={'image': ('x.png', b'x' * 1025, 'image/png')}.
        response = client.post("/recognize", files={"image": ("x.png", b"x" * 1025, "image/png")})
        # Kiểm chứng điều kiện response.status_code == 413; sai thì test dừng bằng AssertionError.
        assert response.status_code == 413
        # Gán cho response: gửi yêu cầu HTTP POST qua client.post(...); dữ liệu/tham số: '/recognize', files={'image': ('x.gif', encoded(format='GIF'), 'image/gif')}.
        response = client.post("/recognize", files={"image": ("x.gif", encoded(format="GIF"), "image/gif")})
        # Kiểm chứng điều kiện response.status_code == 415; sai thì test dừng bằng AssertionError.
        assert response.status_code == 415


# Kiểm tra: Giải mã giữ đúng RGB, sửa chiều EXIF và kiểm tra kích thước.
# Định nghĩa hàm test_decoding_rgb_exif_and_dimensions().
def test_decoding_rgb_exif_and_dimensions():
    # Gán cho image: tạo mảng với mọi phần tử bằng giá trị chỉ định bằng np.full(...); dữ liệu/tham số: (40, 60, 3), [222, 13, 55], np.uint8.
    image = np.full((40, 60, 3), [222, 13, 55], np.uint8)
    # Kiểm chứng điều kiện np.array_equal(decode_image(encoded(image), Settings()), image); sai thì test dừng bằng AssertionError.
    assert np.array_equal(decode_image(encoded(image), Settings()), image)
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Giải mã byte thành ảnh RGB đã kiểm tra bằng decode_image(...); dữ liệu/tham số: encoded(np.zeros((20, 4097, 3), np.uint8)) (mã hóa ảnh test thành byte bằng encoded(...)), Settings() (tạo cấu hình bằng Settings(...)).
        decode_image(encoded(np.zeros((20, 4097, 3), np.uint8)), Settings())
    # Kiểm chứng điều kiện caught.value.status == 413; sai thì test dừng bằng AssertionError.
    assert caught.value.status == 413
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError):
        # Giải mã byte thành ảnh RGB đã kiểm tra bằng decode_image(...); dữ liệu/tham số: encoded(np.zeros((10, 30, 3), np.uint8)) (mã hóa ảnh test thành byte bằng encoded(...)), Settings() (tạo cấu hình bằng Settings(...)).
        decode_image(encoded(np.zeros((10, 30, 3), np.uint8)), Settings())
    # Gán cho exif: gọi Image.Exif(...).
    exif = Image.Exif()
    # Gán cho exif[274]: giá trị 6.
    exif[274] = 6
    # Gán cho data: tạo luồng byte trong bộ nhớ thay vì file trên ổ đĩa bằng BytesIO(...).
    data = BytesIO()
    # Mã hóa và lưu ảnh qua Image.fromarray(image).save(...); dữ liệu/tham số: data, format='JPEG', exif=exif.
    Image.fromarray(image).save(data, format="JPEG", exif=exif)
    # Kiểm chứng điều kiện decode_image(data.getvalue(), Settings()).shape == (60, 40, 3); sai thì test dừng bằng AssertionError.
    assert decode_image(data.getvalue(), Settings()).shape == (60, 40, 3)


# Kiểm tra: Vượt số ảnh hoặc số mặt bị chặn trước khi mạng suy luận.
# Định nghĩa hàm test_max_faces_and_max_images(client, service).
def test_max_faces_and_max_images(client, service):
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'alice', [picture()] * 11.
        service.enroll("alice", [picture()] * 11)
    # Kiểm chứng điều kiện caught.value.status == 422; sai thì test dừng bằng AssertionError.
    assert caught.value.status == 422
    # Kiểm chứng điều kiện post_recognize(client, picture(21)).status_code == 422; sai thì test dừng bằng AssertionError.
    assert post_recognize(client, picture(21)).status_code == 422
    # Kiểm chứng điều kiện service.runtime.net.calls == 0; sai thì test dừng bằng AssertionError.
    assert service.runtime.net.calls == 0


# Kiểm tra: Bộ đếm tăng qua từng frame và độc lập giữa các trạng thái phiên.
# Định nghĩa hàm test_webcam_sequential_frames_and_state_isolation(service).
def test_webcam_sequential_frames_and_state_isolation(service):
    # Gán cho state: từ điển {} (ánh xạ khóa → giá trị).
    state = {}
    # Lặp qua range(3); mỗi lượt gán phần tử cho index.
    for index in range(3):
        # Tách kết quả theo thứ tự và gán cho (annotated, result, state): xử lý một frame và cập nhật thống kê phiên webcam bằng process_frame(...); dữ liệu/tham số: service, picture() (tạo ảnh giả điều khiển kết quả model trong test bằng picture(...)), state.
        annotated, result, state = process_frame(service, picture(), state)
        # Kiểm chứng điều kiện annotated.shape == picture().shape; sai thì test dừng bằng AssertionError.
        assert annotated.shape == picture().shape
        # Kiểm chứng điều kiện result['frames_processed'] == index + 1; sai thì test dừng bằng AssertionError.
        assert result["frames_processed"] == index + 1
        # Kiểm chứng điều kiện result['frame_latency_ms'] > 0 and result['stream_fps'] > 0; sai thì test dừng bằng AssertionError.
        assert result["frame_latency_ms"] > 0 and result["stream_fps"] > 0
    # Kiểm chứng điều kiện process_frame(service, picture(), {})[1]['frames_processed'] == 1; sai thì test dừng bằng AssertionError.
    assert process_frame(service, picture(), {})[1]["frames_processed"] == 1
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Xử lý một frame và cập nhật thống kê phiên webcam bằng process_frame(...); dữ liệu/tham số: service, None.
        process_frame(service, None)
    # Kiểm chứng điều kiện caught.value.code == 'camera_unavailable'; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "camera_unavailable"


# Kiểm tra: Yêu cầu đồng thời được khóa; hai lượt ghi cùng ID chỉ có một lượt thành công.
# Định nghĩa hàm test_concurrent_requests_serialize_model_and_duplicate_writes(service).
def test_concurrent_requests_serialize_model_and_duplicate_writes(service):
    # Mở khối with để tạo nhóm luồng để chạy công việc đồng thời bằng ThreadPoolExecutor(...); dữ liệu/tham số: max_workers=4; gọi tài nguyên là executor; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Gán cho results: tạo danh sách bằng list(...); dữ liệu/tham số: executor.map(service.recognize, [picture()] * 6) (gọi executor.map(...) với service.recognize, [picture()] * 6).
        results = list(executor.map(service.recognize, [picture()] * 6))
    # Kiểm chứng điều kiện all((r['face_count'] == 1 for r in results)); sai thì test dừng bằng AssertionError.
    assert all(r["face_count"] == 1 for r in results)
    # Kiểm tra ID/ảnh, chặn trùng rồi gọi pipeline tạo vector và lưu hồ sơ.
    # Định nghĩa hàm enroll(_).
    def enroll(_):
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Kết thúc hàm và trả lấy phần tử/lát cắt service.enroll('racing_id', [picture()])['status'] từ service.enroll('racing_id', [picture()]).
            return service.enroll("racing_id", [picture()])["status"]
        # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
        except ApplicationError as error:
            # Kết thúc hàm và trả error.code (đọc thuộc tính).
            return error.code
    # Mở khối with để tạo nhóm luồng để chạy công việc đồng thời bằng ThreadPoolExecutor(...); dữ liệu/tham số: max_workers=2; gọi tài nguyên là executor; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Gán cho results: tạo danh sách bằng list(...); dữ liệu/tham số: executor.map(enroll, range(2)) (gọi executor.map(...) với enroll, range(2)).
        results = list(executor.map(enroll, range(2)))
    # Kiểm chứng điều kiện sorted(results) == ['duplicate_person', 'enrolled']; sai thì test dừng bằng AssertionError.
    assert sorted(results) == ["duplicate_person", "enrolled"]


# Kiểm tra: Hai dịch vụ cùng kho dùng chung khóa file và có lỗi khi chờ quá hạn.
# Định nghĩa hàm test_two_service_instances_share_registry_lock(service).
def test_two_service_instances_share_registry_lock(service):
    # Gán cho second: tạo dịch vụ dùng chung cho nhận diện và quản lý hồ sơ bằng ApplicationService(...); dữ liệu/tham số: replace(service.settings, lock_timeout=0.05) (tạo bản cấu hình sao chép với một số trường thay đổi bằng replace(...)), service._loader.
    second = ApplicationService(replace(service.settings, lock_timeout=.05), service._loader)
    # Mở khối with để lấy khóa truy cập model và kho hồ sơ qua service.guard(...); tạo nhóm luồng để chạy công việc đồng thời bằng ThreadPoolExecutor(...); dữ liệu/tham số: max_workers=1; gọi tài nguyên là executor; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with service.guard(), ThreadPoolExecutor(max_workers=1) as executor:
        # Gán cho future: giao công việc cho một luồng phụ qua executor.submit(...); dữ liệu/tham số: second.persons.
        future = executor.submit(second.persons)
        # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with pytest.raises(ApplicationError) as caught:
            # Chờ công việc nền xong và lấy kết quả hoặc lỗi qua future.result(...).
            future.result()
    # Kiểm chứng điều kiện caught.value.code == 'service_busy'; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "service_busy"


# Kiểm tra: Tập số đo rỗng và trung vị được tổng hợp đúng.
# Định nghĩa hàm test_benchmark_summary().
def test_benchmark_summary():
    # Lấy summarize từ application.benchmark.
    from application.benchmark import summarize
    # Kiểm chứng điều kiện summarize([])['status'] == 'not_measured'; sai thì test dừng bằng AssertionError.
    assert summarize([])["status"] == "not_measured"
    # Kiểm chứng điều kiện summarize([10, 20, 30])['p50_ms'] == 20; sai thì test dừng bằng AssertionError.
    assert summarize([10, 20, 30])["p50_ms"] == 20


# Kiểm tra: Dừng webcam không gọi model; bắt đầu lại dùng bộ đếm mới.
# Định nghĩa hàm test_ui_stop_skips_inference_and_start_resets_session(service).
def test_ui_stop_skips_inference_and_start_resets_session(service):
    # Lấy build_ui từ application.ui.
    from application.ui import build_ui
    # Gán cho demo: dựng giao diện Gradio bằng build_ui(...); dữ liệu/tham số: service.
    demo = build_ui(service)
    # Gán cho stream: lấy phần tử tiếp theo, hoặc giá trị mặc định nếu đã hết bằng next(...); dữ liệu/tham số: (fn.fn for fn in demo.fns.values() if getattr(fn.fn, '__name__', '') == 'stream').
    stream = next(fn.fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "stream")
    # Gán cho before: service.runtime.net.calls (đọc thuộc tính).
    before = service.runtime.net.calls
    # Tách kết quả theo thứ tự và gán cho (_, _, state, message): gọi stream(...) với picture(), False, {}; dữ liệu/tham số: picture() (tạo ảnh giả điều khiển kết quả model trong test bằng picture(...)), False, {}.
    _, _, state, message = stream(picture(), False, {})
    # Kiểm chứng điều kiện service.runtime.net.calls == before and state == {}; sai thì test dừng bằng AssertionError.
    assert service.runtime.net.calls == before and state == {}
    # Kiểm chứng điều kiện 'dừng' in message; sai thì test dừng bằng AssertionError.
    assert "dừng" in message
    # Tách kết quả theo thứ tự và gán cho (_, result, state, _): gọi stream(...) với picture(), True, {}; dữ liệu/tham số: picture() (tạo ảnh giả điều khiển kết quả model trong test bằng picture(...)), True, {}.
    _, result, state, _ = stream(picture(), True, {})
    # Kiểm chứng điều kiện result['frames_processed'] == 1; sai thì test dừng bằng AssertionError.
    assert result["frames_processed"] == 1
    # Tách kết quả theo thứ tự và gán cho (_, _, state, _): gọi stream(...) với picture(), True, state; dữ liệu/tham số: picture() (tạo ảnh giả điều khiển kết quả model trong test bằng picture(...)), True, state.
    _, _, state, _ = stream(picture(), True, state)
    # Kiểm chứng điều kiện state['frames'] == 2; sai thì test dừng bằng AssertionError.
    assert state["frames"] == 2


# Kiểm tra: Vector sai checksum khiến kho bị đánh dấu không hợp lệ.
# Định nghĩa hàm test_wrong_schema_and_corrupt_checksum_are_unavailable(service).
def test_wrong_schema_and_corrupt_checksum_are_unavailable(service):
    # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'alice', [picture()].
    service.enroll("alice", [picture()])
    # Mở khối with để bảo đảm tài nguyên được close khi kết thúc khối with bằng closing(...); dữ liệu/tham số: db.connect_database(service.path) (gọi db.connect_database(...) với service.path); gọi tài nguyên là connection; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with closing(db.connect_database(service.path)) as connection:
        # Mở khối with để connection (kết nối SQLite); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with connection:
            # Thực hiện câu lệnh SQL và truyền các tham số đi kèm qua connection.execute(...); dữ liệu/tham số: 'UPDATE FaceDatabase SET embedding_sha256=?', ('0' * 64,).
            connection.execute("UPDATE FaceDatabase SET embedding_sha256=?", ("0" * 64,))
    # Gán cho before: đọc toàn bộ file dưới dạng byte qua service.path.read_bytes(...).
    before = service.path.read_bytes()
    # Kiểm chứng điều kiện service.health()['registry'] == 'invalid'; sai thì test dừng bằng AssertionError.
    assert service.health()["registry"] == "invalid"
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Nhận diện ảnh bằng dịch vụ qua service.recognize(...); dữ liệu/tham số: picture() (tạo ảnh giả điều khiển kết quả model trong test bằng picture(...)).
        service.recognize(picture())
    # Kiểm chứng điều kiện caught.value.status == 503 and service.path.read_bytes() == before; sai thì test dừng bằng AssertionError.
    assert caught.value.status == 503 and service.path.read_bytes() == before


# Kiểm tra: Ảnh thứ hai lỗi thì toàn bộ lượt đăng ký không được ghi.
# Định nghĩa hàm test_late_inference_failure_rolls_back_entire_enrollment(service, monkeypatch).
def test_late_inference_failure_rolls_back_entire_enrollment(service, monkeypatch):
    # Gán cho original: service.runtime.net.forward (đọc thuộc tính).
    original = service.runtime.net.forward
    # Tạo vector 512 chiều với một phần tử bằng 1; có thể bật cờ để giả lỗi model.
    # Định nghĩa hàm forward(name).
    def forward(name):
        # Chỉ chạy khối bên dưới khi service.runtime.net.calls == 1 cho kết quả đúng.
        if service.runtime.net.calls == 1:
            # Phát sinh lỗi: gọi RuntimeError(...) với 'second image failed'; dữ liệu/tham số: 'second image failed'.
            raise RuntimeError("second image failed")
        # Kết thúc hàm và trả gọi original(...) với name; dữ liệu/tham số: name.
        return original(name)
    # Gọi monkeypatch.setattr(...) với service.runtime.net, 'forward', forward; dữ liệu/tham số: service.runtime.net, 'forward', forward.
    monkeypatch.setattr(service.runtime.net, "forward", forward)
    # Gán cho before: đọc toàn bộ file dưới dạng byte qua service.path.read_bytes(...).
    before = service.path.read_bytes()
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'alice', [picture(), picture()].
        service.enroll("alice", [picture(), picture()])
    # Kiểm chứng điều kiện caught.value.status == 500 and service.path.read_bytes() == before; sai thì test dừng bằng AssertionError.
    assert caught.value.status == 500 and service.path.read_bytes() == before


# Kiểm tra: Giao diện có đúng 4 tab và hai luồng webcam liên tục.
# Định nghĩa hàm test_ui_mount_webcam_only_enrollment_and_continuous_streams(service).
def test_ui_mount_webcam_only_enrollment_and_continuous_streams(service):
    # Mở khối with để tạo trình gọi API nội bộ cho kiểm thử bằng TestClient(...); dữ liệu/tham số: create_app(service, with_ui=True) (tạo API và tùy chọn gắn giao diện bằng create_app(...)); gọi tài nguyên là client; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with TestClient(create_app(service, with_ui=True)) as client:
        # Kiểm chứng điều kiện client.get('/ui/').status_code == 200; sai thì test dừng bằng AssertionError.
        assert client.get("/ui/").status_code == 200
        # Gán cho config: đọc JSON từ phản hồi HTTP qua client.get('/ui/config').json(...).
        config = client.get("/ui/config").json()
    # Gán cho tabs: tạo danh sách bằng cách duyệt part trong config['components'] và chỉ giữ phần tử đạt điều kiện if; mỗi phần tử là part['props']['label'].
    tabs = [part["props"]["label"] for part in config["components"] if part["type"] == "tabitem"]
    # Kiểm chứng điều kiện tabs == ['1 · Đăng ký webcam · 4 góc', '2 · Nhận diện ảnh', '3 · Webcam trực tiếp', '4 · Danh sách đăng ký']; sai thì test dừng bằng AssertionError.
    assert tabs == ["1 · Đăng ký webcam · 4 góc", "2 · Nhận diện ảnh", "3 · Webcam trực tiếp", "4 · Danh sách đăng ký"]
    # Gán cho streams: tạo danh sách bằng cách duyệt item trong config['dependencies'] và chỉ giữ phần tử đạt điều kiện if; mỗi phần tử là item.
    streams = [item for item in config["dependencies"] if any(target[1] == "stream" for target in item["targets"])]
    # Kiểm chứng điều kiện len(streams) == 2 and all((item['trigger_mode'] == 'multiple' for item in streams)); sai thì test dừng bằng AssertionError.
    assert len(streams) == 2 and all(item["trigger_mode"] == "multiple" for item in streams)
    # Kiểm chứng điều kiện '#enrollment-camera-status' in config['css']; sai thì test dừng bằng AssertionError.
    assert "#enrollment-camera-status" in config["css"]


# Kiểm tra: Chỉ xóa hồ sơ sau xác nhận và cập nhật bảng cùng kho tên.
# Định nghĩa hàm test_ui_registration_list_confirms_one_deletion(service).
def test_ui_registration_list_confirms_one_deletion(service):
    # Lấy build_ui từ application.ui.
    from application.ui import build_ui
    # Lấy commit_identity, read_names từ application.webcam_registry.
    from application.webcam_registry import commit_identity, read_names

    # Gán cho embedding: tạo mảng gồm các số 0 bằng np.zeros(...); dữ liệu/tham số: 512, dtype=np.float32.
    embedding = np.zeros(512, dtype=np.float32)
    # Gán cho embedding[0]: giá trị 1.
    embedding[0] = 1
    # Lưu vector và tên trong cùng giao dịch bằng commit_identity(...); dữ liệu/tham số: service.path, 'alice', 'Nguyễn An', num_images=4.
    commit_identity(service.path, "alice", "Nguyễn An", embedding, service.runtime.version, num_images=4)
    # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'legacy', [picture()].
    service.enroll("legacy", [picture()])
    # Gán cho demo: dựng giao diện Gradio bằng build_ui(...); dữ liệu/tham số: service.
    demo = build_ui(service)
    # Gán cho callbacks: tạo từ điển bằng vòng lặp rút gọn; khóa là f.fn.__name__, giá trị là f.fn.
    callbacks = {f.fn.__name__: f.fn for f in demo.fns.values() if hasattr(f.fn, "__name__")}

    # Tách kết quả theo thứ tự và gán cho (rows, choice, *_): gọi callbacks['refresh_registrations'](...).
    rows, choice, *_ = callbacks["refresh_registrations"]()
    # Kiểm chứng điều kiện rows == [['alice', 'Nguyễn An', 4], ['legacy', '—', 1]]; sai thì test dừng bằng AssertionError.
    assert rows == [["alice", "Nguyễn An", 4], ["legacy", "—", 1]]
    # Kiểm chứng điều kiện choice['choices'] == [('Nguyễn An · alice', 'alice'), ('legacy', 'legacy')]; sai thì test dừng bằng AssertionError.
    assert choice["choices"] == [("Nguyễn An · alice", "alice"), ("legacy", "legacy")]
    # Tách kết quả theo thứ tự và gán cho (pending, prompt, question, _): gọi callbacks['request_registration_delete'](...) với 'alice'; dữ liệu/tham số: 'alice'.
    pending, prompt, question, _ = callbacks["request_registration_delete"]("alice")
    # Kiểm chứng điều kiện prompt['visible'] and 'alice' in question; sai thì test dừng bằng AssertionError.
    assert prompt["visible"] and "alice" in question
    # Kiểm chứng điều kiện len(service.persons()) == 2; sai thì test dừng bằng AssertionError.
    assert len(service.persons()) == 2

    # Gọi callbacks['confirm_registration_delete'](...) với None; dữ liệu/tham số: None.
    callbacks["confirm_registration_delete"](None)
    # Kiểm chứng điều kiện len(service.persons()) == 2; sai thì test dừng bằng AssertionError.
    assert len(service.persons()) == 2
    # Tách kết quả theo thứ tự và gán cho (rows, _, _, pending_after, prompt_after, message): gọi callbacks['confirm_registration_delete'](...) với pending; dữ liệu/tham số: pending.
    rows, _, _, pending_after, prompt_after, message = callbacks["confirm_registration_delete"](pending)
    # Kiểm chứng điều kiện rows == [['legacy', '—', 1]]; sai thì test dừng bằng AssertionError.
    assert rows == [["legacy", "—", 1]]
    # Kiểm chứng điều kiện pending_after is None and prompt_after['visible'] is False; sai thì test dừng bằng AssertionError.
    assert pending_after is None and prompt_after["visible"] is False
    # Kiểm chứng điều kiện 'Đã xóa' in message and read_names(service.path) == {}; sai thì test dừng bằng AssertionError.
    assert "Đã xóa" in message and read_names(service.path) == {}
