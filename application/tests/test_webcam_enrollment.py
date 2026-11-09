# Kiểm thử đăng ký webcam bằng đồng hồ/model giả; mỗi bài dùng dữ liệu riêng.
# Lấy ThreadPoolExecutor từ concurrent.futures để chạy nhiều công việc qua các luồng và thu kết quả.
from concurrent.futures import ThreadPoolExecutor
# Lấy closing từ contextlib để quản lý tài nguyên bằng with, yield và thao tác dọn tự động.
from contextlib import closing
# Lấy replace từ dataclasses để tạo lớp dữ liệu và sao chép/chuyển đổi dữ liệu của lớp.
from dataclasses import replace
# Lấy BytesIO từ io để đọc/ghi byte ảnh trong bộ nhớ bằng BytesIO.
from io import BytesIO
# Nạp sqlite3 để đọc, ghi và quản lý giao dịch SQLite.
import sqlite3
# Lấy SimpleNamespace từ types để tạo đối tượng dữ liệu đơn giản bằng SimpleNamespace.
from types import SimpleNamespace

# Nạp cv2 để xử lý ảnh, chạy ONNX và tính toán hình học camera bằng OpenCV.
import cv2
# Nạp numpy dưới tên ngắn np để tính toán mảng ảnh, vector và ma trận.
import numpy as np
# Lấy Image từ PIL để đọc/ghi ảnh, xử lý hướng EXIF, chọn font và vẽ lên ảnh bằng Pillow.
from PIL import Image
# Nạp pytest để viết và chạy bài kiểm thử tự động.
import pytest
# Lấy TestClient từ fastapi.testclient để gọi thử API mà không cần mở cổng máy chủ thật.
from fastapi.testclient import TestClient

# Lấy build_model_version, initialize_enrollment_registry từ FaceNet.
from FaceNet import build_model_version, initialize_enrollment_registry
# Lấy database với bí danh db từ FaceNet.
from FaceNet import database as db
# Lấy create_app từ application.api.
from application.api import create_app
# Lấy Settings từ application.config.
from application.config import Settings
# Lấy ApplicationError từ application.errors.
from application.errors import ApplicationError
# Lấy _MODEL_POINTS từ application.head_pose.
from application.head_pose import _MODEL_POINTS
# Lấy ApplicationService, Runtime từ application.service.
from application.service import ApplicationService, Runtime
# Lấy commit_identity, metadata_path, read_names từ application.webcam_registry.
from application.webcam_registry import commit_identity, metadata_path, read_names


# Đồng hồ giả để kiểm tra thời gian giữ và hết hạn mà không phải chờ thật.
# Định nghĩa lớp Clock; lớp gom dữ liệu và các hàm liên quan.
class Clock:
    # Khởi tạo đối tượng và giữ các phụ thuộc/trạng thái cần dùng trong những lần gọi sau.
    # Định nghĩa hàm __init__(self). self là đối tượng đang gọi phương thức.
    def __init__(self):
        # Gán cho self.now: giá trị 100.0.
        self.now = 100.0
    # Trả thời gian giả hiện tại mỗi khi bộ quản lý phiên hỏi đồng hồ.
    # Định nghĩa hàm __call__(self). self là đối tượng đang gọi phương thức.
    def __call__(self):
        # Kết thúc hàm và trả self.now (đọc thuộc tính).
        return self.now
    # Tăng thời gian giả lên số giây chỉ định để mô phỏng các frame đến sau nhau.
    # Định nghĩa hàm advance(self, seconds). self là đối tượng đang gọi phương thức. Mặc định: seconds=0.3.
    def advance(self, seconds=.3):
        # Cập nhật self.now: cộng thêm seconds và lưu lại vào chính nó.
        self.now += seconds


# Detector giả trả dữ liệu được điều khiển bởi ảnh test; không đo độ chính xác model thật.
# Định nghĩa lớp Detector; lớp gom dữ liệu và các hàm liên quan.
class Detector:
    # Tách kết quả theo thứ tự và gán cho (min_face_size, factor, thresholds, batch_size, device): bộ giá trị (20, 0.709, (0.6, 0.7, 0.7), 256, 'cpu') (tuple giữ thứ tự, không gán lại từng phần tử).
    min_face_size, factor, thresholds, batch_size, device = 20, .709, (.6, .7, .7), 256, "cpu"
    # Gán cho fail: False (sai/tắt).
    fail = False

    # Phát hiện khuôn mặt trong ảnh và trả khung mặt, điểm tin cậy, điểm mốc.
    # Định nghĩa hàm detect(self, image). self là đối tượng đang gọi phương thức.
    def detect(self, image):
        # Chỉ chạy khối bên dưới khi self.fail cho kết quả đúng.
        if self.fail:
            # Phát sinh lỗi: gọi RuntimeError(...) với 'model failed'; dữ liệu/tham số: 'model failed'.
            raise RuntimeError("model failed")
        # Tách kết quả theo thứ tự và gán cho (n, pose): áp dụng hàm cho từng phần tử bằng map(...); dữ liệu/tham số: int, image[0, 0, :2].
        n, pose = map(int, image[0, 0, :2])
        # Tách kết quả theo thứ tự và gán cho (yaw, pitch): lấy phần tử/lát cắt [(0, 0), (-25, 0), (25, 0), (0, -25), (0, 25)][pose] từ [(0, 0), (-25, 0), (25, 0), (0, -25), (0, 25)].
        yaw, pitch = [(0, 0), (-25, 0), (25, 0), (0, -25), (0, 25)][pose]
        # Tách kết quả theo thứ tự và gán cho (y, p): đổi độ sang radian bằng np.deg2rad(...); dữ liệu/tham số: [yaw, pitch].
        y, p = np.deg2rad([yaw, pitch])
        # Gán cho ry: tạo mảng NumPy từ dữ liệu bằng np.array(...); dữ liệu/tham số: [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]].
        ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        # Gán cho rx: tạo mảng NumPy từ dữ liệu bằng np.array(...); dữ liệu/tham số: [[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]].
        rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
        # Tách kết quả theo thứ tự và gán cho (rotation, _): chuyển giữa vector quay và ma trận quay bằng cv2.Rodrigues(...); dữ liệu/tham số: ry @ rx.
        rotation, _ = cv2.Rodrigues(ry @ rx)
        # Tách kết quả theo thứ tự và gán cho (points, _): chiếu điểm 3D qua camera để thu tọa độ ảnh 2D bằng cv2.projectPoints(...); dữ liệu/tham số: _MODEL_POINTS, rotation, np.array([0.0, 0.0, 180.0]) (tạo mảng NumPy từ dữ liệu bằng np.array(...)).
        points, _ = cv2.projectPoints(_MODEL_POINTS, rotation, np.array([0., 0., 180.]),
                                     # Tiếp phần biểu thức nhiều dòng: tạo mảng NumPy từ dữ liệu bằng np.array(...).
                                     np.array([[320., 0, 160.], [0, 320., 120.], [0, 0, 1.]]), None)
        # Kết thúc hàm và trả tạo đối tượng đơn giản có các thuộc tính từ tham số đặt tên bằng SimpleNamespace(...); dữ liệu/tham số: boxes=np.tile(np.array([[50, 15, 270, 225]], np.float32), (n, 1)), scores=np.full(n, 0.99, np.float32), landmarks=np.tile(points.reshape(1, 5, 2).astype(np.float32), (n, 1, 1)).
        return SimpleNamespace(boxes=np.tile(np.array([[50, 15, 270, 225]], np.float32), (n, 1)),
                               # Truyền scores = np.full(n, 0.99, np.float32).
                               scores=np.full(n, .99, np.float32),
                               # Truyền landmarks = np.tile(points.reshape(1, 5, 2).astype(np.float32), (n, 1, 1)).
                               landmarks=np.tile(points.reshape(1, 5, 2).astype(np.float32), (n, 1, 1)))

    # Cắt/căn chỉnh các mặt về ảnh RGB có kích thước đầu vào của FaceNet.
    # Định nghĩa hàm extract_faces_rgb(self, image, result, size, align). self là đối tượng đang gọi phương thức.
    def extract_faces_rgb(self, image, result, size, align):
        # Kết thúc hàm và trả tạo mảng với mọi phần tử bằng giá trị chỉ định bằng np.full(...); dữ liệu/tham số: (len(result.boxes), 160, 160, 3), int(image[0, 0, 2]) (đổi sang số nguyên bằng int(...)), np.uint8.
        return np.full((len(result.boxes), 160, 160, 3), int(image[0, 0, 2]), np.uint8)


# Mạng giả tạo vector dự đoán được để kiểm tra nhận diện và lỗi xử lý.
# Định nghĩa lớp Net; lớp gom dữ liệu và các hàm liên quan.
class Net:
    # Khởi tạo đối tượng và giữ các phụ thuộc/trạng thái cần dùng trong những lần gọi sau.
    # Định nghĩa hàm __init__(self). self là đối tượng đang gọi phương thức.
    def __init__(self):
        # Gán cho self.fail: False (sai/tắt).
        self.fail = False
        # Gán cho self.calls: giá trị 0.
        self.calls = 0
    # Chuyển dữ liệu đầu vào cho mạng; giữ nguyên giao diện gọi của OpenCV DNN.
    # Định nghĩa hàm setInput(self, batch, name). self là đối tượng đang gọi phương thức.
    def setInput(self, batch, name):
        # Gán cho self.index: làm tròn số bằng round(...); dữ liệu/tham số: float(batch[0, 0, 0, 0]) * 128 + 127.5.
        self.index = round(float(batch[0, 0, 0, 0]) * 128 + 127.5)
    # Chạy mạng và thu đầu ra; vector đặc trưng là dãy số mô tả khuôn mặt.
    # Định nghĩa hàm forward(self, name). self là đối tượng đang gọi phương thức.
    def forward(self, name):
        # Cập nhật self.calls: cộng thêm 1 và lưu lại vào chính nó.
        self.calls += 1
        # Chỉ chạy khối bên dưới khi self.fail cho kết quả đúng.
        if self.fail:
            # Phát sinh lỗi: gọi RuntimeError(...) với 'inference failure'; dữ liệu/tham số: 'inference failure'.
            raise RuntimeError("inference failure")
        # Gán cho output: tạo mảng gồm các số 0 bằng np.zeros(...); dữ liệu/tham số: (1, 512), np.float32.
        output = np.zeros((1, 512), np.float32)
        # Gán cho output[0, self.index]: giá trị 1.
        output[0, self.index] = 1
        # Kết thúc hàm và trả output (đầu ra).
        return output


# Tạo frame tổng hợp mã hóa góc, số mặt, danh tính và số thứ tự để tránh ảnh lặp.
# Định nghĩa hàm frame(pose, count, identity, sequence). Mặc định: pose=0, count=1, identity=0, sequence=0.
def frame(pose=0, count=1, identity=0, sequence=0):
    # Gán cho image: gọi np.random.default_rng(42).integers(...) với 65, 190, (240, 320, 3); dữ liệu/tham số: 65, 190, (240, 320, 3), dtype=np.uint8.
    image = np.random.default_rng(42).integers(65, 190, (240, 320, 3), dtype=np.uint8)
    # Gán cho image[0, 0]: bộ giá trị (count, pose, identity) (tuple giữ thứ tự, không gán lại từng phần tử).
    image[0, 0] = (count, pose, identity)
    # Gán cho image[1, 1, 0]: sequence.
    image[1, 1, 0] = sequence
    # Kết thúc hàm và trả image (ảnh đang xử lý).
    return image


# Mã hóa mảng ảnh thành byte PNG để kiểm thử đường upload của API.
# Định nghĩa hàm png(image).
def png(image):
    # Gán cho data: tạo luồng byte trong bộ nhớ thay vì file trên ổ đĩa bằng BytesIO(...).
    data = BytesIO()
    # Mã hóa và lưu ảnh qua Image.fromarray(image).save(...); dữ liệu/tham số: data, format='PNG'.
    Image.fromarray(image).save(data, format="PNG")
    # Kết thúc hàm và trả lấy toàn bộ byte đang chứa trong bộ đệm qua data.getvalue(...).
    return data.getvalue()


# Fixture tạo kho SQLite riêng trong tmp_path và nạp model giả cho mỗi bài test.
# Decorator fixture: pytest tự tạo dữ liệu/phụ thuộc này khi bài test yêu cầu tên hàm.
@pytest.fixture
# Định nghĩa hàm service(tmp_path).
def service(tmp_path):
    # Gán cho path: ghép đường dẫn bằng toán tử / của pathlib trong tmp_path / 'registry.sqlite'. Ở đây / nối thư mục với tên file, không phải phép chia số.
    path = tmp_path / "registry.sqlite"
    # Khởi tạo kho SQLite theo cấu trúc FaceNet bằng initialize_enrollment_registry(...); dữ liệu/tham số: path.
    initialize_enrollment_registry(path)
    # Gán cho detector: gọi Detector(...).
    detector = Detector()
    # Gán cho rt: gom detector, mạng và phiên bản model bằng Runtime(...); dữ liệu/tham số: detector, Net() (gọi Net(...)), build_model_version(detector) (tạo thông tin phiên bản model và môi trường chạy bằng build_model_version(...)).
    rt = Runtime(detector, Net(), build_model_version(detector))
    # Gán cho result: tạo dịch vụ dùng chung cho nhận diện và quản lý hồ sơ bằng ApplicationService(...); dữ liệu/tham số: Settings(registry=path) (tạo cấu hình bằng Settings(...)), lambda _: rt.
    result = ApplicationService(Settings(registry=path), lambda _: rt)
    # Khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua result.start(...).
    result.start()
    # Gán cho result.webcam_enrollment.clock: gọi Clock(...).
    result.webcam_enrollment.clock = Clock()
    # Kết thúc hàm và trả result (kết quả xử lý).
    return result


# Gửi đủ frame khác nhau và tăng đồng hồ giả để một góc được tự động chụp.
# Định nghĩa hàm capture(service, state, pose, identity). Mặc định: identity=0.
def capture(service, state, pose, identity=0):
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Lặp qua range(4); mỗi lượt gán phần tử cho i.
    for i in range(4):
        # Tăng đồng hồ giả theo số giây chỉ định qua manager.clock.advance(...).
        manager.clock.advance()
        # Gán cho state: kiểm tra và xử lý một frame đăng ký qua manager.frame(...); dữ liệu/tham số: state['session_id'], frame(pose, identity=identity, sequence=i) (tạo frame giả cho bài kiểm thử bằng frame(...)), state['revision'].
        state = manager.frame(state["session_id"], frame(pose, identity=identity, sequence=i), state["revision"])
    # Kiểm chứng điều kiện state.get('captured'); sai thì test dừng bằng AssertionError.
    assert state.get("captured"), state
    # Kết thúc hàm và trả state (trạng thái phiên giao diện).
    return state


# Lần lượt chụp đủ 4 góc để các bài test tập trung vào bước xem lại/lưu.
# Định nghĩa hàm complete(service, identity). Mặc định: identity=0.
def complete(service, identity=0):
    # Gán cho state: khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua service.webcam_enrollment.start(...); dữ liệu/tham số: 'alice', 'Nguyễn An'.
    state = service.webcam_enrollment.start("alice", "Nguyễn An")
    # Lặp qua range(4); mỗi lượt gán phần tử cho i.
    for i in range(4):
        # Gán cho state: mô phỏng chuỗi frame để chụp được một góc bằng capture(...); dữ liệu/tham số: service, state, i.
        state = capture(service, state, i, identity)
    # Lấy ảnh đã chụp để xem lại qua service.webcam_enrollment.review(...); dữ liệu/tham số: state['session_id'].
    service.webcam_enrollment.review(state["session_id"])
    # Kết thúc hàm và trả state (trạng thái phiên giao diện).
    return state


# Kiểm tra: Chụp đủ 4 góc, xác nhận rồi lưu đồng bộ tên/vector và nhận diện được.
# Định nghĩa hàm test_four_poses_confirm_atomic_name_and_recognition(service).
def test_four_poses_confirm_atomic_name_and_recognition(service):
    # Gán cho before: đọc toàn bộ file dưới dạng byte qua service.path.read_bytes(...).
    before = service.path.read_bytes()
    # Gán cho state: mô phỏng chụp đủ 4 góc bằng complete(...); dữ liệu/tham số: service.
    state = complete(service)
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Kiểm chứng điều kiện state['progress'] == state['total'] == 4 and state['status'] == 'review'; sai thì test dừng bằng AssertionError.
    assert state["progress"] == state["total"] == 4 and state["status"] == "review"
    # Kiểm chứng điều kiện service.path.read_bytes() == before and (not metadata_path(service.path).exists()); sai thì test dừng bằng AssertionError.
    assert service.path.read_bytes() == before and not metadata_path(service.path).exists()
    # Kiểm chứng điều kiện len(manager.review(state['session_id'])) == 4; sai thì test dừng bằng AssertionError.
    assert len(manager.review(state["session_id"])) == 4
    # Gán cho result: xác nhận lưu bộ ảnh đã xem qua manager.confirm(...); dữ liệu/tham số: state['session_id'], True, state['revision'].
    result = manager.confirm(state["session_id"], True, state["revision"])
    # Kiểm chứng điều kiện result == {'person_id': 'alice', 'display_name': 'Nguyễn An', 'num_images': 4, 'status': 'enrolled'}; sai thì test dừng bằng AssertionError.
    assert result == {"person_id": "alice", "display_name": "Nguyễn An", "num_images": 4, "status": "enrolled"}
    # Kiểm chứng điều kiện db.get_person(service.path, 'alice', service.runtime.version)['num_images'] == 4; sai thì test dừng bằng AssertionError.
    assert db.get_person(service.path, "alice", service.runtime.version)["num_images"] == 4
    # Kiểm chứng điều kiện service.persons() == [{'person_id': 'alice', 'num_images': 4, 'display_name': 'Nguyễn An'}]; sai thì test dừng bằng AssertionError.
    assert service.persons() == [{"person_id": "alice", "num_images": 4, "display_name": "Nguyễn An"}]
    # Kiểm chứng điều kiện service.recognize(frame())['faces'][0]['person_id'] == 'alice'; sai thì test dừng bằng AssertionError.
    assert service.recognize(frame())["faces"][0]["person_id"] == "alice"
    # Kiểm chứng điều kiện service.recognize(frame(identity=1))['faces'][0]['status'] == 'unknown'; sai thì test dừng bằng AssertionError.
    assert service.recognize(frame(identity=1))["faces"][0]["status"] == "unknown"
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError, match='Phiên'; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError, match="Phiên"):
        # Xác nhận lưu bộ ảnh đã xem qua manager.confirm(...); dữ liệu/tham số: state['session_id'], True, state['revision'].
        manager.confirm(state["session_id"], True, state["revision"])
    # Gửi lệnh xóa theo ID hoặc URL qua service.delete(...); dữ liệu/tham số: 'alice'.
    service.delete("alice")
    # Kiểm chứng điều kiện service.persons() == [] and read_names(service.path) == {}; sai thì test dừng bằng AssertionError.
    assert service.persons() == [] and read_names(service.path) == {}


# Kiểm tra: Không có mặt hoặc nhiều mặt làm mất chuỗi ổn định và chưa ghi hồ sơ.
# Decorator parametrize: chạy cùng một bài test với nhiều bộ dữ liệu liệt kê bên dưới.
@pytest.mark.parametrize("count", [0, 2])
# Định nghĩa hàm test_face_count_breaks_stability_and_never_writes(service, count).
def test_face_count_breaks_stability_and_never_writes(service, count):
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Gán cho state: khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua manager.start(...); dữ liệu/tham số: 'alice', 'An'.
    state = manager.start("alice", "An")
    # Kiểm tra và xử lý một frame đăng ký qua manager.frame(...); dữ liệu/tham số: state['session_id'], frame() (tạo frame giả cho bài kiểm thử bằng frame(...)), 0.
    manager.frame(state["session_id"], frame(), 0)
    # Gán cho result: kiểm tra và xử lý một frame đăng ký qua manager.frame(...); dữ liệu/tham số: state['session_id'], frame(count=count, sequence=1) (tạo frame giả cho bài kiểm thử bằng frame(...)), 0.
    result = manager.frame(state["session_id"], frame(count=count, sequence=1), 0)
    # Kiểm chứng điều kiện result['error']['code'] == 'enrollment_face_count'; sai thì test dừng bằng AssertionError.
    assert result["error"]["code"] == "enrollment_face_count"
    # Kiểm chứng điều kiện result['stable_frames'] == 0 and result['progress'] == 0; sai thì test dừng bằng AssertionError.
    assert result["stable_frames"] == 0 and result["progress"] == 0
    # Kiểm chứng điều kiện db.list_people(service.path) == []; sai thì test dừng bằng AssertionError.
    assert db.list_people(service.path) == []


# Kiểm tra: Sai góc không được chụp; giữ mãi một góc không làm tăng số góc đã chụp.
# Định nghĩa hàm test_wrong_pose_unstable_and_no_duplicate_pose(service).
def test_wrong_pose_unstable_and_no_duplicate_pose(service):
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Gán cho state: khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua manager.start(...); dữ liệu/tham số: 'alice', 'An'.
    state = manager.start("alice", "An")
    # Gán cho bad: kiểm tra và xử lý một frame đăng ký qua manager.frame(...); dữ liệu/tham số: state['session_id'], frame(1) (tạo frame giả cho bài kiểm thử bằng frame(...)), 0.
    bad = manager.frame(state["session_id"], frame(1), 0)
    # Kiểm chứng điều kiện bad['error']['code'] == 'wrong_pose'; sai thì test dừng bằng AssertionError.
    assert bad["error"]["code"] == "wrong_pose"
    # Lặp qua range(3); mỗi lượt gán phần tử cho i.
    for i in range(3):
        # Tăng đồng hồ giả theo số giây chỉ định qua manager.clock.advance(...).
        manager.clock.advance()
        # Gán cho state: kiểm tra và xử lý một frame đăng ký qua manager.frame(...); dữ liệu/tham số: state['session_id'], frame(sequence=i) (tạo frame giả cho bài kiểm thử bằng frame(...)), 0.
        state = manager.frame(state["session_id"], frame(sequence=i), 0)
        # Kiểm chứng điều kiện state['progress'] == 0; sai thì test dừng bằng AssertionError.
        assert state["progress"] == 0
    # Tăng đồng hồ giả theo số giây chỉ định qua manager.clock.advance(...).
    manager.clock.advance()
    # Gán cho state: kiểm tra và xử lý một frame đăng ký qua manager.frame(...); dữ liệu/tham số: state['session_id'], frame(sequence=3) (tạo frame giả cho bài kiểm thử bằng frame(...)), 0.
    state = manager.frame(state["session_id"], frame(sequence=3), 0)
    # Kiểm chứng điều kiện state['progress'] == 1; sai thì test dừng bằng AssertionError.
    assert state["progress"] == 1
    # Lặp qua range(8); mỗi lượt gán phần tử cho i.
    for i in range(8):
        # Tăng đồng hồ giả theo số giây chỉ định qua manager.clock.advance(...).
        manager.clock.advance()
        # Gán cho state: kiểm tra và xử lý một frame đăng ký qua manager.frame(...); dữ liệu/tham số: state['session_id'], frame(sequence=i) (tạo frame giả cho bài kiểm thử bằng frame(...)), state['revision'].
        state = manager.frame(state["session_id"], frame(sequence=i), state["revision"])
    # Kiểm chứng điều kiện state['progress'] == 1 and state['next_pose'] == 'left'; sai thì test dừng bằng AssertionError.
    assert state["progress"] == 1 and state["next_pose"] == "left"
    # Kiểm chứng điều kiện state['stable_frames'] == 0; sai thì test dừng bằng AssertionError.
    assert state["stable_frames"] == 0


# Kiểm tra: Gửi lặp một ảnh tĩnh không được tính là chuỗi frame mới ổn định.
# Định nghĩa hàm test_repeated_static_frame_cannot_supply_stable_frames(service).
def test_repeated_static_frame_cannot_supply_stable_frames(service):
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Gán cho state: khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua manager.start(...); dữ liệu/tham số: 'alice', 'An'.
    state = manager.start("alice", "An")
    # Lặp qua range(6); mỗi lượt gán phần tử cho _.
    for _ in range(6):
        # Tăng đồng hồ giả theo số giây chỉ định qua manager.clock.advance(...).
        manager.clock.advance()
        # Gán cho result: kiểm tra và xử lý một frame đăng ký qua manager.frame(...); dữ liệu/tham số: state['session_id'], frame() (tạo frame giả cho bài kiểm thử bằng frame(...)), 0.
        result = manager.frame(state["session_id"], frame(), 0)
    # Kiểm chứng điều kiện result['error']['code'] == 'repeated_frame' and result['progress'] == 0; sai thì test dừng bằng AssertionError.
    assert result["error"]["code"] == "repeated_frame" and result["progress"] == 0


# Kiểm tra: Độ sáng và độ nét được kiểm tra trên vùng mặt.
# Decorator parametrize: chạy cùng một bài test với nhiều bộ dữ liệu liệt kê bên dưới.
@pytest.mark.parametrize("mode,code", [("dark", "poor_lighting"), ("bright", "poor_lighting"), ("blur", "blurry_image")])
# Định nghĩa hàm test_quality_on_face_region(service, mode, code).
def test_quality_on_face_region(service, mode, code):
    # Gán cho state: khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua service.webcam_enrollment.start(...); dữ liệu/tham số: 'alice', 'An'.
    state = service.webcam_enrollment.start("alice", "An")
    # Gán cho image: tạo frame giả cho bài kiểm thử bằng frame(...).
    image = frame()
    # Gán cho image[15:225, 50:270]: lấy phần tử/lát cắt {'dark': 5, 'bright': 250, 'blur': 125}[mode] từ {'dark': 5, 'bright': 250, 'blur': 125}.
    image[15:225, 50:270] = {"dark": 5, "bright": 250, "blur": 125}[mode]
    # Gán cho result: kiểm tra và xử lý một frame đăng ký qua service.webcam_enrollment.frame(...); dữ liệu/tham số: state['session_id'], image, 0.
    result = service.webcam_enrollment.frame(state["session_id"], image, 0)
    # Kiểm chứng điều kiện result['error']['code'] == code and result['progress'] == 0; sai thì test dừng bằng AssertionError.
    assert result["error"]["code"] == code and result["progress"] == 0


# Kiểm tra: Chụp lại giữ các góc khác và làm revision/xác nhận cũ hết hiệu lực.
# Định nghĩa hàm test_retake_one_pose_preserves_others_and_invalidates_review(service).
def test_retake_one_pose_preserves_others_and_invalidates_review(service):
    # Gán cho state: mô phỏng chụp đủ 4 góc bằng complete(...); dữ liệu/tham số: service.
    state = complete(service)
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Gán cho old: tạo từ điển khóa–giá trị bằng dict(...); dữ liệu/tham số: manager.review(state['session_id']) (lấy ảnh đã chụp để xem lại qua manager.review(...)).
    old = dict(manager.review(state["session_id"]))
    # Gán cho retake: yêu cầu chụp lại góc đã chọn qua manager.retake(...); dữ liệu/tham số: state['session_id'], 'left'.
    retake = manager.retake(state["session_id"], "left")
    # Kiểm chứng điều kiện retake['progress'] == 3 and retake['next_pose'] == 'left'; sai thì test dừng bằng AssertionError.
    assert retake["progress"] == 3 and retake["next_pose"] == "left"
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Xác nhận lưu bộ ảnh đã xem qua manager.confirm(...); dữ liệu/tham số: state['session_id'], True, state['revision'].
        manager.confirm(state["session_id"], True, state["revision"])
    # Kiểm chứng điều kiện caught.value.code == 'stale_review'; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "stale_review"
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Kiểm tra và xử lý một frame đăng ký qua manager.frame(...); dữ liệu/tham số: state['session_id'], frame(1) (tạo frame giả cho bài kiểm thử bằng frame(...)), state['revision'].
        manager.frame(state["session_id"], frame(1), state["revision"])
    # Kiểm chứng điều kiện caught.value.code == 'stale_frame'; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "stale_frame"
    # Gán cho new: mô phỏng chuỗi frame để chụp được một góc bằng capture(...); dữ liệu/tham số: service, retake, 1.
    new = capture(service, retake, 1)
    # Kiểm chứng điều kiện new['progress'] == 4; sai thì test dừng bằng AssertionError.
    assert new["progress"] == 4
    # Kiểm chứng điều kiện all((np.array_equal(image, old[pose]) for pose, image in manager.review(state['session_id']) if pose != 'left')); sai thì test dừng bằng AssertionError.
    assert all(np.array_equal(image, old[pose]) for pose, image in manager.review(state["session_id"]) if pose != "left")


# Kiểm tra: Hủy phiên loại ảnh khỏi phiên đó và không ảnh hưởng phiên khác.
# Định nghĩa hàm test_cancel_destroys_images_and_cross_session_isolation(service).
def test_cancel_destroys_images_and_cross_session_isolation(service):
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Gán cho first: mô phỏng chuỗi frame để chụp được một góc bằng capture(...); dữ liệu/tham số: service, manager.start('alice', 'An') (khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua manager.start(...)), 0.
    first = capture(service, manager.start("alice", "An"), 0)
    # Gán cho second: khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua manager.start(...); dữ liệu/tham số: 'bob', 'Bình'.
    second = manager.start("bob", "Bình")
    # Kiểm chứng điều kiện len(manager.review(first['session_id'])) == 1 and manager.review(second['session_id']) == []; sai thì test dừng bằng AssertionError.
    assert len(manager.review(first["session_id"])) == 1 and manager.review(second["session_id"]) == []
    # Hủy phiên và ảnh tạm qua manager.cancel(...); dữ liệu/tham số: first['session_id'].
    manager.cancel(first["session_id"])
    # Lặp qua (lambda: manager.status(first['session_id']), lambda: manager.frame(first['session_id'], …; mỗi lượt gán phần tử cho operation.
    for operation in (lambda: manager.status(first["session_id"]),
                      # Tiếp phần biểu thức nhiều dòng: hàm ngắn lambda nhận  và trả manager.frame(first['session_id'], frame(), 0).
                      lambda: manager.frame(first["session_id"], frame(), 0),
                      # Tiếp phần biểu thức nhiều dòng: hàm ngắn lambda nhận  và trả manager.confirm(first['session_id'], True, first['revision']).
                      lambda: manager.confirm(first["session_id"], True, first["revision"])):
        # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with pytest.raises(ApplicationError) as caught:
            # Gọi operation(...).
            operation()
        # Kiểm chứng điều kiện caught.value.code == 'session_missing'; sai thì test dừng bằng AssertionError.
        assert caught.value.code == "session_missing"
    # Kiểm chứng điều kiện db.list_people(service.path) == []; sai thì test dừng bằng AssertionError.
    assert db.list_people(service.path) == []


# Kiểm tra: Frame trống hoặc camera ngắt làm mất ổn định; tiếp tục phải giữ lại từ đầu.
# Định nghĩa hàm test_invalid_camera_disconnect_gap_and_resume(service).
def test_invalid_camera_disconnect_gap_and_resume(service):
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Gán cho state: khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua manager.start(...); dữ liệu/tham số: 'alice', 'An'.
    state = manager.start("alice", "An")
    # Gán cho state: kiểm tra và xử lý một frame đăng ký qua manager.frame(...); dữ liệu/tham số: state['session_id'], frame() (tạo frame giả cho bài kiểm thử bằng frame(...)), 0.
    state = manager.frame(state["session_id"], frame(), 0)
    # Kiểm chứng điều kiện state['stable_frames'] == 1; sai thì test dừng bằng AssertionError.
    assert state["stable_frames"] == 1
    # Gán cho result: kiểm tra và xử lý một frame đăng ký qua manager.frame(...); dữ liệu/tham số: state['session_id'], None, 0.
    result = manager.frame(state["session_id"], None, 0)
    # Kiểm chứng điều kiện result['error']['code'] == 'camera_unavailable' and result['stable_frames'] == 0; sai thì test dừng bằng AssertionError.
    assert result["error"]["code"] == "camera_unavailable" and result["stable_frames"] == 0
    # Kiểm tra và xử lý một frame đăng ký qua manager.frame(...); dữ liệu/tham số: state['session_id'], frame(sequence=1) (tạo frame giả cho bài kiểm thử bằng frame(...)), 0.
    manager.frame(state["session_id"], frame(sequence=1), 0)
    # Tăng đồng hồ giả theo số giây chỉ định qua manager.clock.advance(...); dữ liệu/tham số: 3.
    manager.clock.advance(3)
    # Kiểm chứng điều kiện manager.status(state['session_id'])['stable_frames'] == 0; sai thì test dừng bằng AssertionError.
    assert manager.status(state["session_id"])["stable_frames"] == 0
    # Cờ tạm dừng nhận ảnh; False cho phép tiếp tục xử lý frame.
    paused = manager.pause(state["session_id"])
    # Kiểm chứng điều kiện manager.frame(state['session_id'], frame(sequence=2), paused['revision'])['status'] == 'paused'; sai thì test dừng bằng AssertionError.
    assert manager.frame(state["session_id"], frame(sequence=2), paused["revision"])["status"] == "paused"
    # Gán cho resumed: tiếp tục phiên chụp qua manager.resume(...); dữ liệu/tham số: state['session_id'].
    resumed = manager.resume(state["session_id"])
    # Kiểm chứng điều kiện capture(service, resumed, 0)['progress'] == 1; sai thì test dừng bằng AssertionError.
    assert capture(service, resumed, 0)["progress"] == 1


# Kiểm tra: Giới hạn số phiên và thời gian hết hạn giải phóng chỗ đúng quy định.
# Định nghĩa hàm test_session_expiry_and_capacity(service).
def test_session_expiry_and_capacity(service):
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Gán cho manager.policy: tạo bản cấu hình sao chép với một số trường thay đổi bằng replace(...); dữ liệu/tham số: manager.policy, max_sessions=1, ttl_seconds=10.
    manager.policy = replace(manager.policy, max_sessions=1, ttl_seconds=10)
    # Gán cho state: khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua manager.start(...); dữ liệu/tham số: 'alice', 'An'.
    state = manager.start("alice", "An")
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua manager.start(...); dữ liệu/tham số: 'bob', 'Bình'.
        manager.start("bob", "Bình")
    # Kiểm chứng điều kiện caught.value.code == 'sessions_full'; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "sessions_full"
    # Tăng đồng hồ giả theo số giây chỉ định qua manager.clock.advance(...); dữ liệu/tham số: 11.
    manager.clock.advance(11)
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Lấy tiến độ phiên hiện tại qua manager.status(...); dữ liệu/tham số: state['session_id'].
        manager.status(state["session_id"])
    # Kiểm chứng điều kiện caught.value.code == 'session_missing' and (not manager.sessions); sai thì test dừng bằng AssertionError.
    assert caught.value.code == "session_missing" and not manager.sessions
    # Kiểm chứng điều kiện manager.start('bob', 'Bình')['progress'] == 0; sai thì test dừng bằng AssertionError.
    assert manager.start("bob", "Bình")["progress"] == 0


# Kiểm tra: Chưa đồng ý hoặc chưa đủ góc thì không được lưu.
# Định nghĩa hàm test_confirmation_required_and_missing_poses(service).
def test_confirmation_required_and_missing_poses(service):
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Gán cho state: khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua manager.start(...); dữ liệu/tham số: 'alice', 'An'.
    state = manager.start("alice", "An")
    # Lặp qua [(False, 'confirmation_required'), (True, 'incomplete_poses')]; mỗi lượt gán phần tử cho (accepted, code).
    for accepted, code in [(False, "confirmation_required"), (True, "incomplete_poses")]:
        # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with pytest.raises(ApplicationError) as caught:
            # Xác nhận lưu bộ ảnh đã xem qua manager.confirm(...); dữ liệu/tham số: state['session_id'], accepted, state['revision'].
            manager.confirm(state["session_id"], accepted, state["revision"])
        # Kiểm chứng điều kiện caught.value.code == code; sai thì test dừng bằng AssertionError.
        assert caught.value.code == code
    # Kiểm chứng điều kiện db.list_people(service.path) == []; sai thì test dừng bằng AssertionError.
    assert db.list_people(service.path) == []


# Kiểm tra: Ảnh của nhiều người bị chặn trước khi ghi kho chính.
# Định nghĩa hàm test_mixed_identity_rejected_before_destination_write(service).
def test_mixed_identity_rejected_before_destination_write(service):
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Gán cho state: khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua manager.start(...); dữ liệu/tham số: 'alice', 'An'.
    state = manager.start("alice", "An")
    # Lặp qua range(4); mỗi lượt gán phần tử cho pose.
    for pose in range(4):
        # Gán cho state: mô phỏng chuỗi frame để chụp được một góc bằng capture(...); dữ liệu/tham số: service, state, pose, identity=int(pose == 3).
        state = capture(service, state, pose, identity=int(pose == 3))
    # Lấy ảnh đã chụp để xem lại qua manager.review(...); dữ liệu/tham số: state['session_id'].
    manager.review(state["session_id"])
    # Gán cho before: đọc toàn bộ file dưới dạng byte qua service.path.read_bytes(...).
    before = service.path.read_bytes()
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Xác nhận lưu bộ ảnh đã xem qua manager.confirm(...); dữ liệu/tham số: state['session_id'], True, state['revision'].
        manager.confirm(state["session_id"], True, state["revision"])
    # Kiểm chứng điều kiện caught.value.code == 'identity_inconsistent'; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "identity_inconsistent"
    # Kiểm chứng điều kiện service.path.read_bytes() == before and (not metadata_path(service.path).exists()); sai thì test dừng bằng AssertionError.
    assert service.path.read_bytes() == before and not metadata_path(service.path).exists()


# Kiểm tra: Detector/mạng lỗi ở bước xác nhận không tạo hồ sơ một phần.
# Decorator parametrize: chạy cùng một bài test với nhiều bộ dữ liệu liệt kê bên dưới.
@pytest.mark.parametrize("component", ["detector", "net"])
# Định nghĩa hàm test_model_failure_no_partial_enrollment(service, component).
def test_model_failure_no_partial_enrollment(service, component):
    # Gán cho state: mô phỏng chụp đủ 4 góc bằng complete(...); dữ liệu/tham số: service.
    state = complete(service)
    # Gán cho getattr(service.runtime, component).fail: True (đúng/bật).
    getattr(service.runtime, component).fail = True
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Xác nhận lưu bộ ảnh đã xem qua service.webcam_enrollment.confirm(...); dữ liệu/tham số: state['session_id'], True, state['revision'].
        service.webcam_enrollment.confirm(state["session_id"], True, state["revision"])
    # Kiểm chứng điều kiện caught.value.code == 'pipeline_failed' and db.list_people(service.path) == []; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "pipeline_failed" and db.list_people(service.path) == []
    # Kiểm chứng điều kiện not metadata_path(service.path).exists(); sai thì test dừng bằng AssertionError.
    assert not metadata_path(service.path).exists()


# Kiểm tra: ID được đăng ký bởi lượt khác trong khi chụp vẫn được kiểm tra lại khi lưu.
# Định nghĩa hàm test_duplicate_after_session_start_preserves_existing_record(service).
def test_duplicate_after_session_start_preserves_existing_record(service):
    # Gán cho state: mô phỏng chụp đủ 4 góc bằng complete(...); dữ liệu/tham số: service.
    state = complete(service)
    # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'alice', [frame(identity=1)].
    service.enroll("alice", [frame(identity=1)])
    # Gán cho before: đọc toàn bộ file dưới dạng byte qua service.path.read_bytes(...).
    before = service.path.read_bytes()
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Xác nhận lưu bộ ảnh đã xem qua service.webcam_enrollment.confirm(...); dữ liệu/tham số: state['session_id'], True, state['revision'].
        service.webcam_enrollment.confirm(state["session_id"], True, state["revision"])
    # Kiểm chứng điều kiện caught.value.code == 'duplicate_person' and service.path.read_bytes() == before; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "duplicate_person" and service.path.read_bytes() == before
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua service.webcam_enrollment.start(...); dữ liệu/tham số: 'alice', 'Name'.
        service.webcam_enrollment.start("alice", "Name")
    # Kiểm chứng điều kiện caught.value.code == 'duplicate_person'; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "duplicate_person"


# Kiểm tra: Ghi tên lỗi phải hoàn tác cả bản ghi vector và giữ dữ liệu cũ.
# Định nghĩa hàm test_sidecar_failure_rolls_back_main_insert_and_existing_data(service, monkeypatch).
def test_sidecar_failure_rolls_back_main_insert_and_existing_data(service, monkeypatch):
    # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'existing', [frame(identity=1)].
    service.enroll("existing", [frame(identity=1)])
    # Gán cho state: mô phỏng chụp đủ 4 góc bằng complete(...); dữ liệu/tham số: service.
    state = complete(service)
    # Cố ý để tên cũ tồn tại nhằm làm lệnh INSERT thứ hai thất bại.
    # Gán cho names: xác định file SQLite chứa tên đi kèm bằng metadata_path(...); dữ liệu/tham số: service.path.
    names = metadata_path(service.path)
    # Lấy NAME_SQL từ application.webcam_registry.
    from application.webcam_registry import NAME_SQL
    # Mở khối with để bảo đảm tài nguyên được close khi kết thúc khối with bằng closing(...); dữ liệu/tham số: sqlite3.connect(names) (mở kết nối cơ sở dữ liệu SQLite bằng sqlite3.connect(...)); gọi tài nguyên là conn; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with closing(sqlite3.connect(names)) as conn:
        # Thực hiện câu lệnh SQL và truyền các tham số đi kèm qua conn.execute(...); dữ liệu/tham số: NAME_SQL.
        conn.execute(NAME_SQL)
        # Thực hiện câu lệnh SQL và truyền các tham số đi kèm qua conn.execute(...); dữ liệu/tham số: "INSERT INTO EnrollmentNames VALUES ('alice','old')".
        conn.execute("INSERT INTO EnrollmentNames VALUES ('alice','old')")
        # Xác nhận lưu toàn bộ thay đổi của giao dịch qua conn.commit(...).
        conn.commit()
    # Tách kết quả theo thứ tự và gán cho (before_main, before_names): bộ giá trị (service.path.read_bytes(), names.read_bytes()) (tuple giữ thứ tự, không gán lại từng phần tử).
    before_main, before_names = service.path.read_bytes(), names.read_bytes()
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Xác nhận lưu bộ ảnh đã xem qua service.webcam_enrollment.confirm(...); dữ liệu/tham số: state['session_id'], True, state['revision'].
        service.webcam_enrollment.confirm(state["session_id"], True, state["revision"])
    # Kiểm chứng điều kiện caught.value.code == 'pipeline_failed'; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "pipeline_failed"
    # Kiểm chứng điều kiện service.path.read_bytes() == before_main and names.read_bytes() == before_names; sai thì test dừng bằng AssertionError.
    assert service.path.read_bytes() == before_main and names.read_bytes() == before_names
    # Kiểm chứng điều kiện [p['person_id'] for p in db.list_people(service.path)] == ['existing']; sai thì test dừng bằng AssertionError.
    assert [p["person_id"] for p in db.list_people(service.path)] == ["existing"]


# Kiểm tra: Kho bị hỏng lúc xác nhận được báo lỗi và không tự sửa.
# Định nghĩa hàm test_corrupt_database_at_confirmation_no_repair(service).
def test_corrupt_database_at_confirmation_no_repair(service):
    # Gán cho state: mô phỏng chụp đủ 4 góc bằng complete(...); dữ liệu/tham số: service.
    state = complete(service)
    # Ghi byte xuống file qua service.path.write_bytes(...); dữ liệu/tham số: b'corrupt'.
    service.path.write_bytes(b"corrupt")
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Xác nhận lưu bộ ảnh đã xem qua service.webcam_enrollment.confirm(...); dữ liệu/tham số: state['session_id'], True, state['revision'].
        service.webcam_enrollment.confirm(state["session_id"], True, state["revision"])
    # Kiểm chứng điều kiện caught.value.code == 'registry_invalid' and service.path.read_bytes() == b'corrupt'; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "registry_invalid" and service.path.read_bytes() == b"corrupt"


# Kiểm tra: Giao dịch kiểm tra lại ID trùng và ID dành riêng trước khi chèn.
# Định nghĩa hàm test_transaction_rechecks_duplicate_and_reserved_rows(service).
def test_transaction_rechecks_duplicate_and_reserved_rows(service):
    # Gán cho embedding: lấy phần tử/lát cắt np.eye(1, 512, dtype=np.float32)[0] từ np.eye(1, 512, dtype=np.float32).
    embedding = np.eye(1, 512, dtype=np.float32)[0]
    # Lưu vector và tên trong cùng giao dịch bằng commit_identity(...); dữ liệu/tham số: service.path, 'alice', 'An', num_images=4.
    commit_identity(service.path, "alice", "An", embedding, service.runtime.version, num_images=4)
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Lưu vector và tên trong cùng giao dịch bằng commit_identity(...); dữ liệu/tham số: service.path, 'alice', 'Changed', num_images=4.
        commit_identity(service.path, "alice", "Changed", embedding, service.runtime.version, num_images=4)
    # Kiểm chứng điều kiện caught.value.code == 'duplicate_person' and read_names(service.path)['alice'] == 'An'; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "duplicate_person" and read_names(service.path)["alice"] == "An"
    # Gọi db.add_person(...) với service.path, 'Unknown', embedding; dữ liệu/tham số: service.path, 'Unknown', embedding.
    db.add_person(service.path, "Unknown", embedding, 1, service.runtime.version)
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ValueError, match='reserved'; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ValueError, match="reserved"):
        # Lưu vector và tên trong cùng giao dịch bằng commit_identity(...); dữ liệu/tham số: service.path, 'bob', 'Bình', num_images=4.
        commit_identity(service.path, "bob", "Bình", embedding, service.runtime.version, num_images=4)
    # Kiểm chứng điều kiện db.get_person(service.path, 'bob', service.runtime.version) is None; sai thì test dừng bằng AssertionError.
    assert db.get_person(service.path, "bob", service.runtime.version) is None


# Kiểm tra: Kho ở chế độ WAL bị từ chối vì luồng này yêu cầu giao dịch hai file ở chế độ DELETE.
# Định nghĩa hàm test_wal_registry_refused_without_identity_write(service).
def test_wal_registry_refused_without_identity_write(service):
    # Mở khối with để bảo đảm tài nguyên được close khi kết thúc khối with bằng closing(...); dữ liệu/tham số: db.connect_database(service.path) (gọi db.connect_database(...) với service.path); gọi tài nguyên là conn; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with closing(db.connect_database(service.path)) as conn:
        # Thực hiện câu lệnh SQL và truyền các tham số đi kèm qua conn.execute(...); dữ liệu/tham số: 'PRAGMA journal_mode=WAL'.
        conn.execute("PRAGMA journal_mode=WAL")
    # Gán cho embedding: lấy phần tử/lát cắt np.eye(1, 512, dtype=np.float32)[0] từ np.eye(1, 512, dtype=np.float32).
    embedding = np.eye(1, 512, dtype=np.float32)[0]
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ValueError, match='DELETE'; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ValueError, match="DELETE"):
        # Lưu vector và tên trong cùng giao dịch bằng commit_identity(...); dữ liệu/tham số: service.path, 'alice', 'An', num_images=4.
        commit_identity(service.path, "alice", "An", embedding, service.runtime.version, num_images=4)
    # Kiểm chứng điều kiện db.list_people(service.path) == []; sai thì test dừng bằng AssertionError.
    assert db.list_people(service.path) == []


# Kiểm tra: Hai lượt xác nhận đồng thời chỉ tạo được một hồ sơ và kết thúc phiên một lần.
# Định nghĩa hàm test_concurrent_confirm_and_cancel_serialize(service).
def test_concurrent_confirm_and_cancel_serialize(service):
    # Gán cho state: mô phỏng chụp đủ 4 góc bằng complete(...); dữ liệu/tham số: service.
    state = complete(service)
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Xác minh bộ ảnh đã xem, kiểm tra cùng danh tính trong kho tạm rồi lưu kho chính.
    # Định nghĩa hàm confirm(_).
    def confirm(_):
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Kết thúc hàm và trả lấy phần tử/lát cắt manager.confirm(state['session_id'], True, state['revision'])['status'] từ manager.confirm(state['session_id'], True, state['revision']).
            return manager.confirm(state["session_id"], True, state["revision"])["status"]
        # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
        except ApplicationError as error:
            # Kết thúc hàm và trả error.code (đọc thuộc tính).
            return error.code
    # Mở khối with để tạo nhóm luồng để chạy công việc đồng thời bằng ThreadPoolExecutor(...); dữ liệu/tham số: max_workers=2; gọi tài nguyên là executor; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Gán cho outcomes: tạo danh sách bằng list(...); dữ liệu/tham số: executor.map(confirm, range(2)) (gọi executor.map(...) với confirm, range(2)).
        outcomes = list(executor.map(confirm, range(2)))
    # Kiểm chứng điều kiện sorted(outcomes) == ['enrolled', 'session_missing']; sai thì test dừng bằng AssertionError.
    assert sorted(outcomes) == ["enrolled", "session_missing"]
    # Kiểm chứng điều kiện len(db.list_people(service.path)) == 1; sai thì test dừng bằng AssertionError.
    assert len(db.list_people(service.path)) == 1


# Kiểm tra: Kiểm tra API từ tạo phiên, chụp 4 góc, xem ảnh đến xác nhận và giới hạn tư thế.
# Định nghĩa hàm test_api_full_session_review_retake_confirm_and_limits(service).
def test_api_full_session_review_retake_confirm_and_limits(service):
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Mở khối with để tạo trình gọi API nội bộ cho kiểm thử bằng TestClient(...); dữ liệu/tham số: create_app(service) (tạo API và tùy chọn gắn giao diện bằng create_app(...)); gọi tài nguyên là client; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with TestClient(create_app(service)) as client:
        # Gán cho state: đọc JSON từ phản hồi HTTP qua client.post('/enrollment-sessions', json={'person_id': 'alice', 'display_name': 'An'}).js…(...).
        state = client.post("/enrollment-sessions", json={"person_id": "alice", "display_name": "An"}).json()
        # Gán cho token: lấy phần tử/lát cắt state['session_id'] từ state.
        token = state["session_id"]
        # Gán cho root: chuỗi f-string: chèn giá trị biểu thức nằm trong {...} vào nội dung.
        root = f"/enrollment-sessions/{token}"
        # Lặp qua range(4); mỗi lượt gán phần tử cho pose.
        for pose in range(4):
            # Lặp qua range(4); mỗi lượt gán phần tử cho seq.
            for seq in range(4):
                # Tăng đồng hồ giả theo số giây chỉ định qua manager.clock.advance(...).
                manager.clock.advance()
                # Gán cho response: gửi yêu cầu HTTP POST qua client.post(...); dữ liệu/tham số: root + '/frames', data={'revision': state['revision']}, files={'image': ('frame.png', png(frame(pose, sequence=seq)), 'image/png')}.
                response = client.post(root + "/frames", data={"revision": state["revision"]},
                    # Truyền files = {'image': ('frame.png', png(frame(pose, sequence=seq)), 'image/png')}.
                    files={"image": ("frame.png", png(frame(pose, sequence=seq)), "image/png")})
                # Kiểm chứng điều kiện response.status_code == 200; sai thì test dừng bằng AssertionError.
                assert response.status_code == 200, response.text
                # Gán cho state: đọc JSON từ phản hồi HTTP qua response.json(...).
                state = response.json()
        # Kiểm chứng điều kiện state['progress'] == state['total'] == 4; sai thì test dừng bằng AssertionError.
        assert state["progress"] == state["total"] == 4
        # Kiểm chứng điều kiện state['next_pose'] is None and state['status'] == 'review'; sai thì test dừng bằng AssertionError.
        assert state["next_pose"] is None and state["status"] == "review"
        # Gán cho review: lấy phần tử/lát cắt client.get(root + '/review').json()['images'] từ client.get(root + '/review').json().
        review = client.get(root + "/review").json()["images"]
        # Kiểm chứng điều kiện [item['pose'] for item in review] == ['front', 'left', 'right', 'up']; sai thì test dừng bằng AssertionError.
        assert [item["pose"] for item in review] == ["front", "left", "right", "up"]
        # Kiểm chứng điều kiện all((item['image'].startswith('data:image/jpeg;base64,') for item in review)); sai thì test dừng bằng AssertionError.
        assert all(item["image"].startswith("data:image/jpeg;base64,") for item in review)
        # Kiểm chứng điều kiện client.post(root + '/retake/down').status_code == 422; sai thì test dừng bằng AssertionError.
        assert client.post(root + "/retake/down").status_code == 422
        # Kiểm chứng điều kiện client.post(root + '/confirm', json={'confirmed': False, 'revision': state['revision']}).status_code == 422; sai thì test dừng bằng AssertionError.
        assert client.post(root + "/confirm", json={"confirmed": False, "revision": state["revision"]}).status_code == 422
        # Kiểm chứng điều kiện client.post(root + '/confirm', json={'confirmed': True, 'revision': state['revision']}).status_code == 201; sai thì test dừng bằng AssertionError.
        assert client.post(root + "/confirm", json={"confirmed": True, "revision": state["revision"]}).status_code == 201
        # Kiểm chứng điều kiện client.get(root).status_code == 404; sai thì test dừng bằng AssertionError.
        assert client.get(root).status_code == 404
        # Kiểm chứng điều kiện client.get('/persons').json()['persons'][0]['display_name'] == 'An'; sai thì test dừng bằng AssertionError.
        assert client.get("/persons").json()["persons"][0]["display_name"] == "An"


# Kiểm tra: Upload ảnh lỗi làm mất ổn định; phiên đã hủy không còn truy cập được.
# Định nghĩa hàm test_api_invalid_frame_breaks_stability_cancel_and_missing_token(service).
def test_api_invalid_frame_breaks_stability_cancel_and_missing_token(service):
    # Mở khối with để tạo trình gọi API nội bộ cho kiểm thử bằng TestClient(...); dữ liệu/tham số: create_app(service) (tạo API và tùy chọn gắn giao diện bằng create_app(...)); gọi tài nguyên là client; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with TestClient(create_app(service)) as client:
        # Gán cho state: đọc JSON từ phản hồi HTTP qua client.post('/enrollment-sessions', json={'person_id': 'alice', 'display_name': 'An'}).js…(...).
        state = client.post("/enrollment-sessions", json={"person_id": "alice", "display_name": "An"}).json()
        # Gán cho root: chuỗi f-string: chèn giá trị biểu thức nằm trong {...} vào nội dung.
        root = f'/enrollment-sessions/{state["session_id"]}'
        # Kiểm tra và xử lý một frame đăng ký qua service.webcam_enrollment.frame(...); dữ liệu/tham số: state['session_id'], frame() (tạo frame giả cho bài kiểm thử bằng frame(...)), 0.
        service.webcam_enrollment.frame(state["session_id"], frame(), 0)
        # Gán cho response: gửi yêu cầu HTTP POST qua client.post(...); dữ liệu/tham số: root + '/frames', data={'revision': 0}, files={'image': ('bad.png', b'bad', 'image/png')}.
        response = client.post(root + "/frames", data={"revision": 0}, files={"image": ("bad.png", b"bad", "image/png")})
        # Kiểm chứng điều kiện response.status_code == 422 and client.get(root).json()['stable_frames'] == 0; sai thì test dừng bằng AssertionError.
        assert response.status_code == 422 and client.get(root).json()["stable_frames"] == 0
        # Kiểm chứng điều kiện client.post(root + '/pause').json()['status'] == 'paused'; sai thì test dừng bằng AssertionError.
        assert client.post(root + "/pause").json()["status"] == "paused"
        # Kiểm chứng điều kiện client.post(root + '/resume').json()['status'] == 'capturing'; sai thì test dừng bằng AssertionError.
        assert client.post(root + "/resume").json()["status"] == "capturing"
        # Kiểm chứng điều kiện client.delete(root).json()['status'] == 'cancelled'; sai thì test dừng bằng AssertionError.
        assert client.delete(root).json()["status"] == "cancelled"
        # Kiểm chứng điều kiện client.post(root + '/frames', data={'revision': 0}, files={'image': ('frame.png', png(frame()), 'image/png')}).status_code == 404; sai thì test dừng bằng AssertionError.
        assert client.post(root + "/frames", data={"revision": 0}, files={"image": ("frame.png", png(frame()), "image/png")}).status_code == 404


# Kiểm tra: Callback giao diện cập nhật tiến độ, ảnh xem lại, nút lưu và trạng thái hủy.
# Định nghĩa hàm test_ui_enrollment_callbacks_review_confirmation_and_cancel(service).
def test_ui_enrollment_callbacks_review_confirmation_and_cancel(service):
    # Lấy build_ui từ application.ui.
    from application.ui import build_ui
    # Gán cho demo: dựng giao diện Gradio bằng build_ui(...); dữ liệu/tham số: service.
    demo = build_ui(service)
    # Gán cho callbacks: tạo từ điển bằng vòng lặp rút gọn; khóa là f.fn.__name__, giá trị là f.fn.
    callbacks = {f.fn.__name__: f.fn for f in demo.fns.values() if hasattr(f.fn, "__name__")}
    # Tách kết quả theo thứ tự và gán cho (state, message, _, _, _, save, _): gọi callbacks['start_enrollment'](...) với 'alice', 'An', {}; dữ liệu/tham số: 'alice', 'An', {}.
    state, message, _, _, _, save, _ = callbacks["start_enrollment"]("alice", "An", {})
    # Kiểm chứng điều kiện '0/4' in message and save['interactive'] is False; sai thì test dừng bằng AssertionError.
    assert "0/4" in message and save["interactive"] is False
    # Lặp qua range(4); mỗi lượt gán phần tử cho pose.
    for pose in range(4):
        # Lặp qua range(4); mỗi lượt gán phần tử cho seq.
        for seq in range(4):
            # Tăng đồng hồ giả theo số giây chỉ định qua service.webcam_enrollment.clock.advance(...).
            service.webcam_enrollment.clock.advance()
            # Gán cho output: gọi callbacks['enrollment_stream'](...) với frame(pose, sequence=seq), state; dữ liệu/tham số: frame(pose, sequence=seq) (tạo frame giả cho bài kiểm thử bằng frame(...)), state.
            output = callbacks["enrollment_stream"](frame(pose, sequence=seq), state)
            # Gán cho state: lấy phần tử/lát cắt output[0] từ output.
            state = output[0]
    # Kiểm chứng điều kiện '4/4' in output[1] and output[3].count('data:image/jpeg;base64,') == 4; sai thì test dừng bằng AssertionError.
    assert "4/4" in output[1] and output[3].count("data:image/jpeg;base64,") == 4
    # Kiểm chứng điều kiện output[5]['interactive'] is True; sai thì test dừng bằng AssertionError.
    assert output[5]["interactive"] is True
    # Kiểm chứng điều kiện callbacks['save_enrollment'](state, False)[2]['error']['code'] == 'confirmation_required'; sai thì test dừng bằng AssertionError.
    assert callbacks["save_enrollment"](state, False)[2]["error"]["code"] == "confirmation_required"
    # Kiểm chứng điều kiện callbacks['cancel_enrollment'](state)[0] == {} and db.list_people(service.path) == []; sai thì test dừng bằng AssertionError.
    assert callbacks["cancel_enrollment"](state)[0] == {} and db.list_people(service.path) == []


# Kiểm tra: Sau khi chụp lại phải xem đúng bộ ảnh mới nhất trước khi lưu.
# Định nghĩa hàm test_latest_review_required_after_retake(service).
def test_latest_review_required_after_retake(service):
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Gán cho state: mô phỏng chụp đủ 4 góc bằng complete(...); dữ liệu/tham số: service.
    state = complete(service)
    # Gán cho old_review: lấy ảnh xem lại kèm revision qua manager.review_snapshot(...); dữ liệu/tham số: state['session_id'].
    old_review = manager.review_snapshot(state["session_id"])
    # Gán cho state: yêu cầu chụp lại góc đã chọn qua manager.retake(...); dữ liệu/tham số: state['session_id'], 'left'.
    state = manager.retake(state["session_id"], "left")
    # Gán cho state: mô phỏng chuỗi frame để chụp được một góc bằng capture(...); dữ liệu/tham số: service, state, 1.
    state = capture(service, state, 1)
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Xác nhận lưu bộ ảnh đã xem qua manager.confirm(...); dữ liệu/tham số: state['session_id'], True, state['revision'].
        manager.confirm(state["session_id"], True, state["revision"])
    # Kiểm chứng điều kiện caught.value.code == 'review_required'; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "review_required"
    # Gán cho new_review: lấy ảnh xem lại kèm revision qua manager.review_snapshot(...); dữ liệu/tham số: state['session_id'].
    new_review = manager.review_snapshot(state["session_id"])
    # Kiểm chứng điều kiện new_review['revision'] > old_review['revision']; sai thì test dừng bằng AssertionError.
    assert new_review["revision"] > old_review["revision"]
    # Kiểm chứng điều kiện manager.confirm(state['session_id'], True, new_review['revision'])['status'] == 'enrolled'; sai thì test dừng bằng AssertionError.
    assert manager.confirm(state["session_id"], True, new_review["revision"])["status"] == "enrolled"
    # Kiểm chứng điều kiện manager.cancel(state['session_id'])['status'] == 'session_ended'; sai thì test dừng bằng AssertionError.
    assert manager.cancel(state["session_id"])["status"] == "session_ended"


# Kiểm tra: File tên rỗng do giao dịch đầu thất bại không làm hỏng những lần dùng sau.
# Định nghĩa hàm test_failed_first_commit_empty_sidecar_does_not_poison_registry(service).
def test_failed_first_commit_empty_sidecar_does_not_poison_registry(service):
    # Gán cho embedding: lấy phần tử/lát cắt np.eye(1, 512, dtype=np.float32)[0] từ np.eye(1, 512, dtype=np.float32).
    embedding = np.eye(1, 512, dtype=np.float32)[0]
    # Cố ý gây lỗi kiểm tra lại giao dịch sau khi ATTACH đã tạo file tên.
    # Gọi db.add_person(...) với service.path, 'existing', embedding; dữ liệu/tham số: service.path, 'existing', embedding.
    db.add_person(service.path, "existing", embedding, 1, "changed-version")
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ValueError, match='model_version'; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ValueError, match="model_version"):
        # Lưu vector và tên trong cùng giao dịch bằng commit_identity(...); dữ liệu/tham số: service.path, 'alice', 'An', num_images=4.
        commit_identity(service.path, "alice", "An", embedding, service.runtime.version, num_images=4)
    # Kiểm chứng điều kiện metadata_path(service.path).exists() and read_names(service.path) == {}; sai thì test dừng bằng AssertionError.
    assert metadata_path(service.path).exists() and read_names(service.path) == {}
    # Lấy delete_identity từ application.webcam_registry.
    from application.webcam_registry import delete_identity
    # Xóa đồng bộ vector và tên theo ID bằng delete_identity(...); dữ liệu/tham số: service.path, 'existing'.
    delete_identity(service.path, "existing")
    # Kiểm chứng điều kiện service.persons() == []; sai thì test dừng bằng AssertionError.
    assert service.persons() == []
    # Lưu vector và tên trong cùng giao dịch bằng commit_identity(...); dữ liệu/tham số: service.path, 'alice', 'An', num_images=4.
    commit_identity(service.path, "alice", "An", embedding, service.runtime.version, num_images=4)
    # Kiểm chứng điều kiện read_names(service.path) == {'alice': 'An'}; sai thì test dừng bằng AssertionError.
    assert read_names(service.path) == {"alice": "An"}


# Kiểm tra: Đủ số frame nhưng chưa giữ đủ thời gian vẫn chưa được chụp.
# Định nghĩa hàm test_stable_frame_count_alone_does_not_bypass_hold_time(service).
def test_stable_frame_count_alone_does_not_bypass_hold_time(service):
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Gán cho state: khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua manager.start(...); dữ liệu/tham số: 'alice', 'An'.
    state = manager.start("alice", "An")
    # Lặp qua range(8); mỗi lượt gán phần tử cho seq.
    for seq in range(8):
        # Tăng đồng hồ giả theo số giây chỉ định qua manager.clock.advance(...); dữ liệu/tham số: 0.01.
        manager.clock.advance(.01)
        # Gán cho result: kiểm tra và xử lý một frame đăng ký qua manager.frame(...); dữ liệu/tham số: state['session_id'], frame(sequence=seq) (tạo frame giả cho bài kiểm thử bằng frame(...)), 0.
        result = manager.frame(state["session_id"], frame(sequence=seq), 0)
    # Kiểm chứng điều kiện result['stable_frames'] == 4 and result['progress'] == 0; sai thì test dừng bằng AssertionError.
    assert result["stable_frames"] == 4 and result["progress"] == 0


# Kiểm tra: Góc đầu dao động vượt ngưỡng khiến chuỗi ổn định bị đặt lại.
# Định nghĩa hàm test_head_jitter_resets_stability(service, monkeypatch).
def test_head_jitter_resets_stability(service, monkeypatch):
    # Lấy HeadPose từ application.head_pose.
    from application.head_pose import HeadPose
    # Nạp application.webcam_enrollment dưới tên ngắn module để dùng các hàm của thư viện.
    import application.webcam_enrollment as module
    # Gán cho manager: service.webcam_enrollment (đọc thuộc tính).
    manager = service.webcam_enrollment
    # Gán cho state: khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua manager.start(...); dữ liệu/tham số: 'alice', 'An'.
    state = manager.start("alice", "An")
    # Lặp qua enumerate([-4, 4, -4, 4, -4, 4]); mỗi lượt gán phần tử cho (seq, yaw).
    for seq, yaw in enumerate([-4, 4, -4, 4, -4, 4]):
        # Gọi monkeypatch.setattr(...) với module, 'estimate_head_pose', lambda *_, angle=yaw: HeadPose(ang…; dữ liệu/tham số: module, 'estimate_head_pose', lambda *_, angle=yaw: HeadPose(angle, 0, 0).
        monkeypatch.setattr(module, "estimate_head_pose", lambda *_, angle=yaw: HeadPose(angle, 0, 0))
        # Tăng đồng hồ giả theo số giây chỉ định qua manager.clock.advance(...).
        manager.clock.advance()
        # Gán cho result: kiểm tra và xử lý một frame đăng ký qua manager.frame(...); dữ liệu/tham số: state['session_id'], frame(sequence=seq) (tạo frame giả cho bài kiểm thử bằng frame(...)), 0.
        result = manager.frame(state["session_id"], frame(sequence=seq), 0)
    # Kiểm chứng điều kiện result['stable_frames'] == 1 and result['progress'] == 0; sai thì test dừng bằng AssertionError.
    assert result["stable_frames"] == 1 and result["progress"] == 0


# Kiểm tra: Đầu ra detector sai cấu trúc được coi là lỗi pipeline.
# Định nghĩa hàm test_invalid_model_detection_is_infrastructure_error(service, monkeypatch).
def test_invalid_model_detection_is_infrastructure_error(service, monkeypatch):
    # Gán cho state: khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua service.webcam_enrollment.start(...); dữ liệu/tham số: 'alice', 'An'.
    state = service.webcam_enrollment.start("alice", "An")
    # Gọi monkeypatch.setattr(...) với service.runtime.detector, 'detect', lambda image: SimpleNamespace(boxe…; dữ liệu/tham số: service.runtime.detector, 'detect', lambda image: SimpleNamespace(boxes=None).
    monkeypatch.setattr(service.runtime.detector, "detect", lambda image: SimpleNamespace(boxes=None))
    # Mở khối with để yêu cầu đoạn mã trong khối phải phát sinh kiểu lỗi chỉ định qua pytest.raises(...); dữ liệu/tham số: ApplicationError; gọi tài nguyên là caught; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with pytest.raises(ApplicationError) as caught:
        # Kiểm tra và xử lý một frame đăng ký qua service.webcam_enrollment.frame(...); dữ liệu/tham số: state['session_id'], frame() (tạo frame giả cho bài kiểm thử bằng frame(...)), 0.
        service.webcam_enrollment.frame(state["session_id"], frame(), 0)
    # Kiểm chứng điều kiện caught.value.code == 'pipeline_failed'; sai thì test dừng bằng AssertionError.
    assert caught.value.code == "pipeline_failed"


# Kiểm tra: Luồng frame dùng nhóm hàng đợi riêng để các nút lưu/hủy vẫn được xử lý.
# Định nghĩa hàm test_ui_stream_and_controls_have_independent_queue_slots(service).
def test_ui_stream_and_controls_have_independent_queue_slots(service):
    # Lấy build_ui từ application.ui.
    from application.ui import build_ui
    # Gán cho demo: dựng giao diện Gradio bằng build_ui(...); dữ liệu/tham số: service.
    demo = build_ui(service)
    # Gán cho functions: tạo từ điển bằng vòng lặp rút gọn; khóa là getattr(f.fn, '__name__', ''), giá trị là f.
    functions = {getattr(f.fn, "__name__", ""): f for f in demo.fns.values()}
    # Kiểm chứng điều kiện functions['enrollment_stream'].concurrency_id != functions['save_enrollment'].concurrency_id; sai thì test dừng bằng AssertionError.
    assert functions["enrollment_stream"].concurrency_id != functions["save_enrollment"].concurrency_id
    # Kiểm chứng điều kiện functions['enrollment_stream'].concurrency_id != functions['cancel_enrollment'].concurrency_id; sai thì test dừng bằng AssertionError.
    assert functions["enrollment_stream"].concurrency_id != functions["cancel_enrollment"].concurrency_id


# Kiểm tra: Frame đến trễ làm giao diện đồng bộ revision mới nhất.
# Định nghĩa hàm test_ui_recovers_latest_revision_for_stale_inflight_frame(service).
def test_ui_recovers_latest_revision_for_stale_inflight_frame(service):
    # Lấy build_ui từ application.ui.
    from application.ui import build_ui
    # Gán cho demo: dựng giao diện Gradio bằng build_ui(...); dữ liệu/tham số: service.
    demo = build_ui(service)
    # Gán cho callbacks: tạo từ điển bằng vòng lặp rút gọn; khóa là getattr(f.fn, '__name__', ''), giá trị là f.fn.
    callbacks = {getattr(f.fn, "__name__", ""): f.fn for f in demo.fns.values()}
    # Gán cho state: khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua service.webcam_enrollment.start(...); dữ liệu/tham số: 'alice', 'An'.
    state = service.webcam_enrollment.start("alice", "An")
    # Gán cho old: từ điển {'session_id': state['session_id'], 'revision': state['revision']} (ánh xạ khóa → giá trị).
    old = {"session_id": state["session_id"], "revision": state["revision"]}
    # Gán cho state: mô phỏng chuỗi frame để chụp được một góc bằng capture(...); dữ liệu/tham số: service, state, 0.
    state = capture(service, state, 0)
    # Gán cho result: gọi callbacks['enrollment_stream'](...) với frame(1), old; dữ liệu/tham số: frame(1) (tạo frame giả cho bài kiểm thử bằng frame(...)), old.
    result = callbacks["enrollment_stream"](frame(1), old)
    # Kiểm chứng điều kiện result[0]['revision'] == state['revision'] and result[2]['progress'] == 1; sai thì test dừng bằng AssertionError.
    assert result[0]["revision"] == state["revision"] and result[2]["progress"] == 1


# Kiểm tra: Phản hồi stream đến sau khi phiên kết thúc không ghi đè thông báo của nút điều khiển.
# Định nghĩa hàm test_ui_late_stream_does_not_overwrite_completed_control_message(service).
def test_ui_late_stream_does_not_overwrite_completed_control_message(service):
    # Lấy build_ui từ application.ui.
    from application.ui import build_ui
    # Gán cho demo: dựng giao diện Gradio bằng build_ui(...); dữ liệu/tham số: service.
    demo = build_ui(service)
    # Gán cho callback: lấy phần tử tiếp theo, hoặc giá trị mặc định nếu đã hết bằng next(...); dữ liệu/tham số: (f.fn for f in demo.fns.values() if getattr(f.fn, '__name__', '') == 'enrollment_stream').
    callback = next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "enrollment_stream")
    # Gán cho state: khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua service.webcam_enrollment.start(...); dữ liệu/tham số: 'alice', 'An'.
    state = service.webcam_enrollment.start("alice", "An")
    # Hủy phiên và ảnh tạm qua service.webcam_enrollment.cancel(...); dữ liệu/tham số: state['session_id'].
    service.webcam_enrollment.cancel(state["session_id"])
    # Gán cho result: gọi callback(...) với frame(), {'session_id': state['session_id']…; dữ liệu/tham số: frame() (tạo frame giả cho bài kiểm thử bằng frame(...)), {'session_id': state['session_id'], 'revision': 0}.
    result = callback(frame(), {"session_id": state["session_id"], "revision": 0})
    # Kiểm chứng điều kiện all((output == {'__type__': 'update'} for output in result)); sai thì test dừng bằng AssertionError.
    assert all(output == {"__type__": "update"} for output in result)
