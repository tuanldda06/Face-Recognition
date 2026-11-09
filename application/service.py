# Tầng điều phối: MTCNN tìm/căn chỉnh mặt, FaceNet tạo vector đặc trưng, SQLite lưu hồ sơ.
# Lấy contextmanager từ contextlib để quản lý tài nguyên bằng with, yield và thao tác dọn tự động.
from contextlib import contextmanager
# Lấy dataclass từ dataclasses để tạo lớp dữ liệu và sao chép/chuyển đổi dữ liệu của lớp.
from dataclasses import dataclass
# Nạp hashlib để tính mã băm để kiểm tra nội dung file/ảnh/vector.
import hashlib
# Nạp logging để ghi thông tin lỗi phục vụ chẩn đoán.
import logging
# Nạp os để làm việc với biến môi trường và thông tin hệ điều hành.
import os
# Lấy Path từ pathlib để biểu diễn đường dẫn và thao tác file bằng đối tượng Path.
from pathlib import Path
# Nạp tempfile để tạo file/thư mục tạm có kiểm soát vòng đời.
import tempfile
# Lấy RLock từ threading để đồng bộ các luồng bằng khóa.
from threading import RLock
# Lấy perf_counter từ time để đo thời gian xử lý hoặc mô phỏng độ trễ.
from time import perf_counter

# Lấy FileLock, Timeout từ filelock để khóa kho hồ sơ dùng chung giữa các tiến trình.
from filelock import FileLock, Timeout
# Nạp numpy dưới tên ngắn np để tính toán mảng ảnh, vector và ma trận.
import numpy as np

# Lấy build_model_version, enroll_person_from_images, load_facenet_model, recognize_image từ FaceNet.
from FaceNet import (build_model_version, enroll_person_from_images,
                     # Tiếp danh sách import từ FaceNet: lấy hàm nạp mạng và hàm nhận diện ảnh; dấu ) đóng danh sách.
                     load_facenet_model, recognize_image)
# Lấy database với bí danh db từ FaceNet.
from FaceNet import database as db
# Lấy validate_person_id từ FaceNet.matching.
from FaceNet.matching import validate_person_id

# Lấy verify_model từ .assets; dấu . chỉ module cùng package.
from .assets import verify_model
# Lấy MODEL_PATH, ROOT, Settings từ .config; dấu . chỉ module cùng package.
from .config import MODEL_PATH, ROOT, Settings
# Lấy ApplicationError từ .errors; dấu . chỉ module cùng package.
from .errors import ApplicationError
# Lấy validate_rgb từ .images; dấu . chỉ module cùng package.
from .images import validate_rgb

# Gán cho logger: gọi logging.getLogger(...) với __name__; dữ liệu/tham số: __name__.
logger = logging.getLogger(__name__)


# Gom bộ phát hiện mặt, mạng FaceNet và phiên bản model đang chạy.
# Decorator dataclass: tự sinh hàm khởi tạo từ các trường; frozen=True chặn gán lại trường nếu được bật.
@dataclass
# Định nghĩa lớp Runtime; lớp gom dữ liệu và các hàm liên quan.
class Runtime:
    # Khai báo detector có kiểu gợi ý object; giá trị được cung cấp khi tạo đối tượng.
    detector: object
    # Khai báo net có kiểu gợi ý object; giá trị được cung cấp khi tạo đối tượng.
    net: object
    # Khai báo version có kiểu gợi ý str; giá trị được cung cấp khi tạo đối tượng.
    version: str


# Kiểm tra model và nạp MTCNN/FaceNet trên CPU; giới hạn luồng để kiểm soát tài nguyên.
# Định nghĩa hàm load_runtime(settings).
def load_runtime(settings: Settings):
    # Xác minh file model bằng dung lượng và SHA-256 bằng verify_model(...).
    verify_model()
    # Nạp cv2 để xử lý ảnh, chạy ONNX và tính toán hình học camera bằng OpenCV.
    import cv2
    # Nạp torch để chạy các mạng MTCNN bằng PyTorch.
    import torch
    # Lấy MTCNN từ MTCNN.mtcnn.
    from MTCNN.mtcnn import MTCNN
    # Đặt số luồng CPU cho PyTorch bằng torch.set_num_threads(...); dữ liệu/tham số: settings.cpu_threads.
    torch.set_num_threads(settings.cpu_threads)
    # Đặt số luồng CPU cho OpenCV bằng cv2.setNumThreads(...); dữ liệu/tham số: settings.cpu_threads.
    cv2.setNumThreads(settings.cpu_threads)
    # Gán cho detector: gọi MTCNN(...); dữ liệu/tham số: device='cpu'.
    detector = MTCNN(device="cpu")
    # Gán cho net: nạp mạng FaceNet từ file ONNX bằng load_facenet_model(...); dữ liệu/tham số: MODEL_PATH.
    net = load_facenet_model(MODEL_PATH)
    # Kết thúc hàm và trả gom detector, mạng và phiên bản model bằng Runtime(...); dữ liệu/tham số: detector, net, build_model_version(detector) (tạo thông tin phiên bản model và môi trường chạy bằng build_model_version(...)).
    return Runtime(detector, net, build_model_version(detector))


# Bọc detector để kiểm tra số mặt trước khi bước tạo vector đặc trưng chạy.
# Định nghĩa lớp BoundedDetector; lớp gom dữ liệu và các hàm liên quan.
class BoundedDetector:
    # Chuỗi tài liệu (docstring) mô tả cách dùng; Python lưu nội dung này trong __doc__.
    """Delegate unchanged detection/alignment; reject count limits before embedding."""

    # Khởi tạo đối tượng và giữ các phụ thuộc/trạng thái cần dùng trong những lần gọi sau.
    # Định nghĩa hàm __init__(self, detector, maximum, enrollment). self là đối tượng đang gọi phương thức. Mặc định: enrollment=False.
    def __init__(self, detector, maximum, enrollment=False):
        # Tách kết quả theo thứ tự và gán cho (self.detector, self.maximum, self.enrollment): bộ giá trị (detector, maximum, enrollment) (tuple giữ thứ tự, không gán lại từng phần tử).
        self.detector, self.maximum, self.enrollment = detector, maximum, enrollment

    # Chuyển thuộc tính chưa có trên lớp bọc sang đối tượng gốc để giữ cách sử dụng cũ.
    # Định nghĩa hàm __getattr__(self, name). self là đối tượng đang gọi phương thức.
    def __getattr__(self, name):
        # Kết thúc hàm và trả đọc thuộc tính theo tên bằng getattr(...); dữ liệu/tham số: self.detector, name.
        return getattr(self.detector, name)

    # Phát hiện khuôn mặt trong ảnh và trả khung mặt, điểm tin cậy, điểm mốc.
    # Định nghĩa hàm detect(self, image). self là đối tượng đang gọi phương thức.
    def detect(self, image):
        # Gán cho result: phát hiện các mặt cùng khung và điểm mốc qua self.detector.detect(...); dữ liệu/tham số: image.
        result = self.detector.detect(image)
        # Lấy validate_detection_result từ MTCNN.mtcnn.
        from MTCNN.mtcnn import validate_detection_result
        # Kiểm tra cấu trúc đầu ra của detector bằng validate_detection_result(...); dữ liệu/tham số: result.
        validate_detection_result(result)
        # Gán cho count: đếm số phần tử bằng len(...); dữ liệu/tham số: result.boxes.
        count = len(result.boxes)
        # Chỉ chạy khối bên dưới khi self.enrollment and count != 1 cho kết quả đúng.
        if self.enrollment and count != 1:
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'enrollment_face_count', f'Mỗi ảnh đăng ký cần đúng 1 khuôn mặt; phát hiện {count}.'.
            raise ApplicationError("enrollment_face_count", f"Mỗi ảnh đăng ký cần đúng 1 khuôn mặt; phát hiện {count}.")
        # Chỉ chạy khối bên dưới khi count > self.maximum cho kết quả đúng.
        if count > self.maximum:
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'too_many_faces', f'Ảnh có {count} khuôn mặt, giới hạn {self.maximum}.'.
            raise ApplicationError("too_many_faces", f"Ảnh có {count} khuôn mặt, giới hạn {self.maximum}.")
        # Kết thúc hàm và trả result (kết quả xử lý).
        return result


# Danh sách trường công khai của một kết quả mặt; embedding chỉ dùng nội bộ.
# Gán cho FACE_FIELDS: bộ giá trị ('face_index', 'bbox', 'confidence', 'person_id', 'status', 'similarity', 'threshold… (tuple giữ thứ tự, không gán lại từng phần tử).
FACE_FIELDS = ("face_index", "bbox", "confidence", "person_id", "status", "similarity", "threshold")


# Chỉ trả các trường cho phép; không đưa vector đặc trưng và dữ liệu nội bộ ra ngoài.
# Định nghĩa hàm public_faces(results).
def public_faces(results):
    # Chỉ xuất trường được cho phép: không trả vector, danh tính ứng viên, ảnh cắt hoặc bảng xếp hạng.
    # Với mỗi mặt, tạo dict chỉ chứa trường công khai trong FACE_FIELDS; vector và dữ liệu nội bộ không được đưa ra ngoài.
    return [{key: row[key] for key in FACE_FIELDS} for row in results]


# Dùng chung model và kho hồ sơ giữa API, giao diện và xử lý webcam.
# Định nghĩa lớp ApplicationService; lớp gom dữ liệu và các hàm liên quan.
class ApplicationService:
    # Khởi tạo đối tượng và giữ các phụ thuộc/trạng thái cần dùng trong những lần gọi sau.
    # Định nghĩa hàm __init__(self, settings, runtime_loader). self là đối tượng đang gọi phương thức. Mặc định: settings=None, runtime_loader=load_runtime.
    def __init__(self, settings=None, runtime_loader=load_runtime):
        # Gán cho self.settings: kết hợp bằng or: lấy giá trị đúng đầu tiên nếu có trong settings or Settings.from_env().
        self.settings = settings or Settings.from_env()
        # Gán cho self._loader: runtime_loader.
        self._loader = runtime_loader
        # RLock cho phép cùng một luồng lấy lại khóa, hữu ích khi các hàm dịch vụ gọi lồng nhau.
        # Tạo khóa luồng dùng chung cho model/phiên; RLock cho phép cùng luồng lấy khóa nhiều lần khi gọi hàm lồng nhau.
        self._mutex = RLock()
        # Chưa nạp model nên runtime để None; start() sẽ điền bộ model khi khởi động thành công.
        self.runtime = None
        # Chưa có lỗi nạp model; nếu start() thất bại sẽ lưu lỗi vào đây để health báo lại.
        self.model_error = None
        # Đánh dấu chưa khởi động để start() chỉ thực hiện nạp một lần trong vòng đời đối tượng.
        self._started = False
        # Lấy WebcamEnrollment từ .webcam_enrollment; dấu . chỉ module cùng package.
        from .webcam_enrollment import WebcamEnrollment
        # Gán cho self.webcam_enrollment: gọi WebcamEnrollment(...) với self; dữ liệu/tham số: self.
        self.webcam_enrollment = WebcamEnrollment(self)
        # Gán cho path: gọi db.validate_database_path(...) với self.settings.registry; dữ liệu/tham số: self.settings.registry, for_write=True.
        path = db.validate_database_path(self.settings.registry, for_write=True)
        # Chỉ chạy khối bên dưới khi not (path.is_relative_to(ROOT) or path.is_relative_to(Path(tempfile.gettempdir()).resolve())) cho kết quả đúng.
        if not (path.is_relative_to(ROOT) or path.is_relative_to(Path(tempfile.gettempdir()).resolve())):
            # Phát sinh lỗi: gọi ValueError(...) với 'Registry must be within the proje…; dữ liệu/tham số: 'Registry must be within the project or temporary directory.'.
            raise ValueError("Registry must be within the project or temporary directory.")
        # Gán cho self.path: path (đường dẫn file).
        self.path = path
        # Khóa dùng chung giữa các tiến trình ứng dụng, không cần tạo thư mục kho hồ sơ.
        # Cùng đường dẫn kho sẽ sinh cùng tên khóa, kể cả ở các tiến trình khác nhau.
        # Gán cho lock_key: lấy mã băm dạng chữ hệ 16 qua hashlib.sha256(os.path.normcase(str(path)).encode()).hexdigest(...).
        lock_key = hashlib.sha256(os.path.normcase(str(path)).encode()).hexdigest()
        # Gán cho self._file_lock: tạo khóa file dùng chung giữa các tiến trình bằng FileLock(...); dữ liệu/tham số: Path(tempfile.gettempdir()) / f'face_application_{lock_key}.lock', timeout=self.settings.lock_timeout.
        self._file_lock = FileLock(Path(tempfile.gettempdir()) / f"face_application_{lock_key}.lock",
                                   # Truyền timeout = self.settings.lock_timeout.
                                   timeout=self.settings.lock_timeout)

    # Nạp model một lần; giữ lại lỗi để health vẫn báo được trạng thái khi nạp thất bại.
    # Định nghĩa hàm start(self). self là đối tượng đang gọi phương thức.
    def start(self):
        # Mở khối with để self._mutex (lấy khóa trước khi vào khối, tự nhả khóa khi ra khỏi khối); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self._mutex:
            # Chỉ thử nạp một lần trong vòng đời đối tượng, tránh nạp model lại theo mỗi yêu cầu.
            # Chỉ chạy khối bên dưới khi self._started cho kết quả đúng.
            if self._started:
                # Kết thúc hàm, trả None vì không kèm giá trị.
                return
            # Đánh dấu đã bắt đầu nạp; các lần start() sau không thử nạp lại model.
            self._started = True
            # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
            try:
                # Gán cho self.runtime: gọi self._loader(...) với self.settings; dữ liệu/tham số: self.settings.
                self.runtime = self._loader(self.settings)
            # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
            except ApplicationError as error:
                # Gán cho self.model_error: error (lỗi vừa bắt được).
                self.model_error = error
            # Bắt lỗi Exception và đặt tên error để xử lý trong khối bên dưới.
            except Exception as error:
                # Báo lỗi tham số hoặc ghi log lỗi qua logger.error(...); dữ liệu/tham số: 'Model startup failed (%s)', type(error).__name__.
                logger.error("Model startup failed (%s)", type(error).__name__)
                # Gán cho self.model_error: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'model_load_failed', 'Không nạp được MTCNN/FaceNet. Kiểm tra assets và runtime.', 503.
                self.model_error = ApplicationError("model_load_failed", "Không nạp được MTCNN/FaceNet. Kiểm tra assets và runtime.", 503)

    # Khóa luồng trong tiến trình và khóa file giữa các tiến trình; with tự giải phóng khóa.
    # Decorator contextmanager: biến hàm có yield thành ngữ cảnh dùng với with.
    @contextmanager
    # Định nghĩa hàm guard(self). self là đối tượng đang gọi phương thức.
    def guard(self):
        # Mở khối with để self._mutex (lấy khóa trước khi vào khối, tự nhả khóa khi ra khỏi khối); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self._mutex:
            # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
            try:
                # Mở khối with để self._file_lock (lấy khóa trước khi vào khối, tự nhả khóa khi ra khỏi khối); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                with self._file_lock:
                    # Tạm nhường quyền chạy cho thân khối with; phần sau yield tiếp tục khi khối with kết thúc.
                    yield
            # Bắt lỗi Timeout và đặt tên error để xử lý trong khối bên dưới.
            except Timeout as error:
                # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'service_busy', 'Registry đang bận; thử lại sau.', 503; from giữ lại lỗi gốc để truy nguyên.
                raise ApplicationError("service_busy", "Registry đang bận; thử lại sau.", 503) from error

    # Dừng thao tác sớm nếu model chưa sẵn sàng, kèm nguyên nhân lỗi khởi động.
    # Định nghĩa hàm _require_model(self). self là đối tượng đang gọi phương thức.
    def _require_model(self):
        # Chỉ chạy khối bên dưới khi self.runtime is None cho kết quả đúng.
        if self.runtime is None:
            # Phát sinh lỗi: kết hợp bằng or: lấy giá trị đúng đầu tiên nếu có trong self.model_error or ApplicationError('not_started', 'Model chưa được khởi động.', 50….
            raise self.model_error or ApplicationError("not_started", "Model chưa được khởi động.", 503)

    # Đọc và xác minh kho: cấu trúc/dữ liệu phải hợp lệ và tương thích phiên bản model.
    # Định nghĩa hàm _registry(self). self là đối tượng đang gọi phương thức.
    def _registry(self):
        # Chỉ chạy khối bên dưới khi not self.path.is_file() cho kết quả đúng.
        if not self.path.is_file():
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'registry_missing', 'Registry chưa tạo. Chạy python -m application init-registry.', 503.
            raise ApplicationError("registry_missing", "Registry chưa tạo. Chạy python -m application init-registry.", 503)
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Gọi db.validate_database_path(...) với self.path; dữ liệu/tham số: self.path, for_write=True.
            db.validate_database_path(self.path, for_write=True)
            # Chỉ chạy khối bên dưới khi self.path.stat().st_nlink > 1 cho kết quả đúng.
            if self.path.stat().st_nlink > 1:
                # Phát sinh lỗi: gọi ValueError(...) với 'Hardlinked registry'; dữ liệu/tham số: 'Hardlinked registry'.
                raise ValueError("Hardlinked registry")
            # Gán cho people: gọi db.list_people(...) với self.path; dữ liệu/tham số: self.path.
            people = db.list_people(self.path)
            # Lặp qua people; mỗi lượt gán phần tử cho person.
            for person in people:
                # Chỉ chạy khối bên dưới khi person['person_id'] == 'Unknown' cho kết quả đúng.
                if person["person_id"] == "Unknown":
                    # Phát sinh lỗi: gọi ValueError(...) với 'Reserved identity'; dữ liệu/tham số: 'Reserved identity'.
                    raise ValueError("Reserved identity")
                # Gán cho version: chọn self.runtime.version nếu self.runtime; ngược lại chọn person['model_version'].
                version = self.runtime.version if self.runtime else person["model_version"]
                # Chỉ chạy khối bên dưới khi db.get_person(self.path, person['person_id'], version) is None cho kết quả đúng.
                if db.get_person(self.path, person["person_id"], version) is None:
                    # Phát sinh lỗi: gọi ValueError(...) với 'Registry changed during read'; dữ liệu/tham số: 'Registry changed during read'.
                    raise ValueError("Registry changed during read")
            # Kết thúc hàm và trả people (danh sách hồ sơ).
            return people
        # Bắt lỗi Exception và đặt tên error để xử lý trong khối bên dưới.
        except Exception as error:
            # Báo lỗi tham số hoặc ghi log lỗi qua logger.error(...); dữ liệu/tham số: 'Registry validation failed (%s)', type(error).__name__.
            logger.error("Registry validation failed (%s)", type(error).__name__)
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'registry_invalid', 'Registry lỗi hoặc model_version không tương thích; không tự migrate.', 503; from giữ lại lỗi gốc để truy nguyên.
            raise ApplicationError("registry_invalid", "Registry lỗi hoặc model_version không tương thích; không tự migrate.", 503) from error

    # Tổng hợp tình trạng model và kho hồ sơ để phía gọi biết hệ thống có sẵn sàng hay không.
    # Định nghĩa hàm health(self). self là đối tượng đang gọi phương thức.
    def health(self):
        # Mở khối with để lấy khóa truy cập model và kho hồ sơ qua self.guard(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self.guard():
            # Gán cho problems: danh sách [] ([] giữ các phần tử theo thứ tự).
            problems = []
            # Chỉ chạy khối bên dưới khi self.runtime is None cho kết quả đúng.
            if self.runtime is None:
                # Thêm một phần tử vào cuối danh sách qua problems.append(...); dữ liệu/tham số: (self.model_error or ApplicationError('not_started', 'Chưa khởi động.', 503)).payload()['error'].
                problems.append((self.model_error or ApplicationError("not_started", "Chưa khởi động.", 503)).payload()["error"])
            # Gán cho count: None (chưa có/không có giá trị).
            count = None
            # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
            try:
                # Gán cho count: đếm số phần tử bằng len(...); dữ liệu/tham số: self._registry() (đọc và xác minh kho hồ sơ qua self._registry(...)).
                count = len(self._registry())
                # Gán cho registry_state: chọn 'empty' nếu count == 0; ngược lại chọn 'ready'.
                registry_state = "empty" if count == 0 else "ready"
            # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
            except ApplicationError as error:
                # Gán cho registry_state: chọn 'missing' nếu error.code == 'registry_missing'; ngược lại chọn 'invalid'.
                registry_state = "missing" if error.code == "registry_missing" else "invalid"
                # Thêm một phần tử vào cuối danh sách qua problems.append(...); dữ liệu/tham số: error.payload()['error'].
                problems.append(error.payload()["error"])
            # Kết thúc hàm và trả từ điển {'status': 'ok' if not problems else 'degraded', 'backend': 'running', 'models': 're… (ánh xạ khóa → giá trị).
            return {"status": "ok" if not problems else "degraded", "backend": "running",
                    # Tiếp phần biểu thức nhiều dòng: chọn 'ready' nếu self.runtime; ngược lại chọn 'unavailable'.
                    "models": "ready" if self.runtime else "unavailable",
                    # Tiếp phần biểu thức nhiều dòng: chuỗi 'person_count'.
                    "registry": registry_state, "person_count": count,
                    # Tiếp phần biểu thức nhiều dòng: kết hợp bằng and: mọi điều kiện phải đúng trong self.runtime is not None and count is not None.
                    "registry_version_checked": self.runtime is not None and count is not None,
                    # Tiếp phần biểu thức nhiều dòng: chuỗi 'threshold_profile'.
                    "threshold_profile": "v2", "issues": problems}

    # Chặn ID rỗng, tên dành riêng Unknown, ký tự điều khiển và dấu phân cách đường dẫn.
    # Decorator staticmethod: hàm trong lớp không tự nhận tham số self.
    @staticmethod
    # Định nghĩa hàm check_id(person_id).
    def check_id(person_id):
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Kiểm tra ID theo quy định của tầng FaceNet bằng validate_person_id(...); dữ liệu/tham số: person_id.
            validate_person_id(person_id)
        # Bắt lỗi (ValueError, TypeError) và đặt tên error để xử lý trong khối bên dưới.
        except (ValueError, TypeError) as error:
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'invalid_person_id', 'person_id không được rỗng hoặc có khoảng trắng hai đầu.'; from giữ lại lỗi gốc để truy nguyên.
            raise ApplicationError("invalid_person_id", "person_id không được rỗng hoặc có khoảng trắng hai đầu.") from error
        # Chỉ chạy khối bên dưới khi person_id == 'Unknown' or len(person_id) > 100 or any((ord(c) < 32 or c in '/\\' for c in person_id)) cho kết quả đúng.
        if (person_id == "Unknown" or len(person_id) > 100
                # Bổ sung trường hợp được chấp nhận bằng or: or any(ord(c) < 32 or c in "/\\" for c in person_id)).
                or any(ord(c) < 32 or c in "/\\" for c in person_id)):
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'invalid_person_id', 'ID tối đa 100 ký tự, không dùng Unknown, dấu /, \\ hoặc ký tự điều khiển.'.
            raise ApplicationError("invalid_person_id", "ID tối đa 100 ký tự, không dùng Unknown, dấu /, \\ hoặc ký tự điều khiển.")

    # Giữ nguyên lỗi nghiệp vụ; đổi lỗi bất ngờ của pipeline thành lỗi ứng dụng thống nhất.
    # Decorator contextmanager: biến hàm có yield thành ngữ cảnh dùng với with.
    @contextmanager
    # Định nghĩa hàm _operation(self). self là đối tượng đang gọi phương thức.
    def _operation(self):
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Tạm nhường quyền chạy cho thân khối with; phần sau yield tiếp tục khi khối with kết thúc.
            yield
        # Bắt lỗi ApplicationError để xử lý trong khối bên dưới.
        except ApplicationError:
            # Ném lại lỗi đang xử lý để phía gọi biết thao tác chưa thành công.
            raise
        # Bắt lỗi Exception và đặt tên error để xử lý trong khối bên dưới.
        except Exception as error:
            # Báo lỗi tham số hoặc ghi log lỗi qua logger.error(...); dữ liệu/tham số: 'Pipeline operation failed (%s)', type(error).__name__.
            logger.error("Pipeline operation failed (%s)", type(error).__name__)
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'pipeline_failed', 'Detection/alignment/inference hoặc SQLite thất bại; không có kết quả nhận diện hợp lệ.', 500; from giữ lại lỗi gốc để truy nguyên.
            raise ApplicationError("pipeline_failed", "Detection/alignment/inference hoặc SQLite thất bại; không có kết quả nhận diện hợp lệ.", 500) from error

    # Kiểm tra ID/ảnh, chặn trùng rồi gọi pipeline tạo vector và lưu hồ sơ.
    # Định nghĩa hàm enroll(self, person_id, images). self là đối tượng đang gọi phương thức.
    def enroll(self, person_id, images):
        # Kiểm tra ID hợp lệ qua self.check_id(...); dữ liệu/tham số: person_id.
        self.check_id(person_id)
        # Chỉ chạy khối bên dưới khi not isinstance(images, (list, tuple)) or not 1 <= len(images) <= self.settings.max_images cho kết quả đúng.
        if not isinstance(images, (list, tuple)) or not 1 <= len(images) <= self.settings.max_images:
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'image_count', f'Cần từ 1 đến {self.settings.max_images} ảnh.'.
            raise ApplicationError("image_count", f"Cần từ 1 đến {self.settings.max_images} ảnh.")
        # Lặp qua images; mỗi lượt gán phần tử cho image.
        for image in images:
            # Kiểm tra kiểu mảng, số kênh và kích thước RGB bằng validate_rgb(...); dữ liệu/tham số: image, self.settings.
            validate_rgb(image, self.settings)
        # Mở khối with để lấy khóa truy cập model và kho hồ sơ qua self.guard(...); bao thao tác để chuyển lỗi bất ngờ thành ApplicationError qua self._operation(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self.guard(), self._operation():
            # Bảo đảm model đã nạp trước khi xử lý qua self._require_model(...).
            self._require_model()
            # Gán cho people: đọc và xác minh kho hồ sơ qua self._registry(...).
            people = self._registry()
            # Chỉ chạy khối bên dưới khi any((p['person_id'] == person_id for p in people)) cho kết quả đúng.
            if any(p["person_id"] == person_id for p in people):
                # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'duplicate_person', 'person_id đã tồn tại; không ghi đè.', 409.
                raise ApplicationError("duplicate_person", "person_id đã tồn tại; không ghi đè.", 409)
            # Gán cho rt: self.runtime (đọc thuộc tính).
            rt = self.runtime
            # Gán cho detector: bọc detector để giới hạn số mặt bằng BoundedDetector(...); dữ liệu/tham số: rt.detector, self.settings.max_faces, enrollment=True.
            detector = BoundedDetector(rt.detector, self.settings.max_faces, enrollment=True)
            # Gán cho result: tạo vector đại diện từ các ảnh và ghi hồ sơ bằng enroll_person_from_images(...); dữ liệu/tham số: person_id, images, detector.
            result = enroll_person_from_images(person_id, images, detector, rt.net, self.path, rt.version)
            # Kết thúc hàm và trả tạo từ điển bằng vòng lặp rút gọn; khóa là key, giá trị là result[key].
            return {key: result[key] for key in ("person_id", "num_images", "status")}

    # Chạy nhận diện và trả kết quả cho từng khuôn mặt cùng thời gian xử lý.
    # Định nghĩa hàm recognize(self, image). self là đối tượng đang gọi phương thức.
    def recognize(self, image):
        # Gán cho started: đọc đồng hồ độ chính xác cao để đo thời gian xử lý bằng perf_counter(...).
        started = perf_counter()
        # Kiểm tra kiểu mảng, số kênh và kích thước RGB bằng validate_rgb(...); dữ liệu/tham số: image, self.settings.
        validate_rgb(image, self.settings)
        # Mở khối with để lấy khóa truy cập model và kho hồ sơ qua self.guard(...); bao thao tác để chuyển lỗi bất ngờ thành ApplicationError qua self._operation(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self.guard(), self._operation():
            # Bảo đảm model đã nạp trước khi xử lý qua self._require_model(...).
            self._require_model()
            # Đọc và xác minh kho hồ sơ qua self._registry(...).
            self._registry()
            # Gán cho rt: self.runtime (đọc thuộc tính).
            rt = self.runtime
            # Gán cho rows: chạy pipeline nhận diện khuôn mặt trong ảnh bằng recognize_image(...); dữ liệu/tham số: image, BoundedDetector(rt.detector, self.settings.max_faces) (bọc detector để giới hạn số mặt bằng BoundedDetector(...)), rt.net.
            rows = recognize_image(image, BoundedDetector(rt.detector, self.settings.max_faces),
                                   # Ba đối số cuối của recognize_image: mạng FaceNet, đường dẫn kho và phiên bản model cần khớp với dữ liệu đã lưu.
                                   rt.net, self.path, rt.version)
            # Gán cho faces: lọc các trường kết quả được phép công khai bằng public_faces(...); dữ liệu/tham số: rows.
            faces = public_faces(rows)
        # Độ trễ dùng mili giây; FPS xử lý là nghịch đảo thời gian tính theo giây.
        # Gán cho elapsed: trừ trong perf_counter() - started.
        elapsed = perf_counter() - started
        # Kết thúc hàm và trả từ điển {'faces': faces, 'face_count': len(faces), 'image': {'width': image.shape[1], 'heigh… (ánh xạ khóa → giá trị).
        return {"faces": faces, "face_count": len(faces),
                # Tiếp phần biểu thức nhiều dòng: từ điển {'width': image.shape[1], 'height': image.shape[0]} (ánh xạ khóa → giá trị).
                "image": {"width": image.shape[1], "height": image.shape[0]},
                # Tiếp phần biểu thức nhiều dòng: chuỗi 'processing_fps'.
                "latency_ms": elapsed * 1000, "processing_fps": 1 / elapsed}

    # Đọc danh sách hồ sơ; bổ sung tên hiển thị nếu có trong kho tên đi kèm.
    # Định nghĩa hàm persons(self). self là đối tượng đang gọi phương thức.
    def persons(self):
        # Lấy read_names từ .webcam_registry; dấu . chỉ module cùng package.
        from .webcam_registry import read_names
        # Mở khối with để lấy khóa truy cập model và kho hồ sơ qua self.guard(...); bao thao tác để chuyển lỗi bất ngờ thành ApplicationError qua self._operation(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self.guard(), self._operation():
            # Gán cho people: đọc và xác minh kho hồ sơ qua self._registry(...).
            people = self._registry()
            # Gán cho names: đọc ánh xạ ID với tên hiển thị bằng read_names(...); dữ liệu/tham số: self.path.
            names = read_names(self.path)
            # Kết thúc hàm và trả tạo danh sách bằng cách duyệt p trong people; mỗi phần tử là {'person_id': p['person_id'], 'num_images': ….
            return [{"person_id": p["person_id"], "num_images": p["num_images"],
                     # Tiếp phần biểu thức nhiều dòng: chọn {'display_name': names[p['person_id']]} nếu p['person_id'] in names; ngược lại chọn {}.
                     **({"display_name": names[p["person_id"]]} if p["person_id"] in names else {})}
                    # Phần for của biểu thức rút gọn: duyệt từng phần tử theo thứ tự và tạo kết quả.
                    for p in people]

    # Xóa hồ sơ theo ID; báo lỗi nếu không tìm thấy người cần xóa.
    # Định nghĩa hàm delete(self, person_id). self là đối tượng đang gọi phương thức.
    def delete(self, person_id):
        # Kiểm tra ID hợp lệ qua self.check_id(...); dữ liệu/tham số: person_id.
        self.check_id(person_id)
        # Mở khối with để lấy khóa truy cập model và kho hồ sơ qua self.guard(...); bao thao tác để chuyển lỗi bất ngờ thành ApplicationError qua self._operation(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self.guard(), self._operation():
            # Đọc và xác minh kho hồ sơ qua self._registry(...).
            self._registry()
            # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
            try:
                # Lấy delete_identity từ .webcam_registry; dấu . chỉ module cùng package.
                from .webcam_registry import delete_identity
                # Xóa đồng bộ vector và tên theo ID bằng delete_identity(...); dữ liệu/tham số: self.path, person_id.
                delete_identity(self.path, person_id)
            # Bắt lỗi KeyError và đặt tên error để xử lý trong khối bên dưới.
            except KeyError as error:
                # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'person_not_found', 'Không tìm thấy person_id.', 404; from giữ lại lỗi gốc để truy nguyên.
                raise ApplicationError("person_not_found", "Không tìm thấy person_id.", 404) from error
            # Kết thúc hàm và trả từ điển {'person_id': person_id, 'status': 'deleted'} (ánh xạ khóa → giá trị).
            return {"person_id": person_id, "status": "deleted"}
