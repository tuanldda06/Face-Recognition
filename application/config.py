# Các đường dẫn và giới hạn dùng chung; Settings gom cấu hình vào một đối tượng.
# Lấy dataclass từ dataclasses để tạo lớp dữ liệu và sao chép/chuyển đổi dữ liệu của lớp.
from dataclasses import dataclass
# Nạp os để làm việc với biến môi trường và thông tin hệ điều hành.
import os
# Lấy Path từ pathlib để biểu diễn đường dẫn và thao tác file bằng đối tượng Path.
from pathlib import Path

# Tính từ vị trí file để đường dẫn không phụ thuộc thư mục đang mở terminal.
# Lấy file hiện tại qua __file__, chuẩn hóa đường dẫn rồi lấy thư mục gốc dự án; parents[0] là thư mục application, parents[1] là thư mục cha của nó.
ROOT = Path(__file__).resolve().parents[1]
# Gán cho MODEL_PATH: ghép đường dẫn bằng toán tử / của pathlib trong ROOT / 'FaceNet/models/facenet.onnx'. Ở đây / nối thư mục với tên file, không phải phép chia số.
MODEL_PATH = ROOT / "FaceNet/models/facenet.onnx"
# Dung lượng và mã băm kỳ vọng dùng để phát hiện model tải thiếu hoặc bị thay đổi.
# Gán cho MODEL_SHA256: chuỗi 'f6688ba6c95465fac18df53fdf8e094048fa3f1dd2c09f1cc6dd734a1c8dedf….
MODEL_SHA256 = "f6688ba6c95465fac18df53fdf8e094048fa3f1dd2c09f1cc6dd734a1c8dedf8"
# Gán cho MODEL_SIZE: giá trị 93987560.
MODEL_SIZE = 93987560
# Gán cho MODEL_URL: chuỗi 'https://media.axelera.ai/artifacts/model_cards/weights/face_rec….
MODEL_URL = "https://media.axelera.ai/artifacts/model_cards/weights/face_recognition/facenet.onnx"


# Cấu hình bất biến: frozen=True ngăn gán lại các trường sau khi tạo đối tượng.
# Decorator dataclass: tự sinh hàm khởi tạo từ các trường; frozen=True chặn gán lại trường nếu được bật.
@dataclass(frozen=True)
# Định nghĩa lớp Settings; lớp gom dữ liệu và các hàm liên quan.
class Settings:
    # Khai báo registry có kiểu gợi ý Path; gán ghép đường dẫn bằng toán tử / của pathlib trong ROOT / 'application/data/registry.sqlite'. Ở đây / nối thư mục với tên file, không phải phép chia số.
    registry: Path = ROOT / "application/data/registry.sqlite"
    # Đơn vị byte: 1 MiB = 1024 × 1024 byte; giới hạn từng file khác giới hạn cả request.
    # Dung lượng tối đa cho một ảnh; 8 × 1024 × 1024 byte = 8 MiB. Gợi ý kiểu: int.
    max_file_bytes: int = 8 * 1024 * 1024
    # Tổng kích thước tối đa của yêu cầu upload, gồm cả phần form/multipart; mặc định 32 MiB. Gợi ý kiểu: int.
    max_request_bytes: int = 32 * 1024 * 1024
    # Số ảnh tối đa cho một lượt đăng ký bằng hàm dịch vụ nội bộ. Gợi ý kiểu: int.
    max_images: int = 10
    # Giới hạn chiều rộng × chiều cao ảnh để kiểm soát bộ nhớ và thời gian xử lý. Gợi ý kiểu: int.
    max_pixels: int = 8_000_000
    # Chiều rộng hoặc cao lớn nhất được nhận, tính bằng pixel. Gợi ý kiểu: int.
    max_dimension: int = 4096
    # Số khuôn mặt tối đa được xử lý trong một ảnh nhận diện. Gợi ý kiểu: int.
    max_faces: int = 20
    # Số luồng CPU dành cho PyTorch/OpenCV để tránh dùng tài nguyên quá mức. Gợi ý kiểu: int.
    cpu_threads: int = 2
    # Ngưỡng thu nhỏ frame webcam tính bằng pixel; khung nhỏ hơn giúp giảm thời gian xử lý. Gợi ý kiểu: int.
    webcam_width: int = 640
    # Số giây tối đa chờ khóa file của kho trước khi báo dịch vụ đang bận. Gợi ý kiểu: float.
    lock_timeout: float = 30

    # Đọc biến môi trường; nếu chưa khai báo thì dùng cấu hình mặc định.
    # Decorator classmethod: tham số đầu tiên nhận chính lớp (cls).
    @classmethod
    # Định nghĩa hàm from_env(cls). cls là lớp đang gọi phương thức.
    def from_env(cls):
        # Kết thúc hàm và trả gọi cls(...); dữ liệu/tham số: registry=Path(os.environ.get('FACE_REGISTRY', str(cls.registry))).absolute(), cpu_threads=int(os.environ.get('FACE_CPU_THREADS', '2')).
        return cls(
            # Truyền registry (đường dẫn kho SQLite) = Path(os.environ.get('FACE_REGISTRY', str(cls.registry))).absolute().
            registry=Path(os.environ.get("FACE_REGISTRY", str(cls.registry))).absolute(),
            # Truyền cpu_threads = int(os.environ.get('FACE_CPU_THREADS', '2')).
            cpu_threads=int(os.environ.get("FACE_CPU_THREADS", "2")),
        # Đóng phần biểu thức/tham số nhiều dòng đang mở; các dấu ngoặc không tạo bước xử lý riêng.
        )

    # Dataclass tự gọi hàm này sau khi khởi tạo để kiểm tra cấu hình hợp lệ.
    # Định nghĩa hàm __post_init__(self). self là đối tượng đang gọi phương thức.
    def __post_init__(self):
        # Chỉ chạy khối bên dưới khi not self.registry.is_absolute() cho kết quả đúng.
        if not self.registry.is_absolute():
            # Phát sinh lỗi: gọi ValueError(...) với 'Registry path must be absolute.'; dữ liệu/tham số: 'Registry path must be absolute.'.
            raise ValueError("Registry path must be absolute.")
        # Lặp qua (self.max_file_bytes, self.max_request_bytes, self.max_images, self.max_pixels, self.max_…; mỗi lượt gán phần tử cho value.
        for value in (self.max_file_bytes, self.max_request_bytes, self.max_images,
                      # Tiếp phần biểu thức nhiều dòng: self.max_dimension (đọc thuộc tính).
                      self.max_pixels, self.max_dimension, self.max_faces,
                      # Tiếp phần biểu thức nhiều dòng: self.webcam_width (đọc thuộc tính).
                      self.cpu_threads, self.webcam_width, self.lock_timeout):
            # Chỉ chạy khối bên dưới khi value <= 0 cho kết quả đúng.
            if value <= 0:
                # Phát sinh lỗi: gọi ValueError(...) với 'Application limits must be positi…; dữ liệu/tham số: 'Application limits must be positive.'.
                raise ValueError("Application limits must be positive.")
