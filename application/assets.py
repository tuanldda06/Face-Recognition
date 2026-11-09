# Kiểm tra file model ONNX bằng dung lượng và SHA-256 (dấu vân tay của nội dung file).
# Nạp hashlib để tính mã băm để kiểm tra nội dung file/ảnh/vector.
import hashlib
# Lấy Path từ pathlib để biểu diễn đường dẫn và thao tác file bằng đối tượng Path.
from pathlib import Path

# Lấy MODEL_PATH, MODEL_SHA256, MODEL_SIZE từ .config; dấu . chỉ module cùng package.
from .config import MODEL_PATH, MODEL_SHA256, MODEL_SIZE
# Lấy ApplicationError từ .errors; dấu . chỉ module cùng package.
from .errors import ApplicationError


# Phân biệt model thật với con trỏ Git LFS, rồi đối chiếu dung lượng và mã băm.
# Định nghĩa hàm verify_model(path). Mặc định: path=MODEL_PATH.
def verify_model(path: Path = MODEL_PATH):
    # Chỉ chạy khối bên dưới khi not path.is_file() cho kết quả đúng.
    if not path.is_file():
        # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'model_missing', 'Thiếu FaceNet ONNX. Xem application/README.md.', 503.
        raise ApplicationError("model_missing", "Thiếu FaceNet ONNX. Xem application/README.md.", 503)
    # Mở khối with để mở file hoặc ảnh theo chế độ chỉ định qua path.open(...); dữ liệu/tham số: 'rb'; gọi tài nguyên là stream; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with path.open("rb") as stream:
        # Gán cho prefix: đọc dữ liệu từ luồng qua stream.read(...); dữ liệu/tham số: 200.
        prefix = stream.read(200)
        # Chỉ chạy khối bên dưới khi prefix.startswith(b'version https://git-lfs.github.com/spec/v1') cho kết quả đúng.
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'model_lfs_pointer', 'FaceNet đang là Git LFS pointer; cần khôi phục model gốc.', 503.
            raise ApplicationError("model_lfs_pointer", "FaceNet đang là Git LFS pointer; cần khôi phục model gốc.", 503)
        # Dịch vị trí đọc trong file qua stream.seek(...); dữ liệu/tham số: 0.
        stream.seek(0)
        # Gán cho digest: lấy mã băm dạng chữ hệ 16 qua hashlib.file_digest(stream, 'sha256').hexdigest(...).
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    # Chỉ chạy khối bên dưới khi path.stat().st_size != MODEL_SIZE or digest != MODEL_SHA256 cho kết quả đúng.
    if path.stat().st_size != MODEL_SIZE or digest != MODEL_SHA256:
        # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'model_checksum', 'FaceNet không khớp dung lượng/SHA-256 bàn giao.', 503.
        raise ApplicationError("model_checksum", "FaceNet không khớp dung lượng/SHA-256 bàn giao.", 503)
    # Kết thúc hàm và trả từ điển {'sha256': digest, 'bytes': path.stat().st_size} (ánh xạ khóa → giá trị).
    return {"sha256": digest, "bytes": path.stat().st_size}
