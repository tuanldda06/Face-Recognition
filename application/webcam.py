# Xử lý một khung hình webcam và cập nhật số khung hình, độ trễ, tốc độ FPS.
# Lấy perf_counter từ time để đo thời gian xử lý hoặc mô phỏng độ trễ.
from time import perf_counter

# Nạp numpy dưới tên ngắn np để tính toán mảng ảnh, vector và ma trận.
import numpy as np
# Lấy Image từ PIL để đọc/ghi ảnh, xử lý hướng EXIF, chọn font và vẽ lên ảnh bằng Pillow.
from PIL import Image

# Lấy ApplicationError từ .errors; dấu . chỉ module cùng package.
from .errors import ApplicationError
# Lấy validate_rgb từ .images; dấu . chỉ module cùng package.
from .images import validate_rgb
# Lấy annotate từ .visualization; dấu . chỉ module cùng package.
from .visualization import annotate


# Thu nhỏ frame khi cần, nhận diện, vẽ kết quả rồi cập nhật trạng thái đo FPS.
# Định nghĩa hàm process_frame(service, frame, state). Mặc định: state=None.
def process_frame(service, frame, state=None):
    # Gán cho started: đọc đồng hồ độ chính xác cao để đo thời gian xử lý bằng perf_counter(...).
    started = perf_counter()
    # Chỉ chạy khối bên dưới khi frame is None cho kết quả đúng.
    if frame is None:
        # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'camera_unavailable', 'Chưa có frame. Cho phép camera trong trình duyệt và dùng localhost hoặc HTTPS.'.
        raise ApplicationError("camera_unavailable", "Chưa có frame. Cho phép camera trong trình duyệt và dùng localhost hoặc HTTPS.")
    # Kiểm tra kiểu mảng, số kênh và kích thước RGB bằng validate_rgb(...); dữ liệu/tham số: frame, service.settings.
    validate_rgb(frame, service.settings)
    # Tách kết quả theo thứ tự và gán cho (height, width): lấy phần tử/lát cắt frame.shape[:2] từ frame.shape.
    height, width = frame.shape[:2]
    # Chỉ chạy khối bên dưới khi width > service.settings.webcam_width cho kết quả đúng.
    if width > service.settings.webcam_width:
        # Gán cho target: service.settings.webcam_width (đọc thuộc tính).
        target = service.settings.webcam_width
        # Gán cho frame: biểu diễn dữ liệu thành mảng NumPy, có thể dùng lại bộ nhớ sẵn có bằng np.asarray(...); dữ liệu/tham số: Image.fromarray(frame).resize((target, max(20, round(height * target / width))), Image.Resampling.B… (gọi Image.fromarray(frame).resize(...) với (target, max(20, round(height * ta…, Image.Resampling.BILINEAR).
        frame = np.asarray(Image.fromarray(frame).resize((target, max(20, round(height * target / width))), Image.Resampling.BILINEAR))
    # Gán cho result: nhận diện ảnh bằng dịch vụ qua service.recognize(...); dữ liệu/tham số: frame.
    result = service.recognize(frame)
    # Gán cho annotated: vẽ khung mặt và nhãn nhận diện trên ảnh sao chép bằng annotate(...); dữ liệu/tham số: frame, result['faces'].
    annotated = annotate(frame, result["faces"])
    # Gán cho finished: đọc đồng hồ độ chính xác cao để đo thời gian xử lý bằng perf_counter(...).
    finished = perf_counter()
    # Sao chép trạng thái đầu vào, rồi cập nhật bộ đếm cho phiên gọi hiện tại.
    # Nếu state rỗng/None thì dùng dict rỗng; dict(...) tạo bản sao nông để không gán trực tiếp vào dict đầu vào.
    state = dict(state or {})
    # Chỉ lưu mốc frame đầu tiên nếu chưa có khóa first_received; những frame sau giữ nguyên mốc này.
    state.setdefault("first_received", started)
    # Đọc số frame đã xử lý, mặc định 0 cho phiên mới, rồi tăng thêm 1.
    state["frames"] = state.get("frames", 0) + 1
    # Gán cho elapsed: trừ trong finished - started.
    elapsed = finished - started
    # Đổi thời gian xử lý một frame từ giây sang mili giây bằng cách nhân 1000.
    result["frame_latency_ms"] = elapsed * 1000
    # Nghịch đảo thời gian một frame cho FPS xử lý lý thuyết; chưa tính khoảng chờ frame truyền đến.
    result["processing_fps"] = 1 / elapsed
    # FPS của cả phiên gồm khoảng cách giữa các lần gửi; khác FPS chỉ tính thời gian xử lý frame.
    # Chia tổng số frame cho thời gian từ frame đầu đến hiện tại; FPS này có tính các khoảng chờ giữa frame.
    result["stream_fps"] = state["frames"] / (finished - state["first_received"])
    # Gán cho result['frames_processed']: lấy phần tử/lát cắt state['frames'] từ state.
    result["frames_processed"] = state["frames"]
    # Kết thúc hàm và trả bộ giá trị (annotated, result, state) (tuple giữ thứ tự, không gán lại từng phần tử).
    return annotated, result, state
