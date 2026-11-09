# Đo tốc độ từng công đoạn nhận diện; kết quả đo thời gian không phải đánh giá độ chính xác.
# Nạp argparse để đọc tham số người dùng nhập trong terminal.
import argparse
# Lấy ExitStack từ contextlib để quản lý tài nguyên bằng with, yield và thao tác dọn tự động.
from contextlib import ExitStack
# Lấy replace từ dataclasses để tạo lớp dữ liệu và sao chép/chuyển đổi dữ liệu của lớp.
from dataclasses import replace
# Lấy datetime, timezone từ datetime để ghi ngày giờ UTC vào báo cáo.
from datetime import datetime, timezone
# Nạp hashlib để tính mã băm để kiểm tra nội dung file/ảnh/vector.
import hashlib
# Nạp importlib.metadata để đọc phiên bản các gói đã cài.
import importlib.metadata
# Nạp json để chuyển đổi dữ liệu Python với JSON.
import json
# Nạp os để làm việc với biến môi trường và thông tin hệ điều hành.
import os
# Lấy Path từ pathlib để biểu diễn đường dẫn và thao tác file bằng đối tượng Path.
from pathlib import Path
# Nạp platform để đọc thông tin máy và hệ điều hành cho báo cáo.
import platform
# Nạp tempfile để tạo file/thư mục tạm có kiểm soát vòng đời.
import tempfile
# Lấy perf_counter từ time để đo thời gian xử lý hoặc mô phỏng độ trễ.
from time import perf_counter
# Lấy patch từ unittest.mock để thay tạm hàm/model và theo dõi lời gọi trong kiểm thử.
from unittest.mock import patch
# Nạp uuid để sinh mã ngẫu nhiên để tên báo cáo không trùng nhau.
import uuid

# Nạp numpy dưới tên ngắn np để tính toán mảng ảnh, vector và ma trận.
import numpy as np

# Lấy facenet_model, recognition từ FaceNet.
from FaceNet import facenet_model, recognition
# Lấy recognition_pipeline từ integration.
from integration import recognition_pipeline
# Lấy ROOT, Settings từ .config; dấu . chỉ module cùng package.
from .config import ROOT, Settings
# Lấy ApplicationError từ .errors; dấu . chỉ module cùng package.
from .errors import ApplicationError
# Lấy decode_image từ .images; dấu . chỉ module cùng package.
from .images import decode_image
# Lấy ApplicationService từ .service; dấu . chỉ module cùng package.
from .service import ApplicationService
# Lấy annotate từ .visualization; dấu . chỉ module cùng package.
from .visualization import annotate


# Tính trung bình, trung vị p50 và p95: 95% mẫu có thời gian không vượt mức p95.
# Định nghĩa hàm summarize(values).
def summarize(values):
    # Chỉ chạy khối bên dưới khi not values cho kết quả đúng.
    if not values:
        # Kết thúc hàm và trả từ điển {'status': 'not_measured', 'mean_ms': None, 'p50_ms': None, 'p95_ms': None} (ánh xạ khóa → giá trị).
        return {"status": "not_measured", "mean_ms": None, "p50_ms": None, "p95_ms": None}
    # Kết thúc hàm và trả từ điển {'status': 'measured', 'mean_ms': float(np.mean(values)), 'p50_ms': float(np.percent… (ánh xạ khóa → giá trị).
    return {"status": "measured", "mean_ms": float(np.mean(values)),
            # Tiếp phần biểu thức nhiều dòng: đổi sang số thực bằng float(...).
            "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95))}


# Bọc detector để đo thời gian phát hiện/căn chỉnh mà vẫn gọi thuật toán gốc.
# Định nghĩa lớp TimedDetector; lớp gom dữ liệu và các hàm liên quan.
class TimedDetector:
    # Khởi tạo đối tượng và giữ các phụ thuộc/trạng thái cần dùng trong những lần gọi sau.
    # Định nghĩa hàm __init__(self, detector, timings). self là đối tượng đang gọi phương thức.
    def __init__(self, detector, timings):
        # Tách kết quả theo thứ tự và gán cho (self.detector, self.timings): bộ giá trị (detector, timings) (tuple giữ thứ tự, không gán lại từng phần tử).
        self.detector, self.timings = detector, timings

    # Chuyển thuộc tính chưa có trên lớp bọc sang đối tượng gốc để giữ cách sử dụng cũ.
    # Định nghĩa hàm __getattr__(self, name). self là đối tượng đang gọi phương thức.
    def __getattr__(self, name):
        # Kết thúc hàm và trả đọc thuộc tính theo tên bằng getattr(...); dữ liệu/tham số: self.detector, name.
        return getattr(self.detector, name)

    # Phát hiện khuôn mặt trong ảnh và trả khung mặt, điểm tin cậy, điểm mốc.
    # Định nghĩa hàm detect(self). self là đối tượng đang gọi phương thức. *args gom các đối số vị trí còn lại thành tuple. **kwargs gom các đối số có tên còn lại thành dict.
    def detect(self, *args, **kwargs):
        # Kết thúc hàm và trả gọi timed(self.timings, 'detection', self.detector.detect)(...) với *args; dữ liệu/tham số: *args, **=kwargs.
        return timed(self.timings, "detection", self.detector.detect)(*args, **kwargs)

    # Cắt/căn chỉnh các mặt về ảnh RGB có kích thước đầu vào của FaceNet.
    # Định nghĩa hàm extract_faces_rgb(self). self là đối tượng đang gọi phương thức. *args gom các đối số vị trí còn lại thành tuple. **kwargs gom các đối số có tên còn lại thành dict.
    def extract_faces_rgb(self, *args, **kwargs):
        # Kết thúc hàm và trả gọi timed(self.timings, 'alignment', self.detector.extract_faces_rgb)(...) với *args; dữ liệu/tham số: *args, **=kwargs.
        return timed(self.timings, "alignment", self.detector.extract_faces_rgb)(*args, **kwargs)


# Tạo hàm bọc đo thời gian; finally vẫn ghi thời gian kể cả khi hàm gốc phát sinh lỗi.
# Định nghĩa hàm timed(timings, name, function).
def timed(timings, name, function):
    # Gọi hàm gốc và cộng số mili giây vào bộ đếm của công đoạn tương ứng.
    # Định nghĩa hàm call(). *args gom các đối số vị trí còn lại thành tuple. **kwargs gom các đối số có tên còn lại thành dict.
    def call(*args, **kwargs):
        # Gán cho started: đọc đồng hồ độ chính xác cao để đo thời gian xử lý bằng perf_counter(...).
        started = perf_counter()
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Kết thúc hàm và trả gọi function(...) với *args; dữ liệu/tham số: *args, **=kwargs.
            return function(*args, **kwargs)
        # Khối finally luôn chạy sau try, kể cả khi có lỗi hoặc return; dùng để dọn tài nguyên.
        finally:
            # Gán cho timings[name]: cộng hoặc nối trong timings.get(name, 0.0) + (perf_counter() - started) * 1000.
            timings[name] = timings.get(name, 0.0) + (perf_counter() - started) * 1000
    # Kết thúc hàm và trả call.
    return call


# Chạy làm nóng trước, sau đó đo lặp lại một ảnh; tự phục hồi các hàm đã bọc khi kết thúc.
# Định nghĩa hàm run_case(service, path, repetitions, warmup).
def run_case(service, path, repetitions, warmup):
    # Gán cho data: đọc toàn bộ file dưới dạng byte qua path.read_bytes(...).
    data = path.read_bytes()
    # Gán cho image: giải mã byte thành ảnh RGB đã kiểm tra bằng decode_image(...); dữ liệu/tham số: data, service.settings.
    image = decode_image(data, service.settings)
    # Gán cho timings: từ điển {} (ánh xạ khóa → giá trị).
    timings = {}
    # Gán cho original_detector: service.runtime.detector (đọc thuộc tính).
    original_detector = service.runtime.detector
    # Gán cho measured: tạo từ điển bằng vòng lặp rút gọn; khóa là key, giá trị là [].
    measured = {key: [] for key in ("detection", "alignment", "facenet_preprocessing", "embedding",
                                    # Tiếp phần biểu thức nhiều dòng: chuỗi 'end_to_end'.
                                    "matching", "end_to_end")}
    # Tách kết quả theo thứ tự và gán cho (counts, samples): bộ giá trị ([], []) (tuple giữ thứ tự, không gán lại từng phần tử).
    counts, samples = [], []
    # ExitStack tự dọn tài nguyên và khôi phục các hàm bọc dù phép đo xảy ra lỗi.
    # Mở khối with để tạo bộ quản lý tự dọn nhiều tài nguyên bằng ExitStack(...); gọi tài nguyên là stack; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with ExitStack() as stack:
        # Gán cho service.runtime.detector: bọc detector để đo thời gian từng bước bằng TimedDetector(...); dữ liệu/tham số: original_detector, timings.
        service.runtime.detector = TimedDetector(original_detector, timings)
        # Gọi stack.callback(...) với setattr, service.runtime, 'detector'; dữ liệu/tham số: setattr, service.runtime, 'detector'.
        stack.callback(setattr, service.runtime, "detector", original_detector)
        # Gán cho extraction: tạo hàm bọc ghi thời gian của công đoạn bằng timed(...); dữ liệu/tham số: timings, 'embedding_including_preprocessing', facenet_model.extract_embedding.
        extraction = timed(timings, "embedding_including_preprocessing", facenet_model.extract_embedding)
        # Gọi stack.enter_context(...) với patch.object(recognition, 'extract…; dữ liệu/tham số: patch.object(recognition, 'extract_embedding', extraction) (thay tạm thuộc tính/hàm của đối tượng rồi khôi phục sau khối with bằng patch.object(...)).
        stack.enter_context(patch.object(recognition, "extract_embedding", extraction))
        # Gọi stack.enter_context(...) với patch.object(recognition_pipeline,…; dữ liệu/tham số: patch.object(recognition_pipeline, 'extract_embedding', extraction) (thay tạm thuộc tính/hàm của đối tượng rồi khôi phục sau khối with bằng patch.object(...)).
        stack.enter_context(patch.object(recognition_pipeline, "extract_embedding", extraction))
        # Gọi stack.enter_context(...) với patch.object(facenet_model, 'prepr…; dữ liệu/tham số: patch.object(facenet_model, 'preprocess_face', timed(timings, 'facenet_preprocessing', facenet_mode… (thay tạm thuộc tính/hàm của đối tượng rồi khôi phục sau khối with bằng patch.object(...)).
        stack.enter_context(patch.object(facenet_model, "preprocess_face",
                                        # Tiếp phần biểu thức nhiều dòng: tạo hàm bọc ghi thời gian của công đoạn bằng timed(...).
                                        timed(timings, "facenet_preprocessing", facenet_model.preprocess_face)))
        # Gọi stack.enter_context(...) với patch.object(recognition, 'detect_…; dữ liệu/tham số: patch.object(recognition, 'detect_unknown', timed(timings, 'matching', recognition.detect_unknown)) (thay tạm thuộc tính/hàm của đối tượng rồi khôi phục sau khối with bằng patch.object(...)).
        stack.enter_context(patch.object(recognition, "detect_unknown",
                                        # Tiếp phần biểu thức nhiều dòng: tạo hàm bọc ghi thời gian của công đoạn bằng timed(...).
                                        timed(timings, "matching", recognition.detect_unknown)))
        # Lặp qua range(warmup + repetitions); mỗi lượt gán phần tử cho index.
        for index in range(warmup + repetitions):
            # Xóa các phần tử hoặc đăng ký sự kiện xóa tùy đối tượng qua timings.clear(...).
            timings.clear()
            # Gán cho started: đọc đồng hồ độ chính xác cao để đo thời gian xử lý bằng perf_counter(...).
            started = perf_counter()
            # Gán cho result: nhận diện ảnh bằng dịch vụ qua service.recognize(...); dữ liệu/tham số: image.
            result = service.recognize(image)
            # Vẽ khung mặt và nhãn nhận diện trên ảnh sao chép bằng annotate(...); dữ liệu/tham số: image, result['faces'].
            annotate(image, result["faces"])
            # Gán cho timings['end_to_end']: nhân hoặc lặp trong (perf_counter() - started) * 1000.
            timings["end_to_end"] = (perf_counter() - started) * 1000
            # Chỉ chạy khối bên dưới khi 'embedding_including_preprocessing' in timings cho kết quả đúng.
            if "embedding_including_preprocessing" in timings:
                # Tách thời gian tiền xử lý khỏi tổng thời gian tạo vector để không tính trùng.
                # Gán cho timings['embedding']: trừ trong timings['embedding_including_preprocessing'] - timings.get('facenet_preprocessing', ….
                timings["embedding"] = timings["embedding_including_preprocessing"] - timings.get("facenet_preprocessing", 0)
            # Bỏ các lượt làm nóng vì lần đầu thường chậm hơn do khởi tạo và bộ nhớ đệm.
            # Chỉ chạy khối bên dưới khi index >= warmup cho kết quả đúng.
            if index >= warmup:
                # Thêm một phần tử vào cuối danh sách qua counts.append(...); dữ liệu/tham số: result['face_count'].
                counts.append(result["face_count"])
                # Thêm một phần tử vào cuối danh sách qua samples.append(...); dữ liệu/tham số: dict(timings) (tạo từ điển khóa–giá trị bằng dict(...)).
                samples.append(dict(timings))
                # Lặp qua measured; mỗi lượt gán phần tử cho name.
                for name in measured:
                    # Chỉ chạy khối bên dưới khi name in timings cho kết quả đúng.
                    if name in timings:
                        # Thêm một phần tử vào cuối danh sách qua measured[name].append(...); dữ liệu/tham số: timings[name].
                        measured[name].append(timings[name])
    # Kết thúc hàm và trả từ điển {'status': 'measured', 'image': path.name, 'image_sha256': hashlib.sha256(data).hexd… (ánh xạ khóa → giá trị).
    return {"status": "measured", "image": path.name,
            # Tiếp phần biểu thức nhiều dòng: lấy mã băm dạng chữ hệ 16 qua hashlib.sha256(data).hexdigest(...).
            "image_sha256": hashlib.sha256(data).hexdigest(), "dimensions": [image.shape[1], image.shape[0]],
            # Tiếp phần biểu thức nhiều dòng: chuỗi 'detected_faces_each_run'.
            "detected_faces_each_run": counts, "repetitions": repetitions, "warmup": warmup,
            # Tiếp phần biểu thức nhiều dòng: tạo từ điển bằng vòng lặp rút gọn; khóa là name, giá trị là summarize(values).
            "latency": {name: summarize(values) for name, values in measured.items()},
            # Tiếp phần biểu thức nhiều dòng: chia trong repetitions / (sum(measured['end_to_end']) / 1000).
            "fps": repetitions / (sum(measured["end_to_end"]) / 1000), "samples_ms": samples}


# Đọc tùy chọn đo, chuẩn bị dịch vụ, chạy từng ảnh và ghi báo cáo JSON mới.
# Định nghĩa hàm main(argv). Mặc định: argv=None.
def main(argv=None):
    # Gán cho parser: gọi argparse.ArgumentParser(...); dữ liệu/tham số: description=__doc__.
    parser = argparse.ArgumentParser(description=__doc__)
    # Khai báo một tùy chọn dòng lệnh qua parser.add_argument(...); dữ liệu/tham số: '--images', type=Path, nargs='+', default=[ROOT / 'integration/tests/fixtures/mtcnn_single.png', ROOT / 'application/tests/fix….
    parser.add_argument("--images", type=Path, nargs="+", default=[
        # Tiếp phần biểu thức nhiều dòng: ghép đường dẫn bằng toán tử / của pathlib trong ROOT / 'integration/tests/fixtures/mtcnn_single.png'. Ở đây / nối thư mục với tên file, không phải phép chia số.
        ROOT / "integration/tests/fixtures/mtcnn_single.png",
        # Tiếp phần biểu thức nhiều dòng: ghép đường dẫn bằng toán tử / của pathlib trong ROOT / 'application/tests/fixtures/group_two_faces.png'. Ở đây / nối thư mục với tên file, không phải phép chia số.
        ROOT / "application/tests/fixtures/group_two_faces.png",
        # Tiếp phần biểu thức nhiều dòng: ghép đường dẫn bằng toán tử / của pathlib trong ROOT / 'integration/tests/fixtures/mtcnn_multi.jpg'. Ở đây / nối thư mục với tên file, không phải phép chia số.
        ROOT / "integration/tests/fixtures/mtcnn_multi.jpg"])
    # Khai báo một tùy chọn dòng lệnh qua parser.add_argument(...); dữ liệu/tham số: '--registry', type=Path.
    parser.add_argument("--registry", type=Path)
    # Khai báo một tùy chọn dòng lệnh qua parser.add_argument(...); dữ liệu/tham số: '--temporary-gallery', action='store_true', help='Create a disposable gallery using the bundled single face; never modify a user regi….
    parser.add_argument("--temporary-gallery", action="store_true", help="Create a disposable gallery using the bundled single face; never modify a user registry")
    # Khai báo một tùy chọn dòng lệnh qua parser.add_argument(...); dữ liệu/tham số: '--repeat', type=int, default=10.
    parser.add_argument("--repeat", type=int, default=10)
    # Khai báo một tùy chọn dòng lệnh qua parser.add_argument(...); dữ liệu/tham số: '--warmup', type=int, default=2.
    parser.add_argument("--warmup", type=int, default=2)
    # Khai báo một tùy chọn dòng lệnh qua parser.add_argument(...); dữ liệu/tham số: '--cpu-threads', type=int, default=2.
    parser.add_argument("--cpu-threads", type=int, default=2)
    # Khai báo một tùy chọn dòng lệnh qua parser.add_argument(...); dữ liệu/tham số: '--output', type=Path.
    parser.add_argument("--output", type=Path)
    # Gán cho args: đọc các tham số dòng lệnh qua parser.parse_args(...); dữ liệu/tham số: argv.
    args = parser.parse_args(argv)
    # Chỉ chạy khối bên dưới khi args.repeat < 1 or args.warmup < 1 or (args.temporary_gallery and args.registry) cho kết quả đúng.
    if args.repeat < 1 or args.warmup < 1 or (args.temporary_gallery and args.registry):
        # Báo lỗi tham số hoặc ghi log lỗi qua parser.error(...); dữ liệu/tham số: 'repeat/warmup >= 1; choose either --registry or --temporary-gallery'.
        parser.error("repeat/warmup >= 1; choose either --registry or --temporary-gallery")
    # Gán cho report: từ điển {'schema': 'application-benchmark/1', 'created_utc': datetime.now(timezone.utc).isof… (ánh xạ khóa → giá trị).
    report = {"schema": "application-benchmark/1", "created_utc": datetime.now(timezone.utc).isoformat(),
              # Tiếp phần biểu thức nhiều dòng: kết hợp bằng or: lấy giá trị đúng đầu tiên nếu có trong platform.processor() or os.environ.get('PROCESSOR_IDENTIFIER', 'not_measured').
              "device": "CPU", "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "not_measured"),
              # Tiếp phần biểu thức nhiều dòng: chuỗi 'logical_cpus'.
              "logical_cpus": os.cpu_count(),
              # Tiếp phần biểu thức nhiều dòng: gọi platform.python_version(...).
              "os": platform.platform(), "python": platform.python_version(), "cpu_threads": args.cpu_threads,
              # Tiếp phần biểu thức nhiều dòng: tạo từ điển bằng vòng lặp rút gọn; khóa là name, giá trị là importlib.metadata.version(name).
              "runtime": {name: importlib.metadata.version(name) for name in
                          # Tiếp phần biểu thức nhiều dòng: bộ giá trị ('numpy', 'opencv-python', 'Pillow', 'torch', 'fastapi', 'gradio') (tuple giữ thứ tự, không gán lại từng phần tử).
                          ("numpy", "opencv-python", "Pillow", "torch", "fastapi", "gradio")},
              # Tiếp phần biểu thức nhiều dòng: chuỗi 'Service recognition + visualization; excludes model load, image….
              "scope": "Service recognition + visualization; excludes model load, image decode, HTTP and browser transport. Serial warm-cache calls.",
              # Tiếp phần biểu thức nhiều dòng: chuỗi 'extract_embedding minus preprocess_face; includes forward, vali….
              "stage_definitions": {"embedding": "extract_embedding minus preprocess_face; includes forward, validation and L2",
                                    # Tiếp phần biểu thức nhiều dòng: chuỗi 'existing detect_unknown, cosine/ranking/threshold'.
                                    "matching": "existing detect_unknown, cosine/ranking/threshold",
                                    # Tiếp phần biểu thức nhiều dòng: chuỗi 'existing extract_faces_rgb align=True'.
                                    "alignment": "existing extract_faces_rgb align=True"},
              # Tiếp phần biểu thức nhiều dòng: chuỗi 'accuracy_evaluation'.
              "accuracy_evaluation": False, "cases": []}
    # ExitStack tự dọn tài nguyên và khôi phục các hàm bọc dù phép đo xảy ra lỗi.
    # Mở khối with để tạo bộ quản lý tự dọn nhiều tài nguyên bằng ExitStack(...); gọi tài nguyên là stack; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with ExitStack() as stack:
        # Gán cho settings: tạo bản cấu hình sao chép với một số trường thay đổi bằng replace(...); dữ liệu/tham số: Settings.from_env() (gọi Settings.from_env(...)), cpu_threads=args.cpu_threads.
        settings = replace(Settings.from_env(), cpu_threads=args.cpu_threads)
        # Chỉ chạy khối bên dưới khi args.registry cho kết quả đúng.
        if args.registry:
            # Gán cho settings: tạo bản cấu hình sao chép với một số trường thay đổi bằng replace(...); dữ liệu/tham số: settings, registry=args.registry.absolute().
            settings = replace(settings, registry=args.registry.absolute())
        # Chỉ chạy khối bên dưới khi args.temporary_gallery cho kết quả đúng.
        if args.temporary_gallery:
            # Lấy initialize_enrollment_registry từ FaceNet.
            from FaceNet import initialize_enrollment_registry
            # Gán cho folder: tạo đối tượng đường dẫn bằng Path(...); dữ liệu/tham số: stack.enter_context(tempfile.TemporaryDirectory(prefix='face_benchmark_')) (gọi stack.enter_context(...) với tempfile.TemporaryDirectory(prefix…).
            folder = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="face_benchmark_")))
            # Gán cho settings: tạo bản cấu hình sao chép với một số trường thay đổi bằng replace(...); dữ liệu/tham số: settings, registry=folder / 'registry.sqlite'.
            settings = replace(settings, registry=folder / "registry.sqlite")
            # Khởi tạo kho SQLite theo cấu trúc FaceNet bằng initialize_enrollment_registry(...); dữ liệu/tham số: settings.registry.
            initialize_enrollment_registry(settings.registry)
        # Gán cho service: tạo dịch vụ dùng chung cho nhận diện và quản lý hồ sơ bằng ApplicationService(...); dữ liệu/tham số: settings.
        service = ApplicationService(settings)
        # Khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua service.start(...).
        service.start()
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Bảo đảm model đã nạp trước khi xử lý qua service._require_model(...).
            service._require_model()
            # Chỉ chạy khối bên dưới khi args.temporary_gallery cho kết quả đúng.
            if args.temporary_gallery:
                # Gán cho sample: ghép đường dẫn bằng toán tử / của pathlib trong ROOT / 'integration/tests/fixtures/mtcnn_single.png'. Ở đây / nối thư mục với tên file, không phải phép chia số.
                sample = ROOT / "integration/tests/fixtures/mtcnn_single.png"
                # Đăng ký hồ sơ từ danh sách ảnh qua service.enroll(...); dữ liệu/tham số: 'benchmark_fixture', [decode_image(sample.read_bytes(), settings)].
                service.enroll("benchmark_fixture", [decode_image(sample.read_bytes(), settings)])
            # Gán cho report['gallery_size']: đếm số phần tử bằng len(...); dữ liệu/tham số: service.persons() (đọc danh sách hồ sơ qua service.persons(...)).
            report["gallery_size"] = len(service.persons())
            # Gán cho report['model_version_sha256']: lấy mã băm dạng chữ hệ 16 qua hashlib.sha256(service.runtime.version.encode()).hexdigest(...).
            report["model_version_sha256"] = hashlib.sha256(service.runtime.version.encode()).hexdigest()
            # Lặp qua args.images; mỗi lượt gán phần tử cho path.
            for path in args.images:
                # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
                try:
                    # Thêm một phần tử vào cuối danh sách qua report['cases'].append(...); dữ liệu/tham số: run_case(service, path, args.repeat, args.warmup) (đo lặp lại thời gian xử lý một ảnh bằng run_case(...)).
                    report["cases"].append(run_case(service, path, args.repeat, args.warmup))
                # Bắt lỗi (ApplicationError, OSError) và đặt tên error để xử lý trong khối bên dưới.
                except (ApplicationError, OSError) as error:
                    # Thêm một phần tử vào cuối danh sách qua report['cases'].append(...); dữ liệu/tham số: {'image': path.name, 'status': 'blocked', 'reason': str(error)}.
                    report["cases"].append({"image": path.name, "status": "blocked", "reason": str(error)})
            # Gán cho report['status']: chọn 'measured' nếu all((c['status'] == 'measured' for c in report['cases'…; ngược lại chọn 'partial'.
            report["status"] = "measured" if all(c["status"] == "measured" for c in report["cases"]) else "partial"
        # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
        except ApplicationError as error:
            # Tạo thay đổi thuộc tính thành phần giao diện qua report.update(...); dữ liệu/tham số: status='blocked', reason=error.message, error_code=error.code.
            report.update(status="blocked", reason=error.message, error_code=error.code)
    # Gán cho seen: tạo tập hợp bằng cách duyệt case trong report['cases']; mỗi phần tử là count.
    seen = {count for case in report["cases"] for count in case.get("detected_faces_each_run", [])}
    # Gán cho report['face_count_coverage']: tạo từ điển bằng vòng lặp rút gọn; khóa là str(n), giá trị là 'measured' if n in seen else 'blocked: no verified inp….
    report["face_count_coverage"] = {str(n): "measured" if n in seen else "blocked: no verified input with this face count" for n in (1, 2, 5)}
    # Gán cho output: kết hợp bằng or: lấy giá trị đúng đầu tiên nếu có trong args.output or ROOT / 'application/reports' / ('benchmark_' + datetime.now(timezone.….
    output = args.output or ROOT / "application/reports" / ("benchmark_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8] + ".json")
    # Tạo thư mục và tùy chọn các thư mục cha qua output.parent.mkdir(...); dữ liệu/tham số: parents=True, exist_ok=True.
    output.parent.mkdir(parents=True, exist_ok=True)
    # Mở khối with để mở file hoặc ảnh theo chế độ chỉ định qua output.open(...); dữ liệu/tham số: 'x', encoding='utf-8'; gọi tài nguyên là stream; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with output.open("x", encoding="utf-8") as stream:
        # Ghi đối tượng Python thành JSON vào file bằng json.dump(...); dữ liệu/tham số: report, stream, ensure_ascii=False, indent=2.
        json.dump(report, stream, ensure_ascii=False, indent=2)
    # In thông tin ra terminal bằng print(...); dữ liệu/tham số: f'{report['status']}: {output}'.
    print(f'{report["status"]}: {output}')
    # Kết thúc hàm và trả chọn 0 nếu report['status'] == 'measured'; ngược lại chọn 2.
    return 0 if report["status"] == "measured" else 2


# Chỉ chạy khối bên dưới khi __name__ == '__main__' cho kết quả đúng.
if __name__ == "__main__":
    # Phát sinh lỗi: gọi SystemExit(...) với main(); dữ liệu/tham số: main() (gọi main(...)).
    raise SystemExit(main())
