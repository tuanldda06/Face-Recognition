# Điểm chạy của lệnh python -m application: kiểm tra model, tải model, tạo kho hoặc mở máy chủ.
# Nạp argparse để đọc tham số người dùng nhập trong terminal.
import argparse
# Lấy replace từ dataclasses để tạo lớp dữ liệu và sao chép/chuyển đổi dữ liệu của lớp.
from dataclasses import replace
# Nạp json để chuyển đổi dữ liệu Python với JSON.
import json
# Lấy Path từ pathlib để biểu diễn đường dẫn và thao tác file bằng đối tượng Path.
from pathlib import Path
# Nạp sys để truy cập đường dẫn import, tham số và luồng báo lỗi của Python.
import sys
# Nạp tempfile để tạo file/thư mục tạm có kiểm soát vòng đời.
import tempfile
# Nạp urllib.request để tải file model qua HTTP.
import urllib.request

# Lấy verify_model từ .assets; dấu . chỉ module cùng package.
from .assets import verify_model
# Lấy MODEL_PATH, MODEL_SIZE, MODEL_URL, Settings từ .config; dấu . chỉ module cùng package.
from .config import MODEL_PATH, MODEL_SIZE, MODEL_URL, Settings
# Lấy ApplicationError từ .errors; dấu . chỉ module cùng package.
from .errors import ApplicationError


# Tải model vào file tạm; chỉ thay file đích sau khi kiểm tra toàn vẹn thành công.
# Định nghĩa hàm restore_model().
def restore_model():
    # Chỉ chạy khối bên dưới khi MODEL_PATH.exists() cho kết quả đúng.
    if MODEL_PATH.exists():
        # Mở khối with để mở file hoặc ảnh theo chế độ chỉ định qua MODEL_PATH.open(...); dữ liệu/tham số: 'rb'; gọi tài nguyên là stream; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with MODEL_PATH.open("rb") as stream:
            # Gán cho is_pointer: kiểm tra phần đầu chuỗi qua stream.read(200).startswith(...); dữ liệu/tham số: b'version https://git-lfs.github.com/spec/v1'.
            is_pointer = stream.read(200).startswith(b"version https://git-lfs.github.com/spec/v1")
        # Chỉ chạy khối bên dưới khi not is_pointer cho kết quả đúng.
        if not is_pointer:
            # Kết thúc hàm và trả xác minh file model bằng dung lượng và SHA-256 bằng verify_model(...).
            return verify_model()  # Không ghi đè nếu đây là model thật nhưng khác bản dự kiến.
    # Giữ file tải tạm khi xác minh thất bại để có thể kiểm tra nguyên nhân.
    # Mở khối with để tạo file tạm có tên và chế độ dọn theo tùy chọn bằng tempfile.NamedTemporaryFile(...); dữ liệu/tham số: prefix='facenet_', suffix='.download', dir=MODEL_PATH.parent, delete=False; gọi tài nguyên là temporary; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with tempfile.NamedTemporaryFile(prefix="facenet_", suffix=".download", dir=MODEL_PATH.parent, delete=False) as temporary:
        # Gán cho download: tạo đối tượng đường dẫn bằng Path(...); dữ liệu/tham số: temporary.name.
        download = Path(temporary.name)
        # Mở khối with để gọi urllib.request.urlopen(...) với MODEL_URL; dữ liệu/tham số: MODEL_URL, timeout=60; gọi tài nguyên là response; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            # Gán cho size: giá trị 0.
            size = 0
            # Lặp lại khối bên dưới chừng nào (block := response.read(1024 * 1024)) còn đúng.
            while block := response.read(1024 * 1024):
                # Cập nhật size: cộng thêm len(block) và lưu lại vào chính nó.
                size += len(block)
                # Chỉ chạy khối bên dưới khi size > MODEL_SIZE cho kết quả đúng.
                if size > MODEL_SIZE:
                    # Phát sinh lỗi: gọi ValueError(...) với f'Artifact exceeds expected size; …; dữ liệu/tham số: f'Artifact exceeds expected size; pointer preserved. Download: {download}'.
                    raise ValueError(f"Artifact exceeds expected size; pointer preserved. Download: {download}")
                # Ghi dữ liệu vào luồng qua temporary.write(...); dữ liệu/tham số: block.
                temporary.write(block)
    # Gán cho result: xác minh file model bằng dung lượng và SHA-256 bằng verify_model(...); dữ liệu/tham số: download.
    result = verify_model(download)
    # Chỉ chạy khối bên dưới khi MODEL_PATH.exists() cho kết quả đúng.
    if MODEL_PATH.exists():
        # Gán cho backup: tạo đường dẫn với phần đuôi file mới qua MODEL_PATH.with_suffix(...); dữ liệu/tham số: '.onnx.lfs-pointer'.
        backup = MODEL_PATH.with_suffix(".onnx.lfs-pointer")
        # Chỉ chạy khối bên dưới khi not backup.exists() cho kết quả đúng.
        if not backup.exists():
            # Mở khối with để mở file hoặc ảnh theo chế độ chỉ định qua backup.open(...); dữ liệu/tham số: 'xb'; gọi tài nguyên là output; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
            with backup.open("xb") as output:
                # Ghi dữ liệu vào luồng qua output.write(...); dữ liệu/tham số: MODEL_PATH.read_bytes() (đọc toàn bộ file dưới dạng byte qua MODEL_PATH.read_bytes(...)).
                output.write(MODEL_PATH.read_bytes())
    # Thay chuỗi hoặc thay file đích tùy đối tượng qua download.replace(...); dữ liệu/tham số: MODEL_PATH.
    download.replace(MODEL_PATH)
    # Kết thúc hàm và trả result (kết quả xử lý).
    return result


# Phân tích lệnh terminal rồi chạy tác vụ tương ứng; trả mã 0 khi thành công, 1 khi lỗi.
# Định nghĩa hàm main(argv). Mặc định: argv=None.
def main(argv=None):
    # Gán cho parser: gọi argparse.ArgumentParser(...); dữ liệu/tham số: description='Face Recognition application (run from repository root)'.
    parser = argparse.ArgumentParser(description="Face Recognition application (run from repository root)")
    # Gán cho sub: tạo nhóm lệnh con qua parser.add_subparsers(...); dữ liệu/tham số: dest='command', required=True.
    sub = parser.add_subparsers(dest="command", required=True)
    # Khai báo một lệnh con qua sub.add_parser(...); dữ liệu/tham số: 'check-assets'.
    sub.add_parser("check-assets")
    # Khai báo một lệnh con qua sub.add_parser(...); dữ liệu/tham số: 'restore-model'.
    sub.add_parser("restore-model")
    # Gán cho initialize: khai báo một lệnh con qua sub.add_parser(...); dữ liệu/tham số: 'init-registry'.
    initialize = sub.add_parser("init-registry")
    # Khai báo một tùy chọn dòng lệnh qua initialize.add_argument(...); dữ liệu/tham số: '--registry', type=Path.
    initialize.add_argument("--registry", type=Path)
    # Gán cho serve: khai báo một lệnh con qua sub.add_parser(...); dữ liệu/tham số: 'serve'.
    serve = sub.add_parser("serve")
    # Khai báo một tùy chọn dòng lệnh qua serve.add_argument(...); dữ liệu/tham số: '--registry', type=Path.
    serve.add_argument("--registry", type=Path)
    # Khai báo một tùy chọn dòng lệnh qua serve.add_argument(...); dữ liệu/tham số: '--api-only', action='store_true'.
    serve.add_argument("--api-only", action="store_true")
    # Khai báo một tùy chọn dòng lệnh qua serve.add_argument(...); dữ liệu/tham số: '--port', type=int, default=8000.
    serve.add_argument("--port", type=int, default=8000)
    # Khai báo một tùy chọn dòng lệnh qua serve.add_argument(...); dữ liệu/tham số: '--cpu-threads', type=int, default=2.
    serve.add_argument("--cpu-threads", type=int, default=2)
    # Gán cho args: đọc các tham số dòng lệnh qua parser.parse_args(...); dữ liệu/tham số: argv.
    args = parser.parse_args(argv)
    # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
    try:
        # Chỉ chạy khối bên dưới khi args.command == 'check-assets' cho kết quả đúng.
        if args.command == "check-assets":
            # Lấy load_runtime từ .service; dấu . chỉ module cùng package.
            from .service import load_runtime
            # Gán cho runtime: nạp detector và FaceNet theo cấu hình bằng load_runtime(...); dữ liệu/tham số: Settings.from_env() (gọi Settings.from_env(...)).
            runtime = load_runtime(Settings.from_env())
            # Gán cho result: từ điển {**verify_model(), 'models': 'loaded', 'runtime': json.loads(runtime.version)['runti… (ánh xạ khóa → giá trị).
            result = {**verify_model(), "models": "loaded", "runtime": json.loads(runtime.version)["runtime"]}
        # Chỉ chạy khối bên dưới khi args.command == 'restore-model' cho kết quả đúng.
        elif args.command == "restore-model":
            # Gán cho result: gọi restore_model(...).
            result = restore_model()
        # Nhánh else: chạy khi nhánh if trước đó không được chọn (hoặc vòng lặp kết thúc bình thường nếu thuộc vòng lặp).
        else:
            # Lấy ApplicationService từ .service; dấu . chỉ module cùng package.
            from .service import ApplicationService
            # Gán cho settings: gọi Settings.from_env(...).
            settings = Settings.from_env()
            # Chỉ chạy khối bên dưới khi args.registry cho kết quả đúng.
            if args.registry:
                # Gán cho settings: tạo bản cấu hình sao chép với một số trường thay đổi bằng replace(...); dữ liệu/tham số: settings, registry=args.registry.absolute().
                settings = replace(settings, registry=args.registry.absolute())
            # Chỉ chạy khối bên dưới khi args.command == 'init-registry' cho kết quả đúng.
            if args.command == "init-registry":
                # Lấy initialize_enrollment_registry từ FaceNet.
                from FaceNet import initialize_enrollment_registry
                # Gán cho service: tạo dịch vụ dùng chung cho nhận diện và quản lý hồ sơ bằng ApplicationService(...); dữ liệu/tham số: settings.
                service = ApplicationService(settings)
                # Mở khối with để lấy khóa truy cập model và kho hồ sơ qua service.guard(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                with service.guard():
                    # Khởi tạo kho SQLite theo cấu trúc FaceNet bằng initialize_enrollment_registry(...); dữ liệu/tham số: service.path.
                    initialize_enrollment_registry(service.path)
                # Gán cho result: từ điển {'status': 'created', 'registry': str(service.path)} (ánh xạ khóa → giá trị).
                result = {"status": "created", "registry": str(service.path)}
            # Nhánh else: chạy khi nhánh if trước đó không được chọn (hoặc vòng lặp kết thúc bình thường nếu thuộc vòng lặp).
            else:
                # Nạp uvicorn để chạy máy chủ web ASGI cho ứng dụng.
                import uvicorn
                # Lấy create_app từ .api; dấu . chỉ module cùng package.
                from .api import create_app
                # Gán cho settings: tạo bản cấu hình sao chép với một số trường thay đổi bằng replace(...); dữ liệu/tham số: settings, cpu_threads=args.cpu_threads.
                settings = replace(settings, cpu_threads=args.cpu_threads)
                # Gán cho app: tạo API và tùy chọn gắn giao diện bằng create_app(...); dữ liệu/tham số: ApplicationService(settings) (tạo dịch vụ dùng chung cho nhận diện và quản lý hồ sơ bằng ApplicationService(...)), with_ui=not args.api_only.
                app = create_app(ApplicationService(settings), with_ui=not args.api_only)
                # Gọi uvicorn.run(...) với app; dữ liệu/tham số: app, host='127.0.0.1', port=args.port, workers=1, access_log=False.
                uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False)
                # Kết thúc hàm và trả giá trị 0.
                return 0
        # In thông tin ra terminal bằng print(...); dữ liệu/tham số: json.dumps(result, ensure_ascii=False, indent=2) (đổi đối tượng Python thành chuỗi JSON bằng json.dumps(...)).
        print(json.dumps(result, ensure_ascii=False, indent=2))
        # Kết thúc hàm và trả giá trị 0.
        return 0
    # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
    except ApplicationError as error:
        # In thông tin ra terminal bằng print(...); dữ liệu/tham số: json.dumps(error.payload(), ensure_ascii=False) (đổi đối tượng Python thành chuỗi JSON bằng json.dumps(...)), file=sys.stderr.
        print(json.dumps(error.payload(), ensure_ascii=False), file=sys.stderr)
    # Bắt lỗi (ValueError, OSError) và đặt tên error để xử lý trong khối bên dưới.
    except (ValueError, OSError) as error:
        # In thông tin ra terminal bằng print(...); dữ liệu/tham số: f'{type(error).__name__}: {error}', file=sys.stderr.
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
    # Kết thúc hàm và trả giá trị 1.
    return 1


# Chỉ chạy khối bên dưới khi __name__ == '__main__' cho kết quả đúng.
if __name__ == "__main__":
    # Phát sinh lỗi: gọi SystemExit(...) với main(); dữ liệu/tham số: main() (gọi main(...)).
    raise SystemExit(main())
