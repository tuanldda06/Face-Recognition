# Thư mục application

Thư mục này cung cấp giao diện Gradio và API FastAPI cho Face Recognition Studio. Web hỗ trợ đăng ký qua webcam với 4 góc, nhận diện ảnh, nhận diện trực tiếp và quản lý danh sách hồ sơ. Xem [README tổng của dự án](../README.md) để cài môi trường Python 3.12 trên Windows, kiểm tra model và tạo kho SQLite mới.

## Chạy giao diện

Mở PowerShell tại thư mục gốc dự án. Nếu muốn chạy với kho demo hiện có, dùng lệnh sau.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m application serve --registry application/data/task01-demo.sqlite --port 8002
```

Mở [giao diện web](http://127.0.0.1:8002/ui/) hoặc [tài liệu API](http://127.0.0.1:8002/docs). Nếu máy chủ ở cổng 8002 đã chạy, mở trực tiếp liên kết.

Trong tab **Đăng ký webcam**, nhập ID và họ tên, bắt đầu phiên rồi bật ghi hình. Thực hiện lần lượt **nhìn thẳng → quay trái → quay phải → ngẩng đầu**. Sau khi chụp đủ, xem lại và xác nhận lưu. Tab **Danh sách đăng ký** hiển thị ID, họ tên và số ảnh. Để xóa hồ sơ, chọn hồ sơ rồi bấm **Xóa đăng ký → Xác nhận xóa**.

## Mã ứng dụng

| File | Vai trò |
| --- | --- |
| [`__init__.py`](__init__.py) | Khai báo package Python `application`. |
| [`__main__.py`](__main__.py) | Điểm vào cho `python -m application`. Cung cấp các lệnh `check-assets`, `restore-model`, `init-registry` và `serve`. |
| [`api.py`](api.py) | Khai báo các API kiểm tra hệ thống, nhận diện, đăng ký webcam, liệt kê và xóa hồ sơ. Gắn giao diện Gradio vào `/ui/`. |
| [`ui.py`](ui.py) | Dựng bốn tab Gradio và nối thao tác người dùng với lớp dịch vụ. |
| [`ui.css`](ui.css) | Kiểu hiển thị của trang, khung camera, ô tiến độ và thẻ ảnh theo từng góc. |
| [`service.py`](service.py) | Nạp model, điều phối nhận diện và các thao tác hồ sơ. Dùng khóa để tuần tự hóa truy cập model và SQLite. |
| [`config.py`](config.py) | Cấu hình đường dẫn model, kho mặc định, giới hạn ảnh và số luồng CPU. |
| [`assets.py`](assets.py) | Kiểm tra dung lượng và SHA-256 của model FaceNet ONNX. |
| [`errors.py`](errors.py) | Kiểu lỗi ứng dụng chứa mã lỗi, thông báo và mã HTTP. |
| [`images.py`](images.py) | Giải mã ảnh tải lên sang RGB và kiểm tra định dạng, kích thước, dung lượng. |
| [`head_pose.py`](head_pose.py) | Ước lượng hướng đầu từ các điểm trên khuôn mặt và phân loại tư thế. |
| [`webcam_enrollment.py`](webcam_enrollment.py) | Quản lý phiên đăng ký, kiểm tra chất lượng và độ ổn định, chụp 4 góc, xem lại, chụp lại và xác nhận lưu. |
| [`webcam_registry.py`](webcam_registry.py) | Ghi và xóa vector đại diện cùng họ tên bằng giao dịch SQLite. |
| [`webcam.py`](webcam.py) | Xử lý từng khung hình nhận diện trực tiếp, thu nhỏ ảnh và tính thời gian xử lý, FPS. |
| [`visualization.py`](visualization.py) | Vẽ khung mặt, ID hoặc `Unknown` và điểm tương đồng lên ảnh kết quả. |
| [`benchmark.py`](benchmark.py) | Đo độ trễ và thông lượng nhận diện với ảnh mẫu hoặc kho chỉ đọc. Kết quả được ghi dưới dạng JSON. |

## Cài đặt và kiểm thử

| File hoặc thư mục | Vai trò |
| --- | --- |
| [`requirements-cpu-py312.txt`](requirements-cpu-py312.txt) | Bộ thư viện chạy ứng dụng trên Windows x64 với Python 3.12 và CPU. |
| [`requirements.txt`](requirements.txt) | Phụ thuộc của tầng web cho môi trường ML đã có sẵn. |
| [`pytest.ini`](pytest.ini) | Cấu hình chạy pytest cho tầng ứng dụng. |
| [`tests/conftest.py`](tests/conftest.py) | Thiết lập đường dẫn import khi pytest chạy. |
| [`tests/test_application.py`](tests/test_application.py) | Kiểm tra API, lớp dịch vụ, luồng nhận diện và thao tác xem danh sách, xác nhận xóa hồ sơ. |
| [`tests/test_webcam_enrollment.py`](tests/test_webcam_enrollment.py) | Kiểm tra đăng ký 4 góc, chất lượng ảnh, quản lý phiên, chụp lại và ghi SQLite. |
| [`tests/test_head_pose.py`](tests/test_head_pose.py) | Kiểm tra bộ ước lượng tư thế với các điểm khuôn mặt tổng hợp. |
| [`tests/test_real_smoke.py`](tests/test_real_smoke.py) | Kiểm tra tích hợp bằng MTCNN và FaceNet thật, gồm API nhận diện và đo thời gian xử lý. |
| [`tests/test_real_five_image_persistence.py`](tests/test_real_five_image_persistence.py) | Kiểm tra lưu hồ sơ mới có 4 ảnh và đọc hồ sơ cũ có 5 ảnh với model thật. |
| [`tests/fixtures/group_two_faces.png`](tests/fixtures/group_two_faces.png) | Ảnh mẫu hai khuôn mặt dùng trong kiểm thử và benchmark. |
| [`tests/fixtures/MANIFEST.json`](tests/fixtures/MANIFEST.json) | Ghi nguồn, vùng cắt và checksum của ảnh mẫu. |

Chạy kiểm thử từ thư mục gốc. Chỉ định thư mục tạm riêng để các bài test dùng kho SQLite độc lập.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$testTemp = Join-Path 'application/data' ('pytest-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -q -p no:cacheprovider -c application/pytest.ini application/tests --basetemp $testTemp
```

## Dữ liệu và tài liệu bàn giao

| File hoặc thư mục | Vai trò |
| --- | --- |
| [`data/task01-demo.sqlite`](data/task01-demo.sqlite) | Kho demo đang dùng. Lưu ID, số ảnh, vector đại diện và phiên bản model. |
| [`data/task01-demo.sqlite.names.sqlite`](data/task01-demo.sqlite.names.sqlite) | Kho họ tên đi kèm. Khi sao lưu, dừng ứng dụng rồi sao chép cả hai file SQLite. |
| [`data/`](data/) | Dữ liệu vận hành và file sinh khi chạy. `task01-server.pid` ghi PID của tiến trình khởi chạy, còn `registration-list-stdout.log` và `registration-list-stderr.log` là log của lần chạy hiện tại. |
| [`reports/`](reports/) | Kết quả kiểm thử, benchmark và các bản đối chiếu dùng khi bàn giao. File `benchmark_*.json` chứa số liệu đo. Các file `*.xml` và `*.log` còn lại ghi kết quả từng bộ kiểm thử. Các file JSON tổng hợp ghi trạng thái xác minh, mốc kiểm tra HTTP, dấu vân tay mã nguồn và danh sách bàn giao. |
| [`AUDIT.md`](AUDIT.md) | Rà soát mã, môi trường và các ràng buộc trước khi triển khai tầng ứng dụng. |
| [`HANDOFF.md`](HANDOFF.md) | Hồ sơ bàn giao tầng ứng dụng ban đầu, gồm bằng chứng kiểm thử và benchmark. |
| [`TASK01_AUDIT.md`](TASK01_AUDIT.md) | Rà soát và phân công công việc cho tính năng đăng ký qua webcam. |
| [`TASK01_REPORT.md`](TASK01_REPORT.md) | Lịch sử triển khai và kết quả kiểm thử đăng ký webcam, gồm lần chuyển từ 5 góc sang 4 góc. |
