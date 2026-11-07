# Face Recognition Studio

Ứng dụng nhận diện khuôn mặt chạy trên CPU, có giao diện Gradio và API FastAPI. MTCNN phát hiện và căn chỉnh khuôn mặt. FaceNet tạo vector đặc trưng để đối chiếu với hồ sơ đã lưu trong SQLite.

| Tính năng | Cách sử dụng |
| --- | --- |
| **Đăng ký webcam · 4 góc** | Nhập ID và họ tên, bấm **Bắt đầu phiên đăng ký**, bật camera và bấm ghi hình. Lần lượt nhìn thẳng, quay trái, quay phải và ngẩng đầu. Ứng dụng tự chụp khi giữ đúng tư thế. Xem lại 4 ảnh, chụp lại nếu cần rồi xác nhận lưu. |
| **Nhận diện ảnh** | Chọn ảnh JPG, PNG hoặc WebP để xem kết quả đối chiếu từng khuôn mặt. |
| **Webcam trực tiếp** | Bật camera, bấm **Bắt đầu nhận diện** rồi bấm ghi hình trong khung camera. |
| **Danh sách đăng ký** | Xem ID, họ tên và số ảnh. Chọn hồ sơ, bấm **Xóa đăng ký** rồi **Xác nhận xóa**. |

## Cài đặt trên Windows

Mở PowerShell tại thư mục gốc dự án. Môi trường đã kiểm tra là **Python 3.12.6, Windows x64, CPU**. Nếu đã có `.venv` chạy được, chuyển thẳng đến bước kiểm tra model.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r application/requirements-cpu-py312.txt
.\.venv\Scripts\python.exe -m pip check
```

Nếu đã có môi trường ML tương thích và chỉ cần cài tầng web, dùng `application/requirements.txt`. Bộ `requirements-lock.txt` ở thư mục gốc phục vụ runtime nghiên cứu Python 3.13. Xem [hướng dẫn môi trường](application/README.md) để chọn đúng bộ thư viện.

### Kiểm tra model

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m application check-assets
```

MTCNN đọc weights trong `MTCNN/weights/`. FaceNet dùng `FaceNet/models/facenet.onnx`. Nếu ONNX bị thiếu hoặc chỉ là Git LFS pointer, chạy các lệnh sau để tải và xác minh đúng bản model.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m application restore-model
.\.venv\Scripts\python.exe -B -X utf8 -m application check-assets
```

## Chạy ứng dụng

Tạo kho hồ sơ một lần khi bắt đầu với file mới. Nếu đã có kho cần dùng, truyền đường dẫn đó vào `--registry` ở lệnh chạy.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m application init-registry --registry application/data/registry.sqlite
.\.venv\Scripts\python.exe -B -X utf8 -m application serve --registry application/data/registry.sqlite --port 8000
```

- [Giao diện web](http://127.0.0.1:8000/ui/)
- [Tài liệu API Swagger](http://127.0.0.1:8000/docs)
- [Trạng thái hệ thống](http://127.0.0.1:8000/health)

Bản demo cục bộ đang dùng cổng **8002** với `application/data/task01-demo.sqlite`. Có thể mở trực tiếp nếu tiến trình demo đang chạy. [Mở bản demo](http://127.0.0.1:8002/ui/)

Trình duyệt cần cấp quyền camera cho trang. Webcam hoạt động trên localhost hoặc HTTPS. Dùng `--api-only` nếu chỉ cần FastAPI và nhấn **Ctrl+C** để dừng máy chủ.

### Kho hồ sơ

SQLite lưu ID, số ảnh, phiên bản model và vector đặc trưng đại diện. Họ tên nằm trong file `<registry>.names.sqlite` cạnh kho chính. Ảnh chụp 4 góc được giữ trong bộ nhớ của phiên để xem lại trước khi lưu. Khi sao lưu hoặc chuyển kho, dừng ứng dụng rồi sao chép cả hai file SQLite. Dùng cùng model và runtime đã tạo hồ sơ để giữ phiên bản tương thích.

## API chính

| API | Chức năng |
| --- | --- |
| `GET /health` | Kiểm tra backend, model và kho hồ sơ |
| `GET /persons` | Lấy danh sách ID, họ tên nếu có và số ảnh |
| `DELETE /persons/{person_id}` | Xóa hồ sơ theo ID |
| `POST /recognize` | Nhận diện ảnh từ trường multipart `image` |
| `POST /enrollment-sessions` | Tạo phiên đăng ký bằng `person_id` và `display_name` |
| `POST /enrollment-sessions/{token}/frames` | Gửi khung hình và `revision` của phiên |
| `GET /enrollment-sessions/{token}/review` | Xem ảnh đã chụp và revision hiện tại |
| `POST /enrollment-sessions/{token}/confirm` | Xác nhận lưu bằng `confirmed` và `revision` |

Thử nhận diện bằng ảnh mẫu khi server đang chạy ở cổng 8000.

```powershell
curl.exe -X POST http://127.0.0.1:8000/recognize -F "image=@integration/tests/fixtures/mtcnn_single.png"
```

## Cấu trúc dự án

```text
application/   Giao diện, API, luồng webcam và kiểm thử ứng dụng
FaceNet/       Nạp model ONNX, tạo embedding, so khớp và lưu hồ sơ
MTCNN/         Phát hiện khuôn mặt, căn chỉnh và weights
integration/   Pipeline kết nối hai model, kiểm thử và báo cáo đánh giá
```

## Kiểm thử

Chạy kiểm thử ứng dụng và kiểm tra tính nhất quán của ảnh đăng ký. Lệnh dùng kho dữ liệu riêng trong thư mục tạm của lần chạy.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$testTemp = Join-Path 'application/data' ('pytest-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -q -p no:cacheprovider -c application/pytest.ini application/tests integration/tests/test_webcam_identity.py --basetemp $testTemp
```

Một số bài kiểm thử lịch sử của pipeline cần notebook, dữ liệu mẫu và runtime tại các thư mục nguồn cũ. Xem [hướng dẫn ứng dụng và benchmark](application/README.md), [báo cáo đăng ký webcam](application/TASK01_REPORT.md), [tài liệu tích hợp](UI_HANDOFF.md) và [chỉ mục đánh giá model](integration/reports/evidence_index.json).
