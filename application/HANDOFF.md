# Bàn giao Người 3 — 2026-11-05

Đã triển khai application trực tiếp trên repository, khôi phục ONNX gốc, chạy tests và benchmark thật. **Chưa tuyên bố nghiệm thu 5/5**: camera vật lý/trình duyệt đang chờ người dùng kiểm chứng.

## 1. Code audit và cách tích hợp

Audit trước lập trình nằm tại [AUDIT.md](AUDIT.md), bao gồm phạm vi file đã đọc và bảng API/input/output. Application gọi `load_facenet_model`, `build_model_version`, `initialize_enrollment_registry`, `enroll_person_from_images`, `recognize_image`; list/delete dùng API SQLite có sẵn.

Luồng thực tế: ảnh RGB uint8 → MTCNN detect → alignment 5 điểm 160×160 RGB uint8 → adapter → FaceNet preprocessing duy nhất → embedding float32 (512,) L2 → SQLite representatives → cosine/top-1/threshold core → allowlist public response → vẽ box/tên.

Ngưỡng V2 vẫn là **0.665513450766159**. Model_version luôn lấy JSON canonical từ core, không gán thủ công. SHA descriptor của runtime được kiểm thử là `47be33e07cc5dad54d4848463758e3191ebda37696d84109066e6a71f330dc16`; đây chỉ là mã tham chiếu báo cáo, không thay chuỗi model_version trong DB.

Đã xác minh **22/22** mục source bằng hashes bàn giao trong migration module, gồm base enrollment source. Không đổi thuật toán, matching, schema, threshold hoặc source/tests core. [core-source-check.json](reports/core-source-check.json) giữ kết quả đối chiếu. Reports lịch sử trong `integration/reports/` được giữ nguyên.

Vấn đề xử lý ở tầng application: ID `Unknown` bị chặn khi enrollment; giới hạn file/pixel/face count; phân biệt lỗi inference với Unknown; mutex model và file lock registry; lọc embedding/candidate khỏi output; tạo registry bằng CLI chủ động.

## 2. Files changed

| File mới/sửa | Mục đích |
|---|---|
| `application/AUDIT.md` | Audit hoàn tất trước code application |
| `application/{__init__,config,errors}.py` | Package, cấu hình limits/runtime, lỗi public có code/status |
| `application/assets.py` | Chặn thiếu model, LFS pointer hoặc sai SHA/size |
| `application/__main__.py` | CLI check-assets, restore-model có verify, init-registry, serve |
| `application/images.py` | Decode JPEG/PNG/WebP, kiểm tra kích thước trước load, EXIF→RGB uint8 |
| `application/service.py` | Cầu nối duy nhất giữa UI/API/core, model load một lần, locks và CRUD |
| `application/api.py` | 5 endpoints yêu cầu, body limits trước multipart parser, lifespan, mount UI |
| `application/ui.py` | Enrollment/list/delete có xác nhận, upload recognition, webcam `.stream`, Start/Stop và kiểm tra quyền camera |
| `application/{webcam,visualization}.py` | Xử lý frame liên tiếp, resize CPU, timing/FPS theo session, vẽ kết quả |
| `application/benchmark.py` | Warm-up, perf_counter, stage wrappers, mean/p50/p95/FPS và JSON không overwrite |
| `application/tests/{conftest,test_application,test_real_smoke}.py` | DI transport/concurrency, SQLite thật, model thật và tính nhất quán với core |
| `application/tests/fixtures/{group_two_faces.png,MANIFEST.json}` | Crop cố định từ fixture bundled để đo 2 mặt, có provenance/hash |
| `application/pytest.ini` | Chạy tests application/legacy từ root, không dùng cấu hình MTCNN làm root chung |
| `application/requirements*.txt` | Dependencies UI/API và runtime CPU Python 3.12 đã kiểm thử |
| `application/README.md`, `HANDOFF.md` | Lệnh Windows, demo, hợp đồng API, báo cáo nghiệm thu |
| `application/reports/*` | Evidence mới: test XML/log, benchmark JSON, verification summary và source hashes |
| `FaceNet/models/facenet.onnx` **được khôi phục** | Thay pointer bằng đúng artifact gốc sau verify SHA/size |
| `README.md` **được sửa** | Liên kết hướng dẫn application mới, cập nhật mô tả có API/UI |
| `.gitignore` **được sửa** | Bỏ qua SQLite vận hành/demo, pointer backup và downloads tạm |

`.venv` và `application/data/demo.sqlite` là artifacts local đã tạo để chạy demo, nằm trong gitignore. Bản ZIP không có `.git`, nên không có git diff/commit để báo. Pointer ban đầu được giữ tại `FaceNet/models/facenet.onnx.lfs-pointer`.

## 3. Checklist Người 3

- [x] Backend API hoạt động: TestClient và HTTP thật `/health`, `/docs`, `/ui/`, `/ui/config` đều 200.
- [x] Đăng ký người mới: pipeline thật lưu embedding SQLite; readback shape/dtype/norm đúng; duplicate/ảnh lỗi không overwrite/ghi từng phần.
- [ ] Upload ảnh và webcam: upload/model/frame processing đã pass; **camera browser vật lý chưa xác nhận**. Streaming UI đã xây/mount, Start/Stop callback được test; chưa được xem là webcam demo hoàn tất.
- [x] Tích hợp toàn bộ MTCNN + FaceNet qua core: smoke thật một/nhiều mặt, Known/Unknown, registry rỗng, so sánh kết quả với core.
- [x] Integration tests và benchmark FPS/latency: 37 tests application pass; benchmark thật đủ 1/2/5 mặt.

Tình trạng nghiệm thu: **4/5 mục có bằng chứng hoàn tất; webcam cần kiểm chứng trực tiếp**.

## 4. Verification đã thực sự chạy

### Assets/runtime

Tải URL gốc vào `.download`, kiểm tra **93987560 bytes** và SHA-256 **f6688ba6c95465fac18df53fdf8e094048fa3f1dd2c09f1cc6dd734a1c8dedf8**, sau đó mới thay pointer. `FaceNet.facenet_model.load_facenet_model` nạp thành công. `python -m application check-assets` nạp được cả hai model. `pip check`: **No broken requirements found**.

Runtime thực: Windows 11 x64, Python 3.12.6, NumPy 2.4.6, opencv-python 4.10.0.84, Pillow 12.2.0, torch 2.14.0+cpu, FastAPI 0.162.0, Gradio 6.31.0. CPU inference, 2 threads cho Torch/OpenCV, 22 logical CPUs. Tên CPU chi tiết không đo được: API thông tin phần cứng bị từ chối quyền trong môi trường công cụ; JSON ghi `cpu: not_measured`.

### Application tests

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:GRADIO_ANALYTICS_ENABLED='False'
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -q -p no:cacheprovider -c application/pytest.ini application/tests --tb=short --junitxml=application/reports/application-tests-final.xml
```

**37 passed, 0 failed, 0 errors, 0 skipped**, 53.19s. [XML](reports/application-tests-final.xml).

Real smoke thực hiện enrollment/readback, nhận diện lại fixture, ảnh nhóm 5 mặt với cả accepted/rejected decisions ở V2, registry rỗng, ảnh không mặt, từ chối group enrollment, so sánh output application với direct core, ba frame liên tiếp, deletion và stage profiling. Nhận diện lại ảnh enrollment chỉ là correctness smoke, không đánh giá generalization. Camera browser không nằm trong các assertion đó.

Đã tái hiện lỗi Gradio `Image.stop_recording` không tồn tại, test UI build fail, rồi sửa tối thiểu sang event `Image.clear` được hỗ trợ và dùng nút Stop riêng với `cancels`. Lần chạy cuối pass.

### Suite cũ nguyên trạng sau khôi phục model

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -q -p no:cacheprovider -c application/pytest.ini --tb=line FaceNet/tests integration/tests MTCNN/tests --junitxml=application/reports/legacy-after-model.xml
```

Raw pytest: **176 passed, 56 failed, 55 errors, 1 skipped**, 59 subtests passed, 57.00s. [Log](reports/legacy-after-model.log), [XML](reports/legacy-after-model.xml).

Pytest tính subtest failures trong tổng failed; không cộng XML tests trực tiếp thành số test methods. Phân loại 111 failure/error entries: **108** liên quan source/notebooks/IMM/baseline/cwd `D:/Facenet-Recognition` thiếu, **3** là migration gate từ chối runtime khác frozen profile. Một test adapter real-MTCNN skip vì ảnh nguồn `D:/MTCNN` thiếu. Các ca lỗi/skip này không được đổi thành passed. Baseline trước khôi phục pointer nằm trong AUDIT, là một lần chạy khác.

### HTTP và UI

Server đã được khởi động trên `127.0.0.1:8000` với registry demo riêng qua CLI. HTTP kiểm tra thực tế: `/health` 200, models ready, registry empty; `/docs` 200; `/ui/` 200; `/ui/config` 200. Startup không tạo lại registry.

UI build/mount và callbacks được test; việc xem layout/camera bằng công cụ tự động bị chặn: browser host unavailable, sau đó truy cập Chrome hết thời gian chờ phê duyệt. Người dùng đã trả lời sẽ tự kiểm tra webcam, nhưng chưa cung cấp kết quả. Vì vậy chưa có bằng chứng camera browser trực tiếp.

## 5. Benchmark thực

Lệnh cuối:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m application.benchmark --temporary-gallery --repeat 20 --warmup 3
```

[JSON đầy đủ](reports/benchmark_20261105T091114Z_f7f07034.json). Gallery 1 identity tạo tạm từ fixture; threshold V2 nguyên bản. Mỗi input có 3 lượt warm-up + 20 lượt đo; số mặt ổn định 1/2/5 trong mọi lượt. Ảnh 2 mặt là crop có manifest từ ảnh nhóm, chỉ đo throughput.

| Mặt | Kích thước | E2E mean ms | p50 ms | p95 ms | FPS thực đo |
|---:|---:|---:|---:|---:|---:|
| 1 | 160×160 | 229.12 | 235.01 | 264.35 | 4.36 |
| 2 | 326×300 | 372.76 | 317.57 | 513.73 | 2.68 |
| 5 | 666×409 | 608.54 | 606.62 | 788.89 | 1.64 |

| Mặt | Detection mean ms | Alignment mean ms | FaceNet preprocessing mean ms | Embedding mean ms | Matching mean ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 22.82 | 0.67 | 0.12 | 56.73 | 0.19 |
| 2 | 78.62 | 1.27 | 0.28 | 124.90 | 0.37 |
| 5 | 154.24 | 2.82 | 0.79 | 288.32 | 1.15 |

JSON có p50/p95 từng stage và raw samples. Embedding trừ preprocessing; matching gồm cosine/ranking/threshold; E2E gồm service và visualization, registry/version hashing/validation/adapter/locks. Không gồm decode file, startup, HTTP hoặc camera/browser transport. Wrappers profiling có overhead; đây là đo desktop CPU tại thời điểm chạy, không phải latency camera trực tiếp.

Lần `benchmark_20261105T035134Z_4d4dba8d.json` trước đó chạy cùng lúc test và chỉ có 1/5 mặt, được giữ làm lịch sử; bảng trên dùng riêng lần cuối đủ 1/2/5 mặt.

## 6. Demo commands và phần còn cần xác minh

Model đã khôi phục và registry demo đã có. Có thể chạy lại bằng terminal riêng, chọn port khác nếu server 8000 còn hoạt động:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m application check-assets
.\.venv\Scripts\python.exe -B -X utf8 -m application serve --registry application/data/demo.sqlite --port 8001
```

Mở `http://127.0.0.1:8001/ui/` hoặc `/docs`. Muốn registry vận hành mới, chạy `python -m application init-registry` một lần rồi `python -m application serve`. Toàn bộ lệnh dùng đường dẫn Python `.venv` và các bước demo chi tiết ở [README.md](README.md).

Phần còn chưa xác minh:

1. **Webcam vật lý**: cần thấy video/box, frames tăng, Start/Stop có hiệu lực, xử lý từ chối quyền camera trên trình duyệt thực. Người dùng đang kiểm tra. Không tự đánh dấu pass từ HTTP hoặc frame tests.
2. **Legacy regressions**: cần nguồn notebook/IMM/baseline gốc và frozen runtime phù hợp để chạy các tests còn blocked. Không tạo dữ liệu oracle giả hoặc sửa tests để pass.
3. **Cài sạch máy mới**: package versions được ghi và môi trường hiện tại đã pip-check, nhưng chưa dựng một Windows khác từ đầu.
4. **Accuracy/production**: không có evaluation độc lập mới; fixtures correctness/benchmark không chứng minh độ chính xác ngoài tập, robustness hoặc hiệu năng triển khai nhiều người dùng. Phạm vi hiện tại là demo local một worker, writers tuân file lock.
