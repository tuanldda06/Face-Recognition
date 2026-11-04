# Face Recognition — MTCNN + FaceNet

Backend Python CPU cho enrollment và recognition trên ảnh tĩnh. Ứng dụng gọi
Python API; chưa có UI, webcam hoặc server.

## Cấu trúc và trách nhiệm

```text
final_20261102T123214Z_0ae0102924/
  MTCNN/                         # Thành viên 1: detection, alignment, weights
  FaceNet/                       # Thành viên 2: implementation thực tế
    __init__.py
    contracts.py
    facenet_model.py
    matching.py
    database.py
    enrollment.py
    recognition.py
    models/facenet.onnx
    tests/                       # 115 tests FaceNet và helper
  integration/                   # Code dùng chung
    mtcnn_adapter.py
    enrollment_pipeline.py
    recognition_pipeline.py
    recognition_runtime.py
    tests/                       # 120 tests gốc + 10 tests migration; fixtures
    scripts/run_verification.py
    reports/                     # Scientific evidence của pipeline hai model
      history/                   # Hai manifest lịch sử, giữ nguyên bytes
  .venv/
  requirements-integration.txt
  requirements-test.txt
  requirements-lock.txt
  README.md
  UI_HANDOFF.md
```

Implementation FaceNet và ONNX chỉ có một bản; không còn src/ hoặc models/ ở root.
Reports và runner hiện có thuộc pipeline MTCNN + FaceNet nên nằm trong integration/.
Bản final ban đầu không có notebooks hoặc script riêng FaceNet để di chuyển;
không tạo các thư mục con trống. Các tests FaceNet được đưa từ suite nguồn đang
dùng vào FaceNet/tests/; notebook/IMM/database tham chiếu vẫn chỉ đọc ở nguồn.

## Môi trường Windows CPU

Mở terminal tại project root. Không cần sửa sys.path hoặc đặt PYTHONPATH/src:

```powershell
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -B -m pip check
```

Môi trường đã dùng: CPython 3.13.14, NumPy 2.1.3, opencv-python 4.10.0.84,
Pillow 12.2.0, Torch 2.12.1+cpu. .venv đi kèm còn phụ thuộc Windows x64 và
Python hệ thống C:/Program Files/Python313; chưa xác minh portable/fresh-machine install.
Không đổi runtime tùy tiện vì phiên bản thư viện thuộc model_version.

Chỉ khi .venv chưa có, tạo bằng Python tương thích và cài lock:

```powershell
& 'C:\Program Files\Python313\python.exe' -B -m venv .venv
.\.venv\Scripts\python.exe -B -m pip install --require-hashes --no-deps --no-cache-dir -r requirements-lock.txt
```

VS Code: Python: Select Interpreter → Enter interpreter path →
.venv/Scripts/python.exe. Không ghi đè môi trường hiện có.

## Nạp models và gọi backend

```python
from pathlib import Path
from MTCNN.mtcnn import MTCNN
from FaceNet import (
    load_facenet_model, build_model_version,
    initialize_enrollment_registry, enroll_person_from_images, recognize_image,
)

root = Path.cwd().resolve()  # Terminal tại project root.
detector = MTCNN(device="cpu")
facenet_net = load_facenet_model(root / "FaceNet/models/facenet.onnx")
model_version = build_model_version(detector)
```

Tái sử dụng models đã nạp; loader FaceNet dùng OpenCV DNN CPU và path tuyệt đối.
MTCNN dùng bundled weights trong MTCNN/weights. UI gửi RGB uint8 (H,W,3);
backend alignment 160×160, preprocessing (pixel-127.5)/128.0, embedding
float32 (512,) L2, cosine/top-1 và similarity >= threshold.
V2 mặc định 0.665513450766159; threshold_profile="v1" rollback về 0.925533688089963.
Xem [UI_HANDOFF.md](UI_HANDOFF.md) cho enrollment, recognition và errors.

Imports core dùng FaceNet.*; facenet_recognition.* và PYTHONPATH/src đã được thay thế.
Các entry points integration và chữ ký hàm hiện có giữ nguyên.

## Model version và SQLite cũ

model_version là canonical JSON đầy đủ, không phải hash ONNX hoặc SHA descriptor.
Source/path mới có fingerprint mới; không dùng source SHA lịch sử cho code đã đổi.
SHA-256 descriptor với cấu hình CPU mặc định và môi trường trên:

- Trước refactor: 26f1432e29721a4d1e881164a62db603a968a5aa3c0e505afcba29c59668d99b.
- Sau refactor: 6f532f61054054f4ed2a1e432faab58b2bc6a6506f21743f351b1a0e3323494f.

Schema và codec SQLite giữ nguyên. Recognition không tự chấp nhận, sửa hoặc relabel
registry mang version cũ. Nếu cần chuyển đúng frozen profile cũ, caller chủ động dùng
migrate_legacy_registry trong integration.enrollment_pipeline, ghi sang file đích mới.
Cơ chế này chỉ cho phép bộ source đã kiểm chứng, cùng weights/config/runtime;
từ chối mixed/unknown versions, checksum lỗi, target có sẵn hoặc SQLite sidecars.
Xem ví dụ tại UI_HANDOFF. Không chỉnh model_version trực tiếp trong database.
Không có registry vận hành được tạo hoặc chuyển trong lần refactor này.

## Tests

Tại project root:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m integration.scripts.run_verification all
```

Runner chạy 115 tests FaceNet, 130 tests integration và 30 tests MTCNN:
giữ đủ 265 tests baseline, thêm 10 tests migration trong test enrollment hiện có.
Có thể thay all bằng facenet, integration, mtcnn, adapter hoặc embedding.
Runner chặn ghi ngoài thư mục tạm và mạng; database/log tạm được cleanup.

Các tests tham chiếu notebooks, IMM, database/ảnh mẫu gốc ở
D:/Facenet-Recognition và D:/MTCNN trong chế độ chỉ đọc; full suite chưa độc lập
với các fixtures đó. MTCNN tests dùng code bundled trong final.
Runtime không cần các project nguồn; FEI smoke cần dataset ngoài project có quyền sử dụng.
Smoke trên outcomes đã biết là correctness regression, không phải evaluation độc lập mới.

## Evidence và giới hạn

Scientific JSON được di chuyển nguyên bytes vào
[integration/reports/evidence_index.json](integration/reports/evidence_index.json).
Các paths/hashes trong chúng mô tả thời điểm frozen; không sửa theo layout mới.
Đối với references tương đối lịch sử reports/x, bản hiện có ở integration/reports/x.
COPY_MANIFEST.json và final_handoff_manifest.json nằm ở integration/reports/history/,
là hồ sơ lịch sử, không phải inventory mới sau refactor.
Protocol gốc và báo cáo handoff cũ vẫn ở
[EVALUATION_PROTOCOL.md](D:/Face-Recognition-System/EVALUATION_PROTOCOL.md) và
[FINAL_HANDOFF_REPORT.md](D:/Face-Recognition-System/FINAL_HANDOFF_REPORT.md).

ONNX SHA-256 giữ nguyên:
f6688ba6c95465fac18df53fdf8e094048fa3f1dd2c09f1cc6dd734a1c8dedf8.
Giấy phép/provenance model và FEI nằm trong evidence; không suy diễn quyền thương mại.
FEI 45 ảnh chỉ là LIMITED_INDEPENDENT_EVIDENCE, không production-grade accuracy.
Thread safety, concurrent registry writers và UI performance chưa được xác minh;
serialize model access và registry writes.
