# Audit trước triển khai — 2026-11-04

Audit này được viết trước application code. Bản đang làm việc là ZIP, không có `.git`, `.venv` hoặc registry vận hành.

## Phạm vi đã đọc

- `README.md`, `UI_HANDOFF.md`, requirements integration/test/lock và MTCNN.
- `MTCNN/mtcnn.py`, `model_loader.py`, toàn bộ `preprocessing/*.py`, `detection/*.py`, `models/*.py`, `tests/*.py`, README và TESTING.
- `FaceNet/{__init__,facenet_model,database,enrollment,recognition,matching,contracts}.py`, toàn bộ tests và `_support.py`.
- `integration/{mtcnn_adapter,enrollment_pipeline,recognition_pipeline,recognition_runtime}.py`, toàn bộ tests, fixture manifest và `scripts/run_verification.py`.
- Các phần provenance/version trong `integration/reports/evidence_index.json` và pipeline profile. Evidence lịch sử không phải kết quả chạy mới.

## API thực tế tái sử dụng

| Module | Lời gọi | Hợp đồng |
|---|---|---|
| MTCNN | `MTCNN(device="cpu")`; `detect(rgb)` | RGB uint8 HWC; boxes N×4 xyxy half-open, scores N, landmarks N×5×2 |
| Alignment | `extract_faces_rgb(rgb, result, size=160, align=True)` | N×160×160×3 RGB uint8; không normalize; giữ face order |
| Adapter | `mtcnn_to_facenet_detections(rgb, result, faces)` | Clip bản sao bbox theo ảnh gốc; lược landmarks âm; không bỏ face |
| FaceNet public | `load_facenet_model(absolute_path)` | OpenCV DNN CPU; input in:0 / output out:0 |
| FaceNet preprocessing | `extract_embedding(face, net)` | `(pixel-127.5)/128`, NCHW; embedding float32 (512,), L2 với norm float64 |
| Integration public | `build_model_version(detector)` | JSON canonical chứa hashes source/weights + runtime, không phải chỉ hash model |
| Integration public | `initialize_enrollment_registry(absolute_path)` | Thao tác chủ động; từ chối file có sẵn |
| Integration public | `enroll_person_from_images(id, images, detector, net, path, version)` | N>0; đúng 1 face/ảnh; mean float64 rồi L2; INSERT atomic sau toàn bộ inference |
| Runtime public | `recognize_image(rgb, detector, net, path, version)` | Mặc định V2 0.665513450766159; không override từ application |
| Database | `list_people`, `get_person(..., expected_model_version=version)`, `delete_person` | Schema/checksum/version validation; SQLite connections theo call |

Recognition giữ thứ tự detector, trả face_index/bbox/confidence/status/person_id/similarity/threshold. Unknown có person_id=None; registry rỗng hợp lệ có similarity=None; ảnh không mặt trả []. Lỗi inference phải là lỗi request, không phải Unknown. Enrollment trả cả embedding: application bắt buộc lọc qua allowlist trước trả client.

## Phát hiện và quyết định

1. ONNX chỉ **133 bytes**, Git LFS pointer. Model thực phải **93,987,560 bytes**, SHA-256 `f6688ba6c95465fac18df53fdf8e094048fa3f1dd2c09f1cc6dd734a1c8dedf8`. Evidence gốc ghi URL `https://media.axelera.ai/artifacts/model_cards/weights/face_recognition/facenet.onnx`. Có thể khôi phục đúng artifact này sau xác minh hash; không lấy model khác. P/R/O weights có sẵn.
2. Runtime hiện có Python 3.12.6, numpy 2.4.6, opencv-python 4.10.0.84, Pillow 12.2.0, torch 2.14.0+cpu. Khác frozen Python 3.13/numpy 2.1.3/torch 2.12.1. Không relabel/migrate registry cũ; version phải do core tạo trên môi trường thực. FastAPI và Gradio chưa cài.
3. Lock gốc dành Windows CPython 3.13; không áp thẳng lên Python 3.12. MTCNN requirements đề nghị headless, nhưng version builder yêu cầu metadata `opencv-python`: không cài đồng thời hai distribution OpenCV.
4. `build_model_version` băm cả ONNX mỗi request và toàn bộ core source. Không cache/thay thuật toán fingerprint; benchmark phải bao gồm overhead này.
5. OpenCV `Net.setInput/forward` dùng state chung. Core yêu cầu registry ổn định xuyên nhiều SELECT. Application cần mutex cho model + file lock cho tất cả đọc/ghi registry qua application; một worker phục vụ UI và API. Không hỗ trợ writer bên ngoài không tuân lock.
6. Core enrollment vẫn chấp nhận ID `Unknown` nhưng recognition từ chối registry chứa ID đó. Application sẽ chặn ID dành riêng tại biên, không sửa matching/schema/core.
7. Core ném ValueError cho cả input và output model. Không map tất cả ValueError thành 422: validate input/face count riêng, còn lỗi inference/profile/schema trả lỗi hạ tầng rõ ràng.
8. Các tests cũ phụ thuộc `D:/Facenet-Recognition` notebooks/IMM/baseline và `D:/MTCNN`. Runner cũ còn gọi pytest ở ổ D:. Giữ nguyên tests và runner; thêm tests application portable, smoke thật trên fixtures bundled.
9. Không có bằng chứng webcam trong repo. Cần streaming liên tiếp, giới hạn backlog, Start/Stop và phân biệt FPS xử lý với FPS trình duyệt.

## Baseline đã chạy trước triển khai

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -X utf8 -m pytest -q -p no:cacheprovider --tb=line FaceNet/tests integration/tests MTCNN/tests
```

Pytest exit 1, **169 passed, 50 failed, 68 errors, 1 skipped**, 59 subtests passed, 28.91s (pytest 9.1.1 tính cả subtest failures trong 50). Lỗi chủ yếu: nguồn D: thiếu, ONNX pointer, migration gate từ chối runtime/source không khớp frozen profile. Đây là kết quả raw, không đổi thành pass/skip. MTCNN pretrained smoke thuộc các tests đã chạy. Báo cáo cuối sẽ tách kiểm chứng mới khỏi baseline này.

## Thiết kế phần Người 3

Thêm riêng `application/`: service, FastAPI, Gradio streaming, CLI registry/assets, benchmark và tests. Không sửa bytes của core, tests cũ hoặc reports lịch sử. UI và API dùng chung service trong một process; startup chỉ load model, không tạo/xóa/migrate registry. Benchmark dùng chính recognize_image, đo boundaries bằng instrumentation trong process benchmark riêng; báo not_measured cho stage không đo được và blocked khi thiếu prerequisites.
