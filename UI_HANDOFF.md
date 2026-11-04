# UI handoff — API cho thành viên 3

MTCNN thuộc thành viên 1, implementation trong FaceNet/ thuộc thành viên 2;
integration/ điều phối hai model. Setup và tests tại [README.md](README.md).
Chạy ứng dụng từ project root bằng Python .venv; không thêm src vào sys.path.

## Imports và model loading

```python
from pathlib import Path
import cv2
from MTCNN.mtcnn import MTCNN
from FaceNet import (
    load_facenet_model, build_model_version,
    initialize_enrollment_registry, enroll_person_from_images, recognize_image,
)

root = Path.cwd().resolve()
detector = MTCNN(device="cpu")
facenet_net = load_facenet_model(root / "FaceNet/models/facenet.onnx")
model_version = build_model_version(detector)

def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
```

Nạp models một lần, giữ detector/network giữa các calls và serialize truy cập.
Dùng đúng bundled weights; FaceNet loader cần path tuyệt đối.
Core APIs cũng được import trực tiếp qua FaceNet.facenet_model,
FaceNet.database, FaceNet.matching, v.v.; không dùng facenet_recognition.*.

## Caller cung cấp

- RGB uint8 (H,W,3), H/W > 0; chuyển OpenCV BGR→RGB một lần.
- person_id hợp lệ; N > 0 ảnh enrollment cùng người và có quyền sử dụng.
  Mỗi ảnh phải có đúng một mặt; backend không chọn mặt từ ảnh nhóm.
- registry_path tuyệt đối trong project hoặc vùng tạm, parent đã tồn tại.
- Models đã nạp và toàn bộ canonical JSON model_version ở trên.

Backend tự alignment 160×160, preprocessing và embedding float32 (512,) L2.
UI không tự normalize/alignment lần nữa; không log ảnh, vectors hoặc BLOB.

## Registry, enrollment và recognition

Chỉ khi người dùng chủ động tạo registry mới:

```python
initialize_enrollment_registry(registry_path)
```

Không initialize/overwrite/migrate registry đang có. Sau khi caller cung cấp
person_id, images_rgb và query_rgb:

```python
enrolled = enroll_person_from_images(
    person_id, images_rgb, detector, facenet_net, registry_path, model_version
)
results = recognize_image(
    query_rgb, detector, facenet_net, registry_path, model_version
)
rollback_results = recognize_image(
    query_rgb, detector, facenet_net, registry_path, model_version,
    threshold_profile="v1",
)
```

Enrollment trả status="enrolled", person_id, num_images, model_version và representative
embedding. Duplicate ID hoặc một ảnh lỗi bị từ chối; không upsert/ghi từng phần.
Recognition chỉ đọc SQLite. Giữ registry ổn định trong lúc đọc, dùng single writer.

Output recognition là list[dict] theo face order: face_index, bbox xyxy half-open,
confidence, status, person_id, best_match_id, similarity, threshold, model_version,
landmarks khi có. Không trả ảnh/embedding/ranking.

- recognized: dùng person_id đã chấp nhận.
- unknown: person_id=None; best_match_id là candidate, không là danh tính xác nhận.
- Registry rỗng hợp lệ: Unknown với candidate/score=None.
- Không có mặt: [] sau preflight input/version/DB hợp lệ.
- TypeError/ValueError: input/profile/metadata; FileNotFoundError: assets/registry thiếu.
  Lỗi SQLite/detection/alignment/inference truyền lên; UI phân biệt lỗi hạ tầng
  với Unknown, không hiển thị batch một phần.

similarity là cosine score, không phải xác suất; confidence là detection confidence.
Không dùng person_id="Unknown" hoặc tự đổi threshold khi nhận diện thất bại.

## Threshold và version

V2 mặc định: 0.665513450766159. V1 rollback: 0.925533688089963,
chỉ cho call có threshold_profile="v1". Giữ similarity >= threshold.
threshold=... override hỗ trợ số hữu hạn [-1,1]; không bool, None, chuỗi, NaN/Inf.
Profile vẫn phải hợp lệ "v1"/"v2" khi override.

model_version sau refactor khác version cũ do fingerprint source/path thay đổi.
Dùng chuỗi do build_model_version trả về; không thay nó bằng SHA descriptor,
hash ONNX hoặc ghi đè metadata trong SQLite để bỏ qua kiểm tra.

## Registry thuộc frozen version cũ

Chỉ sau khi chủ sở hữu đồng ý, đóng registry nguồn và chọn một file đích mới:

```python
from integration.enrollment_pipeline import migrate_legacy_registry

migrated_path = migrate_legacy_registry(
    old_registry_path, new_registry_path, detector, model_version=model_version
)
```

new_registry_path phải tuyệt đối, chưa tồn tại, parent đã có, trong project hoặc vùng tạm.
Hàm chỉ chuyển đúng frozen profile cũ có descriptor SHA
26f1432e29721a4d1e881164a62db603a968a5aa3c0e505afcba29c59668d99b
khi source đã kiểm chứng, weights, runtime và cấu hình detector còn khớp.
Giữ nguyên nguồn; đích giữ schema, BLOB, checksum, ID và số ảnh, chỉ đổi version metadata.
Sau đó UI chủ động chọn migrated_path cho các lời gọi backend.

Không tự chạy migration khi mở UI. Unknown/mixed versions, changed runtime/source/weights,
checksum lỗi, nguồn rỗng hoặc nguồn có SQLite sidecars bị từ chối; cần kiểm tra hoặc
enrollment lại tùy trường hợp. Migration chỉ được kiểm chứng bằng SQLite tạm trong phase này.

.venv còn phụ thuộc Windows/CPython 3.13.14 hiện tại. FEI không đi kèm và kết quả tập nhỏ
không chứng minh production-grade accuracy; concurrent writers/thread safety chưa được bảo đảm.
