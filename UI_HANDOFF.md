# UI handoff — integration contract

Đọc [README.md](README.md) cho setup/models/tests. Đây là contract cho lớp UI,
không phải GUI hoặc server đã triển khai.

- Recognition: `from integration.recognition_runtime import recognize_image`.
- Enrollment: `initialize_enrollment_registry`, `enroll_person_from_images` từ
  `integration.enrollment_pipeline`;version từ `build_model_version(detector)`.
- Nạp một lần `MTCNN(device="cpu")` và `load_facenet_model(root/"models/facenet.onnx")`.
  Dùng đúng bundled assets,giữ objects sống giữa các requests;serialize model access.
- UI gửi RGB uint8(H,W,3) gốc.Chuyển OpenCV BGR→RGB một lần;không tự alignment/normalize.
- Core aligned face160×160 vàembedding512D/L2 do backend xử lý.
- Caller chọn registry_path tuyệt đối.Tạo parent directory,tạo registry mới chỉ khi
  người dùng chủ động yêu cầu;file có sẵn không được initialize/overwrite/migrate.
- Enrollment nhận person_id hợp lệ,N>0 RGB ảnh cùng người do caller xác nhận,
  đúng một face mỗi ảnh;duplicate/lỗi bị từ chối,không ghi từng phần.
- Recognition đọc registry tương thích canonical model_version;không tự ghi hoặc enrollment.
- Không log vectors/BLOB/ảnh;enrollment result có representative embedding.
- V2 mặc định;`threshold_profile="v1"` rollback một call;override số hữu hạn[-1,1]
  được hỗ trợ,không bool/None/NaN/Inf.Profile luôn phải hợp lệ.
- Unknown là status="unknown",person_id=None.Candidate best_match_id không là accepted ID.
- No-face trả[] sau preflight;DB/model/input lỗi là exception,không phải Unknown.
- Output mỗi face:face_index,bbox,confidence,status,person_id,best_match_id,
  similarity,threshold,model_version,landmarks khi có.Giữ thứ tự;similarity không là probability.
- Giữ registry ổn định khi đọc;single writer.Thread-safe/concurrent writer chưa được chứng minh.

Ví dụ gọi ở lớp ứng dụng sau bootstrap vàregistry/enrollment đã hợp lệ:
~~~python
from integration.recognition_runtime import recognize_image
from integration.enrollment_pipeline import enroll_person_from_images

# Các biến do caller cung cấp;ảnh/ID phải có quyền sử dụng vàcùng danh tính.
enrolled = enroll_person_from_images(
    person_id, images_rgb, detector, facenet_net, registry_path, model_version
)
faces = recognize_image(
    image_rgb, detector, facenet_net, registry_path, model_version
)
for face in faces:
    accepted_id = face["person_id"] if face["status"] == "recognized" else None
    # UI tự hiển thị bbox vàaccepted_id/Unknown;không dùng candidate thay accepted ID.
~~~

TypeError/ValueError:input/profile/metadata;FileNotFoundError:asset hoặcregistry thiếu.
SQLite/detector/alignment/FaceNet lỗi truyền lên;UI xử lý riêng infrastructure errors.
Không retry enrollment như upsert hoặc tự thay threshold để đạt recognition.

Handoff chứa .venv đã sao chép;vẫn cần Windows/CPython3.13.14 hệ thống tương thích.
Một số regression tests còn cần hai source projects;FEI không được phân phối kèm.
Đã kiểm thử trong phạm vi ghi tại [FINAL_HANDOFF_REPORT.md](FINAL_HANDOFF_REPORT.md);
không tuyên bố portable hoặc production-grade accuracy.
