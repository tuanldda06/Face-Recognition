# TASK 01 — Báo cáo triển khai đăng ký webcam

Ngày: **2026-11-07, Asia/Saigon**. Trạng thái: **đã triển khai và kiểm thử tự động; đang chờ kết quả webcam thực tế, chưa đề xuất nghiệm thu**.

Luồng hiện tại yêu cầu **4 góc** theo thứ tự nhìn thẳng, quay trái, quay phải và ngẩng đầu. Phiên chuyển sang xem lại ở 4/4, kiểm tra độ tương đồng giữa 6 cặp ảnh và lưu số ảnh thực tế là 4. Các mục 5 góc bên dưới ghi lại các bản triển khai trước đó.

### Cập nhật sau lượt thử webcam

Lượt thử của người dùng hiển thị được hình camera nhưng dừng ở 0/5 và báo mất frame. Chrome với camera mô phỏng tái hiện được việc gửi frame đầu rồi dừng. Hai sự kiện stream nay dùng chế độ mặc định của Gradio cho phép gửi frame trong lúc phiên đang chạy. Bản cập nhật nhận 28 yêu cầu frame liên tiếp thành công trong lượt thử mô phỏng. Ô tiến độ nằm bên trái phía trên trong khung camera trên cả khung rộng và khung 390 px.

Giao diện hiện dùng đăng ký webcam 5 góc và API đăng ký công khai dùng luồng phiên tương ứng. Ngưỡng độ nét hiện là 15 và giới hạn yaw của góc nhìn thẳng là 16°. Vùng ảnh webcam do người dùng gửi đo được 28,75 điểm độ nét và được chấp nhận ở bước nhìn thẳng. **222 bài kiểm thử pass.** Cần thử trọn phiên trên camera vật lý để xác nhận luồng đăng ký và nhận diện sau bản cập nhật. Các mục dưới đây ghi lại bản triển khai ban đầu.

Ở lượt thử tiếp theo, phiên đạt 4/5 rồi báo `pose_unavailable` khi cúi đầu. Trên mẫu gửi kèm, bộ đo ước lượng pitch 15,64° với sai số RMS 9,84% khoảng cách hai mắt. Bản sửa cho phép riêng tư thế cúi có sai số RMS tối đa 11%, sai số từng điểm tối đa 14% và yaw tối đa 14°. Cả 12 phiên bản của mẫu, gồm ảnh gốc và các biến thể cắt, đệm, nén JPEG, đều được phân loại `down`. **114 kiểm thử tư thế và phiên webcam pass.** Bản sửa chạy ở cổng 8003 để giữ phiên đang mở ở cổng 8002.

## 1. Kiến trúc trước và sau

Trước nâng cấp:

```text
Gradio upload / POST /enroll
  -> ApplicationService.enroll + mutex/file lock
  -> enroll_person_from_images
  -> MTCNN.detect -> alignment 160x160 RGB -> adapter
  -> FaceNet (512 float32, L2) -> mean float64 + L2 -> SQLite INSERT

Ảnh / webcam nhận diện -> recognize_image -> integration reader N>0
  -> matching với threshold V2 gốc -> public response + vẽ kết quả
```

Sau nâng cấp, bổ sung nhánh sau; các nhánh cũ tiếp tục sử dụng pipeline hiện tại:

```text
ID + tên -> phiên RAM có token ngẫu nhiên, model_version, TTL và revision
Webcam không mirror -> resize tối đa 640 px cạnh dài -> MTCNN
  -> đúng một mặt -> kiểm tra biên/kích thước/sáng/nét
  -> landmarks ảnh gốc -> PnP yaw/pitch/roll -> tư thế đang thiếu
  -> ổn định nhiều frame -> lưu ảnh trong phiên -> tiến độ 0/5 ... 5/5
  -> xem lại 5 ảnh/revision -> xác nhận
  -> kiểm tra lại toàn bộ ảnh/tư thế
  -> enrollment pipeline GỐC trên SQLite staging tạm
  -> kiểm tra 10 cặp embedding từ đúng các forward của lượt pipeline đó
  -> BEGIN IMMEDIATE trên registry chính + ATTACH names
  -> kiểm tra toàn bộ bản ghi trong cùng snapshot, model_version và duplicate
  -> INSERT đại diện + INSERT tên -> COMMIT cả hai hoặc ROLLBACK cả hai
```

Không sửa schema core: `FaceNet.database.check_schema` cấm thêm cột/bảng vào registry. Tên được lưu trong `<registry>.names.sqlite`, bảng `EnrollmentNames(person_id,display_name)`. Registry cũ không cần migrate. `/persons` chỉ thêm trường `display_name` khi danh tính có tên; format upload enrollment và recognition cũ giữ nguyên. Xóa qua application xóa tên và danh tính trong cùng transaction.

Atomic commit qua hai database yêu cầu main trên đĩa, cả hai `journal_mode=DELETE`, `synchronous>=FULL`; code kiểm tra và từ chối cấu hình khác, không tự đổi journal mode. Điều kiện này dựa trên [SQLite ATTACH](https://www.sqlite.org/lang_attach.html) và [atomic commit/super-journal](https://www.sqlite.org/atomiccommit.html). Đã kiểm tra rollback bằng lỗi INSERT tên sau INSERT embedding; chưa mô phỏng mất điện/hỏng thiết bị lưu trữ.

Trước xác nhận không ghi danh tính vào database. Phiên giữ ảnh trong RAM, chỉ server chọn ảnh đã đạt điều kiện. Preview dùng JPEG data URI. Gradio có cache tạm với `delete_cache=(60,300)`; không lưu ảnh webcam vào registry. Staging chỉ được tạo trong bước xác nhận và được dọn bằng `TemporaryDirectory` khi thành công/lỗi. Lưu xong/hủy/xử lý hết hạn sẽ bỏ ảnh phiên; hết hạn được quét khi có thao tác session tiếp theo và qua lifecycle Gradio State.

## 2. Phân công và phạm vi rà soát

| Trách nhiệm | Đã thực hiện | Giới hạn trách nhiệm |
|---|---|---|
| Người 1 | Đọc MTCNN detector, loader, models, detection, preprocessing/alignment, tests và adapter; xác nhận thứ tự/tọa độ landmarks; bổ sung pose estimator và 76 test | Không đổi MTCNN, weights hay alignment; cần phối hợp hiệu chuẩn pose trên webcam |
| Người 2 | Đọc toàn bộ FaceNet modules/tests; integration enrollment/recognition/runtime/adapter/tests; bổ sung helper consistency và 73 test | Không đổi FaceNet/preprocessing/matching/threshold; ngưỡng consistency chưa có calibration FAR/FRR |
| Người 3 | Đọc toàn bộ application modules/tests, luồng từ UI/API đến SQLite; thiết kế phiên, UI/API, quality, lưu nguyên tử, hồi quy, báo cáo | Tích hợp giữ core nguyên byte; thử nghiệm webcam vật lý cần người vận hành phối hợp |

Các hàm tái sử dụng: `validate_detection_result`, `MTCNN.detect`, `extract_faces_rgb`, `mtcnn_to_facenet_detections`, `enroll_person_from_images`, `initialize_enrollment_registry`, `representative_from_embeddings` qua pipeline, `validate_database_embedding`, `cosine_similarity`, `validate_threshold`, `row_to_person`, `embedding_to_blob`, `recognize_image`, `ApplicationService.guard/_operation`, `validate_rgb/decode_image`.

Đối chiếu hash **41 file Python core đã có tại thời điểm snapshot: không có thay đổi**, xem [task01-core-integrity.json](reports/task01-core-integrity.json). Fingerprint runtime/model đo lại vẫn là `47be33e07cc5dad54d4848463758e3191ebda37696d84109066e6a71f330dc16`. Không thay weights, thuật toán MTCNN/FaceNet, schema, source fingerprinted hoặc threshold recognition V2 `0.665513450766159`. Workspace là bản không có `.git`; không có git diff/commit để cung cấp.

## 3. Danh sách file thay đổi

| File | Vai trò |
|---|---|
| `application/head_pose.py` (mới) | PnP 5 điểm, kiểm tra hình học/nghiệm, phân loại tư thế |
| `integration/webcam_identity.py` (mới) | Kiểm tra và chuẩn hóa độc lập đúng 5 embedding, đủ 10 cặp cosine |
| `application/webcam_enrollment.py` (mới) | Chính sách chất lượng, phiên RAM, ổn định, chụp, review/revision, retake/pause/resume/cancel/confirm, pipeline staging |
| `application/webcam_registry.py` (mới) | Metadata tên riêng, kiểm tra schema, commit/delete atomic, xử lý sidecar rỗng sau rollback lần đầu |
| `application/service.py` | Gắn manager, đọc tên khi liệt kê, xóa atomic tên/danh tính |
| `application/api.py` | Endpoints phiên, body limit cho uploads/JSON mới, review snapshot có revision |
| `application/ui.py` | Tab đăng ký camera, hướng dẫn/tiến độ, 5 preview, chụp lại/hủy/xem lại/xác nhận; tách queue stream khỏi controls, chống response cũ ghi đè kết quả |
| `application/tests/test_head_pose.py` (mới) | 76 ca hình chiếu tổng hợp/invalid data, dấu và ngưỡng góc |
| `integration/tests/test_webcam_identity.py` (mới) | 73 ca consistency, sai dữ liệu, mọi cặp, boundary, không sửa input |
| `application/tests/test_webcam_enrollment.py` (mới) | 33 ca phiên/API/UI/SQLite và các ca lỗi/hồi quy |
| `application/tests/test_real_five_image_persistence.py` (mới) | Model thật, staging/commit 5 ảnh mẫu lặp, nhận diện cả danh tính cũ/mới |
| `application/README.md` | Hướng dẫn tính năng và lưu/sao lưu metadata |
| `application/TASK01_AUDIT.md`, `TASK01_REPORT.md` (mới) | Audit, phân công, thiết kế và báo cáo này |
| `application/reports/task01-*` (mới) | XML/log tests, baseline hashes/so sánh lỗi cũ, static model diagnostic và verification summary |

`application/data/task01-demo.sqlite`, PID và thư mục tạm pytest là artifacts local, đã thuộc gitignore `application/data/`. Không dùng registry demo/vận hành cũ để chạy test hoặc ghi danh tính thử.

## 4. Thuật toán tư thế, chất lượng và quản lý phiên

Landmarks lấy từ `DetectionResult.landmarks[0]`, **trước alignment**, theo thứ tự mắt trái ảnh, mắt phải ảnh, mũi, miệng trái ảnh, miệng phải ảnh. Tọa độ là pixel của frame detector nhận, không phải ảnh aligned 160×160.

Fit năm điểm ảnh vào mô hình mặt 3D không đồng phẳng bằng `SOLVEPNP_SQPNP`, refine LM, rồi phân rã rotation thành yaw/pitch/roll theo `Rz @ Ry @ Rx`. Không dùng vector Rodrigues như Euler, không dùng detection confidence làm góc. SQPnP hỗ trợ số điểm này; DLT khởi tạo ITERATIVE không có guess cần tối thiểu sáu điểm. Xem [OpenCV PnP](https://docs.opencv.org/4.12.0/d5/d1f/calib3d_solvePnP.html).

Camera xấp xỉ `fx=fy=max(H,W)`, tâm `(W/2,H/2)`, distortion bằng 0. Loại landmarks ngoài ảnh, mắt quá gần, hình học suy biến, điểm sau camera, residual quá lớn hoặc nghiệm gần ngang nhau nhưng góc khác nhau nhiều. RMS tối đa 6%, residual lớn nhất tối đa 12% khoảng cách hai mắt. Không có fallback tự gán tư thế nhìn thẳng.

Ảnh không mirror: yaw dương = quay sang **phải của người dùng**, pitch dương = cúi, roll dương = nghiêng theo chiều kim đồng hồ trong ảnh. UI cấu hình `gr.WebcamOptions(mirror=False)`, phù hợp [Gradio Image](https://gradio.app/docs/image).

| Góc | Điều kiện độ |
|---|---|
| Nhìn thẳng | `abs(yaw)<=10`, `abs(pitch)<=10` |
| Trái | `-50<=yaw<=-18`, `abs(pitch)<=12` |
| Phải | `18<=yaw<=50`, `abs(pitch)<=12` |
| Ngẩng | `-35<=pitch<=-15`, `abs(yaw)<=12` |
| Cúi | `15<=pitch<=35`, `abs(yaw)<=12` |

Tất cả cần `abs(roll)<=15`. Có vùng chuyển tiếp không được chụp. Chỉ tư thế đang thiếu mới được chấp nhận; một tư thế không thể tăng hai ô tiến độ.

Quality policy ban đầu: đúng 1 mặt; bbox nằm trong ảnh; cạnh mặt tối thiểu 80 px sau resize; crop mặt chuyển gray/resize 160×160; mean sáng 45–220, variance Laplacian >=65. Cần ít nhất **4 frame khác nhau và 0,8 giây**, spread yaw/pitch/roll <=6°, gap <=2 giây. Frame lỗi/sai tư thế/mất camera/ảnh lặp reset chuỗi ổn định. Timer báo ngắt luồng khi không có frame. Chất lượng và góc chưa được calibration theo từng thiết bị.

Mỗi phiên có token ngẫu nhiên, ID/tên cố định, model_version, ảnh theo đủ 5 keys, revision, TTL 20 phút; giới hạn 8 phiên và resize cạnh dài <=640 để chặn tăng RAM. Client chỉ gửi frame và revision, không gửi ảnh “đã thu” hoặc tư thế tự khai để confirm. Retake chỉ xóa góc chọn, giữ bốn góc khác, tăng revision. Review trả ảnh và revision cùng snapshot; confirm cần revision mới nhất đã review và checkbox xác nhận. Stream lỗi từ phiên đã kết thúc không được ghi đè thông báo lưu/hủy.

Sau inference của pipeline staging, chuẩn hóa bản sao của đúng năm raw output và yêu cầu **cả 10 cặp cosine >=0.65**. Đây là ngưỡng consistency riêng, không đổi ngưỡng recognition. Nếu khác người hoặc không nhất quán, registry chính và tên chưa được ghi. Threshold này chỉ là chính sách ban đầu, không bảo đảm cùng danh tính.

## 5. Kết quả kiểm thử tự động và model thật

Lệnh release:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:GRADIO_ANALYTICS_ENABLED='False'
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -q -p no:cacheprovider -c application/pytest.ini application/tests integration/tests/test_webcam_identity.py --basetemp=application/data/task01-release-tests --tb=short --junitxml=application/reports/task01-release-tests.xml
```

**220 passed, 0 failed, 0 errors, 0 skipped, 83.04 giây**. Bằng chứng: [XML](reports/task01-release-tests.xml), [log](reports/task01-release-tests.log). Lệnh chứa `--basetemp` riêng vì thư mục mặc định `C:/Users/Admin/AppData/Local/Temp/pytest-of-Admin` bị Access denied. Chọn đường dẫn basetemp mới khi chạy lại.

| Trường hợp yêu cầu | Kết quả | Bằng chứng/phạm vi |
|---|---|---|
| Đủ 5 góc, review/confirm, tên, SQLite, nhận diện lại | PASS | Landmarks và net giả lập, pipeline/SQLite thật, API full cycle |
| Không mặt / nhiều mặt | PASS | Báo lỗi/reset ổn định; không ghi |
| Sai tư thế / chưa đủ frame / chưa đủ thời gian / jitter | PASS | Không tăng tiến độ, không dùng lặp góc |
| Mờ / thiếu sáng / dư sáng | PASS | Kiểm tra crop mặt, reset chuỗi |
| ID trùng lúc tạo và phát sinh trước confirm | PASS | 409, byte dữ liệu cũ không đổi |
| Hủy giữa chừng, cách ly hai phiên, TTL/capacity | PASS | Token cũ không thể frame/review/confirm; ảnh bị bỏ |
| Chụp lại một góc, frame/review revision cũ | PASS | Giữ góc khác, bắt review bộ mới |
| Lỗi detector/net/database/camera | PASS | Không ghi một phần; lỗi model không thành Unknown |
| Lỗi INSERT tên sau INSERT embedding | PASS | Rollback cả hai file, dữ liệu danh tính đã có nguyên byte |
| Lỗi commit lần đầu tạo sidecar rỗng | PASS | List/delete vẫn dùng được, có thể lưu lại sau đó |
| WAL, model_version sai, ID Unknown trong registry | PASS | Từ chối ghi; không tự migrate/sửa cấu hình |
| Hai confirm đồng thời | PASS | Một success, một session_missing; chỉ một bản ghi |
| UI stream và controls, response cũ sau cancel | PASS | Slot độc lập, đồng bộ revision và không ghi đè thông báo |
| Enrollment ảnh, recognition ảnh và webcam frame cũ | PASS | Toàn bộ suite application trước đây vẫn pass |
| Model thật và compatibility bản ghi 5 ảnh | PASS | MTCNN/FaceNet thật, pipeline staging/commit/readback, nhận diện old/new |
| Camera vật lý đăng ký 5 góc rồi nhận diện trực tiếp | **CHƯA XÁC NHẬN** | Không suy ra từ test tổng hợp/model smoke |

Model thật: `test_real_smoke.py` kiểm tra upload enrollment/readback, single/group/no face, known/unknown, HTTP, webcam frame processing và đối chiếu direct core. Test 5 ảnh mới lặp **cùng một fixture tĩnh năm lần** để kiểm tra representative, consistency và lưu/đọc count 5; không gọi đây là thử 5 tư thế. Test xác minh embedding/count/version của danh tính cũ không đổi, rồi nhận diện độc lập bản ghi cũ và mới.

Hồi quy toàn bộ core:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -q -p no:cacheprovider -c application/pytest.ini FaceNet/tests integration/tests MTCNN/tests --basetemp=application/data/task01-legacy-final --tb=line --junitxml=application/reports/task01-legacy.xml
```

Raw pytest: **249 passed, 56 failed, 55 errors, 1 skipped; 59 subtests passed**, 58.96 giây. [XML](reports/task01-legacy.xml), [log](reports/task01-legacy.log). **Không phải toàn bộ suite pass.** Các lỗi còn do nguồn notebook/IMM/baseline `D:/Facenet-Recognition`, ảnh local `D:/MTCNN` thiếu và migration gate từ chối runtime khác frozen profile. 73 helper tests mới nằm trong 249 pass; phần legacy có 176 pass như báo cáo nền sau restore model.

So sánh với `legacy-after-model.xml` đã có: cùng tập **100 failure/error testcase IDs trong XML**, không thêm hoặc hết ID lỗi nào; raw pytest tính thêm subtest failures nên không cộng trực tiếp số XML vào tổng raw. Xem [so sánh](reports/task01-legacy-comparison.json). Không sửa test cũ/nguồn frozen để làm xanh suite.

## 6. Webcam thực tế và lỗi/giới hạn còn tồn tại

Server demo riêng: [http://127.0.0.1:8002/ui/](http://127.0.0.1:8002/ui/), registry `application/data/task01-demo.sqlite`. `/health`, `/ui/`, `/ui/config` trả 200, model sẵn sàng. HTTP/UI config không chứng minh camera hoạt động.

Đã dùng skill computer-use để tìm trình duyệt và thử mở UI. Browser connector Chrome không có; native computer-use bị dừng vì không thể xác định URL browser trên Windows đủ tin cậy để áp dụng policy. Đã dừng thao tác UI, không vượt chặn. Người vận hành trả lời **đang thử webcam và sẽ báo kết quả**; chưa có kết quả đăng ký/nhận diện xác nhận để ghi PASS. Server được nạp lại bản cuối, phiên RAM dở trước restart cần tạo lại.

Chẩn đoán độc lập bằng model thật trên fixture crop 160×160: detector thấy một mặt, score 0.999900; bbox y2=162.86 vượt ảnh nên quality gate trả `face_clipped`. Nếu đo riêng landmarks, pose estimator trả `None`: RMS 8.42%, max residual 12.44% khoảng cách hai mắt, vượt giới hạn 6%/12%. Xem [diagnostic JSON](reports/task01-static-diagnostic.json). Đây là dữ liệu ảnh tĩnh, không phải webcam hoặc ground truth pose; cho thấy mô hình 3D chung có thể từ chối ảnh thực và cần đánh giá tiếp.

Các phần còn chưa giải quyết/kiểm chứng:

1. **Webcam E2E chưa có bằng chứng**: phải thử thật capture/review/save rồi nhận diện bằng frame mới, đồng thời kiểm tra chiều trái/phải và ngẩng/cúi. Không đề nghị nghiệm thu khi mục này chưa đạt.
2. **Pose chưa hiệu chuẩn**: chỉ 5 landmarks, không có cằm/contour, intrinsics/distortion xấp xỉ. Khác tỷ lệ mặt, che khuất, kính, góc lớn hoặc camera góc rộng có thể lệch pitch/từ chối ảnh đúng. Chưa có tập 5 góc thật với nhãn yaw/pitch để đo confusion/error.
3. **Quality/consistency chưa calibration**: ánh sáng trung bình có thể bỏ sót ngược sáng cục bộ; Laplacian nhạy noise; cùng người có thể fail consistency khi quay nhiều, khác người có thể pass. Không có bằng chứng FAR/FRR mới.
4. **Chưa có liveness**: không bảo đảm chống ảnh in/video replay; hash ảnh chỉ chặn dùng lại y nguyên một frame để tích lũy ổn định.
5. **Suite core chưa xanh** vì dữ liệu/môi trường frozen thiếu; kết quả không được gọi là full regression pass. Chưa kiểm tra power loss hoặc cài sạch máy khác.
6. **Phạm vi vận hành local một worker**: session RAM mất khi restart; chưa hỗ trợ sessions chia sẻ nhiều workers. Writers ngoài application không tuân file lock không nằm trong đảm bảo snapshot của recognition cũ. Cần giữ/copy cả main SQLite và sidecar tên.

## 7. Đề xuất và điều kiện nghiệm thu tiếp theo

- Người 1 phối hợp thu tập webcam đa dạng người/thiết bị, nhãn trái/phải/ngẩng/cúi thực, hiệu chuẩn camera; đánh giá residual và pitch bias trước khi chỉnh ngưỡng. Nếu 5 điểm không đủ, thống nhất interface landmark dày hơn ở adapter/application, không tự sửa MTCNN.
- Người 2 đánh giá phân bố cosine cùng/khác người trên đủ 5 góc, chọn consistency threshold theo FAR/FRR; kiểm tra chất lượng representative và recognition query độc lập sau enrollment.
- Người 3 chạy checklist thật bên dưới, lưu kết quả quan sát và lỗi, rồi mới cập nhật trạng thái nghiệm thu. Khôi phục nguồn dataset/runtime frozen để chạy lại các regression còn lỗi.

Checklist người vận hành:

1. Tải lại UI, nhập ID chưa có và tên, bắt đầu phiên, bật camera và nút ghi hình.
2. Giữ nhìn thẳng → trái của bạn → phải của bạn → ngẩng → cúi. Mỗi lần cần mặt đủ lớn, rõ, đủ sáng, tiến độ tăng đúng một ô. Thử đưa thêm mặt/che camera: không được chụp.
3. Xem đủ 5 ảnh, chụp lại riêng một góc, xác nhận chỉ góc đó bị thay; xem lại bộ mới rồi đánh dấu xác nhận và lưu.
4. Kiểm tra `/persons` có đúng ID/tên/count 5; thử ID trùng không overwrite. Thử hủy một phiên khác: không có record.
5. Chuyển tab nhận diện webcam, bật nhận diện/ghi hình; kiểm tra **frame mới** nhận đúng ID, stop/start và mất camera hoạt động. Nhận diện lại một danh tính cũ đã đăng ký để kiểm tra hồi quy thực.
6. Ghi rõ thiết bị/browser, thời gian, bước đạt/chưa đạt, thông báo và kết quả nhận diện; không đánh PASS chỉ từ UI 5/5.

**Kết luận hiện tại: bản triển khai có bằng chứng tự động và model smoke; nghiệm thu webcam thực tế còn mở.**
