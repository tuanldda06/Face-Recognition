# TASK 01 — Audit và kế hoạch trước triển khai (2026-11-07)

Luồng hiện tại: Gradio/FastAPI -> ApplicationService.enroll -> integration.enrollment_pipeline.enroll_person_from_images -> MTCNN.detect -> extract_faces_rgb align=True 160x160 -> mtcnn_to_facenet_detections -> FaceNet.extract_embedding -> representative_from_embeddings -> database.add_person. Recognition dùng integration.recognition_runtime, chấp nhận N>0 (reader nghiên cứu FaceNet.database riêng yêu cầu 3 ảnh, không dùng cho application).

Đã đọc application: service, api, ui, webcam, config, images, errors, assets, visualization, benchmark, CLI, tests, README/AUDIT/HANDOFF. Người 1 rà soát MTCNN models/detection/preprocessing/loader/detector/tests; Người 2 rà soát toàn bộ FaceNet và integration cùng tests. Kết quả chi tiết bổ sung trong TASK01_REPORT.md.

Hợp đồng: ảnh RGB uint8 HWC; DetectionResult boxes Nx4 xyxy half-open, landmarks Nx5x2 theo thứ tự mắt trái ảnh, mắt phải ảnh, mũi, khóe miệng trái ảnh, phải ảnh. Embedding float32 (512,), L2 float64 atol 1e-6. model_version canonical hash cả core source; không sửa các file đó hoặc migration/schema cũ.

Thay đổi dự kiến và trách nhiệm:
- Người 1: application/head_pose.py + tests; ước lượng góc từ landmarks hình học 3D/PnP, không dùng detection confidence làm góc; không sửa MTCNN.
- Người 2: integration/webcam_identity.py + tests; kiểm tra đủ 5 embedding, hữu hạn/norm và cosine pairwise; không sửa FaceNet.
- Người 3: application/webcam_enrollment.py (server-owned session/token, TTL, ổn định nhiều frame, quality, retake/cancel/review/confirm); application/webcam_registry.py (metadata tên riêng + atomic commit với schema core giữ nguyên); service/api/ui; tests và báo cáo.

Thiết kế lưu: chưa confirm không ghi database. Confirm chạy lại kiểm tra ảnh và gọi enrollment pipeline hiện có trên SQLite staging tạm, kiểm tra embedding consistency từ đúng lượt inference đó. Sau đó một transaction BEGIN IMMEDIATE trên registry đích, kiểm tra snapshot toàn bộ records/model version/duplicate rồi INSERT đại diện và tên trong database metadata attach (rollback journal DELETE cho atomic cross-file). Không add_person qua connection khác trong transaction cuối. Metadata tên riêng vì core check_schema chỉ chấp nhận đúng bảng FaceDatabase, thêm cột/bảng sẽ phá reader và fingerprint. Không tự migrate hoặc chỉnh danh tính cũ.

Giới hạn cần báo thật: PnP 5 điểm + camera intrinsics xấp xỉ chưa được calibration ngoài đời; cosine gate chưa được đánh giá FAR/FRR. Test synthetic không thay thử webcam thật. Chỉ đề nghị nghiệm thu sau capture 5 góc và recognition webcam thực, không dựa vào UI 5 bước.
