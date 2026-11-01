# MTCNN tests (chỉ kiểm thử module MTCNN)

Thư mục `MTCNN/tests/` chứa unit tests và một smoke test sử dụng checkpoint thật.
Các test này **không kiểm thử FaceNet, camera/webcam, SQLite hay kết nối end-to-end**.
Các module khác có thể có test riêng mà không cần chuyển vào MTCNN.

## Chạy tại thư mục gốc dự án (`Face-Recognition/`)

```bash
# Cài dependencies MTCNN như bình thường (nếu chưa cài):
# python -m pip install -r requirements.txt
python -m pip install -r MTCNN/requirements-test.txt

# 30 tests của riêng MTCNN, gồm cả pretrained inference
python -m pytest -q MTCNN/tests

# Bỏ qua smoke test thực tế
python -m pytest -q MTCNN/tests -m "not integration"

# Chạy demo như trước, độc lập với pytest
python -m MTCNN.demo --align --faces-dir aligned_faces --output detected.jpg
```

## Test nào nằm ở đâu?

- `test_face_alignment.py`: shape, dtype, SVD alignment, crop.
- `test_validation.py`: landmarks lỗi, metadata sai, bounding box vượt biên, interface RGB uint8.
- `test_bbox_contract.py`: NMS, calibration, landmark coordinate mapping.
- `test_integration_smoke.py`: inference thực tế với MTCNN weights + ảnh `MTCNN/face.jpg`.

## Khi tích hợp vào repo nhiều module

```text
Face-Recognition/
├── MTCNN/
│   ├── mtcnn.py
│   ├── demo.py
│   ├── tests/                  # riêng MTCNN
│   └── TESTING.md
├── FaceNet/                   # giữ nguyên code và tests riêng nếu có
├── camera/                    # giữ nguyên code và tests riêng nếu có
└── tests/                     # tùy chọn: integration / end-to-end toàn hệ thống
```

Lưu ý: nếu repo cũ có `pytest.ini` với `testpaths = tests`, dùng lệnh
`python -m pytest -q MTCNN/tests` để chọn chính xác MTCNN tests mà không ảnh hưởng
các test của module khác. Không cần thay thế `pytest.ini` cấp repo.

Nếu MTCNN phát hiện dữ liệu bất thường, validation có thể `raise ValueError`;
phía adapter nên quyết định bỏ qua detection hay báo lỗi tùy tình huống.
