# MTCNN Face Detection (PyTorch + NumPy + Pillow)

Triển khai 3 mạng **P-Net, R-Net, O-Net** của MTCNN, dùng bộ pretrained weights có sẵn để **phát hiện khuôn mặt**. Các tác vụ hình học và xử lý ảnh dùng NumPy/Pillow, CNN inference dùng PyTorch. Module **chưa nhận diện danh tính**; nếu muốn Face Recognition, cần nối đầu ra ảnh mặt với FaceNet và cơ sở dữ liệu embeddings.

## 1. Cấu trúc và chức năng từng file

```text
Face-Recognition-main/
├── README.md                     # Tài liệu này
├── requirements.txt              # Thư viện chạy inference
├── requirements-dev.txt          # Thư viện chạy tests
├── pytest.ini                    # Cấu hình pytest
├── tests/                        # Unit tests + smoke test MTCNN
└── MTCNN/
    ├── __init__.py               # Đánh dấu Python package
    ├── demo.py                   # Entry point chạy inference và vẽ kết quả
    ├── face.jpg                  # Ảnh demo đi kèm
    ├── mtcnn.py                  # Ghép các stage P-Net -> R-Net -> O-Net
    ├── model_loader.py           # Khởi tạo 3 CNN, load pretrained weights
    ├── models/
    │   ├── pnet.py               # Proposal Network (12x12 receptive field)
    │   ├── rnet.py               # Refine Network (crop 24x24)
    │   └── onet.py               # Output Network (crop 48x48, 5 landmarks)
    ├── detection/
    │   ├── image_pyramid.py      # Resize ảnh theo nhiều scale cho P-Net
    │   └── bounding_box.py       # Generate boxes, NMS, regression, landmarks
    ├── preprocessing/
    │   ├── image_processing.py   # Đọc RGB, chuẩn hóa, crop và resize
    │   └── face_alignment.py     # Căn chỉnh 5 landmarks (tùy chọn)
    └── weights/
        ├── pnet.pt                # Checkpoint P-Net
        ├── rnet.pt                # Checkpoint R-Net
        └── onet.pt                # Checkpoint O-Net
```

### Các hàm quan trọng

| File | Hàm / Class | Ý nghĩa |
|---|---|---|
| `models/pnet.py` | `PNet` | Phát hiện vùng mặt tiềm năng trên từng scale |
| `models/rnet.py` | `RNet` | Sàng lọc và hiệu chỉnh các proposals từ P-Net |
| `models/onet.py` | `ONet` | Sàng lọc lần cuối và dự đoán 5 facial landmarks |
| `detection/image_pyramid.py` | `build_image_pyramid()` | `yield (scale, scaled_image)` cho P-Net |
| `detection/bounding_box.py` | `generate_bounding_box()` | Biến prediction map thành boxes, scores, offsets |
| `detection/bounding_box.py` | `nms()` | Loại các boxes chồng lấp, hỗ trợ `union` và `min` |
| `detection/bounding_box.py` | `calibrate_boxes()` | Áp dụng 4 bbox offsets để hiệu chỉnh tọa độ |
| `detection/bounding_box.py` | `square_boxes()` | Mở rộng box thành hình vuông, giữ nguyên tâm |
| `detection/bounding_box.py` | `decode_landmarks()` | Biến 10 giá trị O-Net thành `(N, 5, 2)` |
| `preprocessing/image_processing.py` | `read_image()` | Đọc ảnh, áp dụng EXIF orientation, chuyển RGB uint8 |
| `preprocessing/image_processing.py` | `normalize_pixels()` | `(pixel - 127.5) / 128.0` |
| `preprocessing/image_processing.py` | `crop_and_resize()` | Cắt khuôn mặt, resize 24/48/160, chuẩn hóa |
| `preprocessing/image_processing.py` | `crop_and_resize_rgb()` | Cắt khuôn mặt thành RGB uint8, không normalize |
| `preprocessing/face_alignment.py` | `align_face()` | Căn chỉnh khuôn mặt dựa trên 5 landmarks |
| `mtcnn.py` | `MTCNN.extract_faces_rgb()` | Xuất khuôn mặt RGB uint8 cho FaceNet |
| `model_loader.py` | `load_pretrained_models()` | Load `pnet.pt`, `rnet.pt`, `onet.pt` |
| `mtcnn.py` | `MTCNN.detect()` | Chạy detection, trả boxes, scores, landmarks |
| `mtcnn.py` | `MTCNN.extract_faces()` | Trả tensor khuôn mặt `(N, 3, 160, 160)` |
| `demo.py` | `main()` | Demo ảnh mẫu hoặc ảnh do người dùng chỉ định |

## 2. Cài đặt

Tất cả lệnh phía dưới **chạy từ thư mục gốc** `Face-Recognition-main/`, tức thư mục chứa `requirements.txt`.

```bash
cd Face-Recognition-main
python -m pip install -r requirements.txt
```

Nếu muốn dùng GPU NVIDIA, hãy cài **PyTorch có CUDA** phù hợp theo https://pytorch.org/get-started/locally/ rồi kiểm tra:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Các weights `.pt` của cả 3 mạng **đã nằm trong `MTCNN/weights/`**, không cần tải lại.

## 3. Chạy demo

```bash
# Xem cách dùng
python -m MTCNN.demo --help

# Chạy trên MTCNN/face.jpg (được chọn mặc định)
python -m MTCNN.demo

# Chạy ảnh mẫu, xuất ảnh đánh dấu bounding box và 5 landmarks
python -m MTCNN.demo --output detected.jpg

# Chạy ảnh khác
python -m MTCNN.demo /path/to/other.jpg --output detected.jpg

# Chạy GPU (nếu CUDA khả dụng)
python -m MTCNN.demo --device cuda --output detected.jpg
```

**Quan trọng:** Vì `demo.py` nằm trong package `MTCNN/` và các module đang dùng absolute imports `from MTCNN...`, **hãy chạy `python -m MTCNN.demo` tại thư mục gốc**, không chạy `cd MTCNN && python demo.py` (sẽ dễ gặp `ModuleNotFoundError: No module named 'MTCNN'`).

Nếu thành công, terminal sẽ in số boxes sau từng stage và kết quả:

```text
P-Net: ... boxes
R-Net: ... boxes
O-Net: ... boxes
Boxes: [[x1 y1 x2 y2] ...]
Scores: [0.99 ...]
Landmarks: [[[x y] ...] ...]
FaceNet input RGB: (N, 160, 160, 3) uint8
```

Nếu không có mặt, `N = 0`; `--output` vẫn lưu ảnh gốc, không có box/landmarks.

## 4. Pipeline inference

```text
Ảnh RGB uint8 (H, W, 3)
  -> Image Pyramid -> P-Net -> generate_bounding_box -> NMS (union)
  -> BBox Regression -> Square Boxes
  -> Crop 24x24 -> R-Net -> NMS (union) -> BBox Regression -> Square Boxes
  -> Crop 48x48 -> O-Net -> Decode 5 Landmarks -> BBox Regression -> NMS (min)
  -> DetectionResult: boxes (N,4), scores (N,), landmarks (N,5,2)
  -> Square Crop hoặc 5-point Alignment (tuỳ chọn)
  -> Ảnh RGB uint8 (N,160,160,3) hoặc tensor chuẩn hóa (N,3,160,160)
```

**Lưu ý:** `extract_faces()` mặc định crop theo box vuông, có thể bật `align=True` để sử dụng 5 landmarks. Dữ liệu ảnh đầu vào phải là RGB `uint8`; không đưa ảnh đã normalize vào `detect()`. Tensor trả về đã chuẩn hóa theo `(pixel - 127.5)/128` nên không chuẩn hóa lại lần nữa. Nếu FaceNet tự normalize, dùng `extract_faces_rgb()` để nhận RGB uint8.

## 5. Sử dụng MTCNN từ code Python

```python
from MTCNN.mtcnn import MTCNN

mtcnn = MTCNN(device="cpu")
result = mtcnn.detect("MTCNN/face.jpg")
print(result.boxes.shape)       # (N, 4)
print(result.scores.shape)      # (N,)
print(result.landmarks.shape)   # (N, 5, 2)

faces = mtcnn.extract_faces("MTCNN/face.jpg", result, size=160)
print(faces.shape)              # torch.Size([N, 3, 160, 160])
```

## 6. Lỗi thường gặp

- `ModuleNotFoundError: No module named 'MTCNN'`: Đứng tại thư mục gốc và chạy `python -m MTCNN.demo`.
- `FileNotFoundError: weights`: Kiểm tra đủ `MTCNN/weights/{pnet,rnet,onet}.pt`.
- `Boxes: []`: Kiểm tra xem P-Net có tạo boxes không và ảnh đầu vào có khuôn mặt đủ rõ không. Không sửa threshold trước khi xác nhận CNN outputs đúng thứ tự.
- `CUDA unavailable`: Dùng `--device cpu` hoặc cài đúng bản PyTorch CUDA.
- Kết quả `extract_faces` có shape `(0,3,160,160)`: Không tìm thấy mặt; không phải lỗi tensor.

## 6. Face Alignment và interface bàn giao cho FaceNet

MTCNN hiện hỗ trợ **hai cách tiền xử lý ảnh mặt** mà không cần thay đổi P-Net, R-Net hoặc O-Net:

- **`align=False` (mặc định)**: mở rộng bounding box thành hình vuông quanh tâm, crop và resize. Không xoay ảnh.
- **`align=True`**: dùng 5 landmarks của O-Net (mắt trái, mắt phải, mũi, khóe miệng trái, khóe miệng phải) để ước lượng phép biến đổi similarity (xoay, phóng to/thu nhỏ, tịnh tiến), sau đó tạo ảnh mặt bằng `cv2.warpAffine()`.

### File và thư viện mới

- `MTCNN/preprocessing/face_alignment.py`: `similarity_transform()` tính ma trận 2×3, `align_face()` xuất một ảnh RGB `uint8`, mặc định 160×160.
- `MTCNN/preprocessing/image_processing.py`: `crop_and_resize_rgb()` crop ảnh không normalize; `crop_and_resize()` giữ nguyên hợp đồng cũ cho R-Net/O-Net.
- `MTCNN/mtcnn.py`: `extract_faces_rgb(..., align=True)` cho người làm FaceNet, `extract_faces(..., align=True)` cho đầu ra tensor đã normalize.
- Thêm dependency `opencv-python-headless` để chạy alignment trên môi trường không có desktop GUI.

**Lưu ý:** Template 5 điểm sử dụng **ArcFace 112×112** và được scale theo kích thước đầu ra. Đây **không phải** pipeline crop mặc định của `facenet-pytorch`. Cần so sánh kết quả trên tập validation trước khi quyết định dùng alignment cho FaceNet và điều chỉnh threshold nhận diện nếu cần. Enrollment và recognition phải dùng cùng cách preprocessing.

### Chạy demo và xuất các khuôn mặt đã căn chỉnh

Tại thư mục gốc dự án:

```bash
# Dùng ảnh mẫu MTCNN/face.jpg
python -m MTCNN.demo --align --output detected.jpg --faces-dir aligned_faces

# Dùng ảnh riêng trong MTCNN/
python -m MTCNN.demo MTCNN/anhcho.jpg --align --faces-dir aligned_faces

# Crop thông thường, không alignment
python -m MTCNN.demo --faces-dir cropped_faces

# Chạy tests
python -m pytest -q tests/test_face_alignment.py
```

Trong `aligned_faces/face_1.png` là ảnh RGB 160×160 có pixel `0..255`, chưa normalize, sẵn sàng cho bước preprocessing của FaceNet.

### Giao diện Python cho người làm FaceNet

```python
from MTCNN.mtcnn import MTCNN

mtcnn = MTCNN(device="cpu")
result = mtcnn.detect("MTCNN/face.jpg")

# Input FaceNet: ảnh chưa normalize, (N, 160, 160, 3), dtype uint8
faces_rgb = mtcnn.extract_faces_rgb(
    "MTCNN/face.jpg", result, size=160, align=True
)

# Metadata khớp thứ tự với faces_rgb
boxes = result.boxes             # (N, 4)
scores = result.scores           # (N,)
landmarks = result.landmarks     # (N, 5, 2)

# Nếu FaceNet cần tensor đã normalize thay vì uint8:
faces_tensor = mtcnn.extract_faces(
    "MTCNN/face.jpg", result, size=160, align=True
)  # (N, 3, 160, 160), float32, normalized
```

**Không chuẩn hóa hai lần.** Nếu FaceNet của người 2 tự normalize, hãy sử dụng `extract_faces_rgb()` thay vì tensor trả về bởi `extract_faces()`.

Phần FaceNet/database/enrollment vẫn nằm ngoài package MTCNN. Ảnh mặt được xuất chỉ là dữ liệu đầu vào; chưa thể kết luận người đó đã đăng ký hay chưa.


## 7. Kiểm thử và validation trước khi bàn giao FaceNet

Cài các dependency dành cho kiểm thử:

```bash
python -m pip install -r requirements-dev.txt

# Chạy tất cả tests, gồm CPU smoke test dùng ảnh mặt thật và weights có sẵn
python -m pytest -q

# Chỉ chạy unit tests, bỏ qua inference pretrained
python -m pytest -q -m "not integration"
```

**Các trường hợp được kiểm tra:**

- Landmarks thiếu điểm, NaN/Inf, hai mắt trùng nhau, năm điểm thẳng hàng
- Bounding boxes không hợp lệ, hoàn toàn ngoài ảnh, hoặc chỉ nằm một phần ngoài ảnh
- Metadata `boxes (N,4)`, `scores (N,)`, `landmarks (N,5,2)` đồng bộ, confidence nằm trong `[0,1]`
- NMS `union`/`min`, hiệu chỉnh bbox và thứ tự 10 tọa độ landmarks từ O-Net
- Đầu ra `extract_faces_rgb` giữ RGB `uint8` trong `[0,255]`, batch `(N,160,160,3)` và thứ tự khuôn mặt
- Trường hợp không có khuôn mặt; output tensor `float32` và batch rỗng
- Smoke test tích hợp ba pretrained networks trên `MTCNN/face.jpg`

Validation được dùng trong `extract_faces_rgb()` để **báo lỗi rõ ràng** khi metadata không hợp lệ hoặc crop có dưới 25% diện tích nằm trong ảnh. Trong `align_face()`, ảnh căn chỉnh có dưới 25% vùng pixel lấy từ ảnh gốc cũng bị từ chối. Các ngưỡng là **chính sách chất lượng sơ bộ**, không phải ngưỡng được tối ưu từ dữ liệu nhận diện. Không âm thầm loại detection rồi làm lệch thứ tự boxes/landmarks/faces.

**Quy ước:** Input image là RGB `uint8`; bbox `[x1,y1,x2,y2]` theo pixel của ảnh gốc; landmarks theo thứ tự *mắt trái, mắt phải, mũi, khóe miệng trái, khóe miệng phải* như định nghĩa template (nhìn theo phía trái/phải của ảnh frontal), không suy ra nhãn trái/phải giải phẫu từ camera mirror. O-Net raw landmarks dùng `[x1..x5,y1..y5]`; `decode_landmarks()` biến đổi thành `(N,5,2)` theo tọa độ gốc. Cần đảm bảo cùng chuẩn alignment/preprocessing ở enrollment và recognition; template ArcFace có thể chưa tối ưu với FaceNet ONNX cụ thể.

**Giới hạn:** Các tests này không chứng minh độ chính xác recognition hay threshold Known/Unknown. Điều đó thuộc đánh giá end-to-end cùng FaceNet và database.
