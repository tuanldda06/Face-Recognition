# MTCNN Face Detection (PyTorch + NumPy + Pillow)

Project này tự triển khai 3 mạng **P-Net, R-Net, O-Net** của MTCNN, dùng bộ pretrained weights có sẵn để **phát hiện khuôn mặt**. Các tác vụ hình học và xử lý ảnh dùng NumPy/Pillow, CNN inference dùng PyTorch. Module **chưa nhận diện danh tính**; nếu muốn Face Recognition, cần nối đầu ra ảnh mặt với FaceNet và cơ sở dữ liệu embeddings.

## 1. Cấu trúc và chức năng từng file

```text
Face-Recognition-main/
├── README.md                     # Tài liệu này
├── requirements.txt              # Thư viện cần cài
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
    │   └── image_processing.py   # Đọc RGB, chuẩn hóa, crop và resize
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
FaceNet input shape: (N, 3, 160, 160)
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
  -> Crop + Resize 160x160 -> FaceNet input tensor (N,3,160,160)
```

**Lưu ý:** `extract_faces()` chỉ crop/resize theo bounding box, **không sử dụng 5 landmarks để align**, đúng với quyết định đơn giản hóa project. Dữ liệu ảnh đầu vào phải là RGB `uint8`; không đưa ảnh đã normalize vào `detect()`. Tensor trả về đã chuẩn hóa theo `(pixel - 127.5)/128` nên không chuẩn hóa lại lần nữa. Nếu chuyển sang model FaceNet khác, cần xác nhận preprocessing/normalization mà model đó yêu cầu.

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
