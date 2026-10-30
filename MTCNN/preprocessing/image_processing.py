import numpy as np
from PIL import Image , ImageOps


#Đọc file
def read_image(path):
    with Image.open(path) as image :
        return np.array(ImageOps.exif_tranpose(image).convert('RGB', dtype = np.unit8))


def check_rgb(image):
    image = np.asarray(image)
    if image.ndim !=3 or image.shape[2] != 3 or min(image.shape[:2]) < 1 :
        raise ValueError("Cần ảnh RGB shape (H,W,3), H/W >0.")
    if image.dtype != np.uint8 :
        raise ValueError("Ảnh gốc phải là uint8 0..255; không truyền ảnh đã chuẩn hóa vào pipeline.")
    return image

def normalize_pixels(image):
    return (check_rgb(image).astype(np.float32)-127.5) / 128.0

def crop_and_resize(image, boxes, size):
    """Crop bounding boxes and resize for R-Net/O-Net."""
    image = check_rgb(image)
    boxes = np.asarray(boxes, dtype=np.float32)
    if not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size phải là số nguyên dương.")
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes phải có shape (N, 4).")
    if not np.isfinite(boxes).all():
        raise ValueError("boxes chứa NaN hoặc Inf.")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Bounding box không hợp lệ.")

    # Chuyển tọa độ float sang pixel coordinates
    actual_boxes = np.concatenate([np.floor(boxes[:, :2]),np.ceil(boxes[:, 2:])],axis=1).astype(np.int32)

    crops = np.empty((len(boxes), size, size, 3),dtype=np.float32)

    source = Image.fromarray(image)

    for i, box in enumerate(actual_boxes):
        crop = source.crop(tuple(map(int, box)))

        resized = crop.resize((size, size),Image.Resampling.BILINEAR)

        crops[i] = normalize_pixels(np.asarray(resized, dtype=np.uint8))

    return crops, actual_boxes.astype(np.float32)