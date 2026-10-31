import numpy as np
from PIL import Image , ImageOps


#Đọc file
def read_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image, dtype=np.uint8)


def check_rgb(image):
    image = np.asarray(image)
    if image.ndim !=3 or image.shape[2] != 3 or min(image.shape[:2]) < 1 :
        raise ValueError("Cần ảnh RGB shape (H,W,3), H/W >0.")
    if image.dtype != np.uint8 :
        raise ValueError("Ảnh gốc phải là uint8 0..255; không truyền ảnh đã chuẩn hóa vào pipeline.")
    return image

def normalize_pixels(image):
    return (check_rgb(image).astype(np.float32)-127.5) / 128.0

def crop_and_resize_rgb(image, boxes, size):
    """Crop + resize boxes; return RGB uint8 NHWC and the rounded crop boxes.

    PIL pads pixels outside the source image with black. Boxes are [x1,y1,x2,y2]
    in half-open pixel coordinates, and may extend beyond image boundaries.
    """
    image = check_rgb(image)
    boxes = np.asarray(boxes, dtype=np.float32)

    if not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size phải là số nguyên dương")
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes phải có shape (N, 4)")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Bounding box không hợp lệ")

    actual = np.concatenate([np.floor(boxes[:, :2]), np.ceil(boxes[:, 2:])], axis=1).astype(np.int32)
    crops = np.empty((len(actual), size, size, 3), dtype=np.uint8)
    source = Image.fromarray(image)

    for i, box in enumerate(actual):
        crop = source.crop(tuple(int(v) for v in box))
        crops[i] = np.asarray(crop.resize((size, size), Image.Resampling.BILINEAR),dtype=np.uint8,)

    return crops, actual.astype(np.float32)


def crop_and_resize(image, boxes, size):
    """Crop + resize for R-Net/O-Net; return float32 normalized NHWC."""
    crops, actual = crop_and_resize_rgb(image, boxes, size)
    return (crops.astype(np.float32) - 127.5) / 128.0, actual
