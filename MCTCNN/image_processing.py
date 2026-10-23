import numpy as np
from PIL import Image , ImageOps

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