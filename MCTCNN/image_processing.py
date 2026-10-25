import numpy as np
from PIL import Image , ImageOps

from bounding_box import valid_box_indices

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

def crop_and_resize(image , boxes , size):
    image = check_rgb(image)
    boxes = np.asarray(boxes , dtype = np.float32)
    if len(valid_box_indices(boxes)) != len(boxes) or size < 1:
        raise ValueError("Box hoặc kích thước crop không hợp lệ.")
    actual_boxes = np.concatenate([np.floor(boxes[: , :2]) , np.ceil(boxes[:,2:])] , axis = 1)
    crops = np.empty((len(boxes) , size , size , 3) , dtype = np.float32)
    source = Image.fromarray(image)
    for index , box in enumerate(actual_boxes) :
        crop = source.cop(tuple(int(value) for value in box))
        resized = np.asarray((crop.resize((size,size)) ,Image.Resampling.BILINEAR) ,   dtype = np.unit8)
        crops[index] = normalize_pixels[resized]
    return crops , actual_boxes.astype(np.float32)   