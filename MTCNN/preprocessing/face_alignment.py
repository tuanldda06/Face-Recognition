import cv2
import numpy as np

from MTCNN.preprocessing.image_processing import check_rgb


# 5 điểm chuẩn: mắt trái, mắt phải, mũi, miệng trái, miệng phải
TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def similarity_transform(source, target):
    """Tính ma trận căn chỉnh từ 5 landmarks."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if source.shape != (5, 2) or target.shape != (5, 2):
        raise ValueError("Landmarks phải có shape (5, 2)")

    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Landmarks chứa NaN/Inf")

    # Đưa các điểm về tâm
    src_center = source.mean(axis=0)
    dst_center = target.mean(axis=0)

    src = source - src_center
    dst = target - dst_center

    energy = np.sum(src ** 2)
    if energy < 1e-8:
        raise ValueError("Landmarks không hợp lệ")

    # Tính phép xoay và scale
    u, s, vt = np.linalg.svd(src.T @ dst)
    d = np.diag([1.0, np.linalg.det(u @ vt)])
    rotation = u @ d @ vt
    scale = np.sum(s * np.diag(d)) / energy

    # Tính phép dịch chuyển
    translation = dst_center - scale * (src_center @ rotation)

    return np.column_stack([
        (scale * rotation).T,
        translation
    ]).astype(np.float32)


def align_face(image, landmarks, size=160):
    """Căn chỉnh khuôn mặt, trả về RGB uint8."""
    image = check_rgb(image)

    if not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size phải là số nguyên dương")

    target = TEMPLATE * (size / 112.0)
    matrix = similarity_transform(landmarks, target)

    return cv2.warpAffine(
        image,
        matrix,
        (int(size), int(size)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )