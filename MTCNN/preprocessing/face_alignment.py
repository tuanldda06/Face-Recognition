"""Five-landmark similarity alignment (NumPy + OpenCV, no PyTorch).

Landmark order: left eye, right eye, nose, left mouth, right mouth.
Coordinates must be in the original image's pixel coordinate system.

The reference landmarks below come from the ArcFace 112x112 template.
This is an optional preprocessing choice, NOT the default facenet-pytorch crop.
"""

import cv2
import numpy as np

from MTCNN.preprocessing.image_processing import check_rgb


TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def validate_landmarks(landmarks):
    """Check basic geometry of five landmarks in original-image pixels."""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Landmarks phải có shape (5, 2) và giá trị hữu hạn")

    left_eye, right_eye = points[:2]
    eye_vector = right_eye - left_eye
    eye_distance = np.linalg.norm(eye_vector)
    span = np.ptp(points, axis=0)
    if eye_distance < max(1.0, 0.05 * np.linalg.norm(span)):
        raise ValueError("Hai mắt quá gần nhau hoặc bị trùng")

    eye_center = points[:2].mean(axis=0)
    mouth_center = points[3:].mean(axis=0)
    direction = mouth_center - eye_center
    separation = np.linalg.norm(direction)
    # Reject five points lying (nearly) on one line.
    cross = abs(np.linalg.det(np.stack([eye_vector, direction])))
    if separation < 1.0 or cross / (eye_distance * separation) < 0.1:
        raise ValueError("Landmarks gần thẳng hàng, không thể align tin cậy")
    return points


def similarity_transform(source, target):
    """Return the 2x3 affine matrix mapping five source points to target."""
    src = validate_landmarks(source)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != (5, 2) or dst.shape != (5, 2):
        raise ValueError("source và target phải có shape (5, 2)")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Landmarks không được chứa NaN/Inf")

    src_center, dst_center = src.mean(axis=0), dst.mean(axis=0)
    x, y = src - src_center, dst - dst_center
    energy = np.sum(x * x)
    if energy < 1e-8:
        raise ValueError("Không thể align: các landmarks nguồn bị trùng nhau")

    # Least-squares similarity transform using orthogonal Procrustes.
    u, singular_values, vt = np.linalg.svd(x.T @ y)
    correction = np.diag([1.0, np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    scale = np.sum(singular_values * np.diag(correction)) / energy
    translation = dst_center - scale * (src_center @ rotation)

    # Row-vector convention above -> column-vector convention of cv2.warpAffine.
    return np.column_stack(((scale * rotation).T, translation)).astype(np.float32)


def align_face(image, landmarks, size=160):
    """Align one full RGB uint8 image using 5 original-image landmarks.

    Return one RGB uint8 image with shape (size, size, 3).
    """
    image = check_rgb(image)
    if not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size phải là số nguyên dương")

    target = TEMPLATE_112 * (size / 112.0)
    matrix = similarity_transform(landmarks, target)

    # Test source-image coverage independently of pixel brightness.
    mask = np.ones(image.shape[:2], dtype=np.uint8)
    coverage = cv2.warpAffine(mask, matrix, (int(size), int(size)))
    if np.mean(coverage > 0) < 0.25:
        raise ValueError("Ảnh align có quá nhiều vùng padding ngoài biên")

    return cv2.warpAffine(
        image, matrix, (int(size), int(size)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
