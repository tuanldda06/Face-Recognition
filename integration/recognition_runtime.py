
from __future__ import annotations

from pathlib import Path

import numpy as np

from FaceNet.matching import validate_threshold
from . import recognition_pipeline

V2_CONTROLLED_THRESHOLD = float.fromhex("0x1.54be2dd42d71dp-1")
V1_ROLLBACK_THRESHOLD = float.fromhex("0x1.d9df8d3362a3cp-1")
DEFAULT_THRESHOLD_PROFILE = "v2"

_PROFILE_THRESHOLDS = {
    "v2": V2_CONTROLLED_THRESHOLD,
    "v1": V1_ROLLBACK_THRESHOLD,
}
_UNSET_THRESHOLD = object()


def recognize_image(
    image_rgb: np.ndarray,
    detector,
    facenet_net,
    db_path: str | Path,
    model_version: str,
    *,
    threshold_profile: str = DEFAULT_THRESHOLD_PROFILE,
    threshold: object = _UNSET_THRESHOLD,
) -> list[dict[str, object]]:
    """Nhận ảnh RGB gốc, trả nguyên output của recognition pipeline.

    Bỏ qua hai keyword để dùng V2; threshold_profile="v1" để rollback.
    threshold tường minh được ưu tiên hơn profile và dùng validator hiện có.
    Profile phải hợp lệ cả khi có override. None, bool, NaN/Inf và giá trị
    ngoài [-1,1] bị từ chối; không coi None là yêu cầu dùng mặc định.

    Caller tái sử dụng detector/network đã nạp và registry tương thích với
    model_version. Không cache, thay model hoặc chuyển lỗi thành Unknown.
    """
    if not isinstance(threshold_profile, str):
        raise TypeError("threshold_profile phải là chuỗi 'v1' hoặc 'v2'.")
    if threshold_profile not in _PROFILE_THRESHOLDS:
        raise ValueError("threshold_profile phải là 'v1' hoặc 'v2'.")

    # Sentinel phân biệt không truyền threshold với override None không hợp lệ.
    selected = (
        _PROFILE_THRESHOLDS[threshold_profile]
        if threshold is _UNSET_THRESHOLD
        else threshold
    )
    selected = validate_threshold(selected)
    return recognition_pipeline.recognize_faces_from_image(
        image_rgb=image_rgb,
        detector=detector,
        facenet_net=facenet_net,
        db_path=db_path,
        model_version=model_version,
        threshold=selected,
    )
