from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HeadPose:
    """Signed yaw, pitch and roll in degrees; see the module's axis convention."""

    yaw: float
    pitch: float
    roll: float


# Arbitrary length units; only relative geometry matters. At rest x points
# image-right, y down; negative z brings the nose toward the camera. Eyes and
# mouth are bilateral, with a projecting nose providing non-planar depth.
_MODEL_POINTS = np.array([
    [-32.0, -28.0, 0.0],
    [32.0, -28.0, 0.0],
    [0.0, 0.0, -24.0],
    [-24.0, 32.0, -5.0],
    [24.0, 32.0, -5.0],
], dtype=np.float64)
_MODEL_POINTS.setflags(write=False)

FRONT_MAX_YAW = 16.0
FRONT_MAX_PITCH = 10.0
TURN_MIN_YAW = 18.0
TURN_MAX_YAW = 50.0
TURN_MAX_PITCH = 12.0
TILT_MIN_PITCH = 15.0
TILT_MAX_PITCH = 35.0
UP_MAX_YAW = 12.0
DOWN_MAX_YAW = 14.0
MAX_CLASSIFIED_ROLL = 15.0


def _valid_geometry(points: np.ndarray, height: int, width: int) -> float | None:
    """Return interocular pixel distance, or None for unreliable geometry."""
    if np.any(points < 0) or np.any(points >= [width, height]):
        return None
    eyes = points[1] - points[0]
    eye_distance = float(np.linalg.norm(eyes))
    if eye_distance < 12.0 or eyes[0] <= 0:
        return None
    # A local eye-line basis makes geometry checks invariant to moderate roll.
    horizontal = eyes / eye_distance
    downward = np.array([-horizontal[1], horizontal[0]])
    local = (points - points[:2].mean(axis=0)) @ np.column_stack(
        [horizontal, downward]
    ) / eye_distance
    mouth = local[3:].mean(axis=0)
    mouth_width = local[4, 0] - local[3, 0]
    if not (0.25 <= mouth_width <= 1.25 and 0.30 <= mouth[1] <= 1.80):
        return None
    if abs(mouth[0]) > 0.65 or abs(local[4, 1] - local[3, 1]) > 0.40:
        return None
    if not (-0.15 <= local[2, 1] <= 1.15 * mouth[1]):
        return None
    if abs(local[2, 0]) > 0.85:
        return None
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    if np.any(distances[np.triu_indices(5, 1)] < max(2.0, 0.06 * eye_distance)):
        return None
    return eye_distance


def _angles(rotation: np.ndarray) -> HeadPose:
    """Decompose Rz @ Ry @ Rx without treating a Rodrigues vector as Euler."""
    pitch = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
    roll = np.arctan2(rotation[1, 0], rotation[0, 0])
    return HeadPose(*map(float, np.rad2deg([yaw, pitch, roll])))


def estimate_head_pose(landmarks: object, image_shape: object) -> HeadPose | None:
    """Fit one original-image MTCNN ndarray (5,2), returning pose or None.

    ``image_shape`` is (height,width) or frame.shape (height,width,channels).
    Real numeric array-like points are accepted without modifying the input.
    Missing/wrong-shape/nonfinite/out-of-frame/degenerate points, tiny faces,
    failed/ambiguous PnP, large reprojection error and extreme views return None.
    There is no fallback to a fabricated frontal pose.

    SQPnP supports five non-planar points; unseeded ITERATIVE's DLT needs six.
    Refine each SQPnP candidate with LM and require all five points to fit
    (RMS <= 6% and worst residual <= 12% of interocular distance), positive
    camera depth, |yaw|<=60, |pitch|<=45, |roll|<=60. Fits within the down
    pose bounds allow up to 11% RMS and 14% worst residual. Near-equal fits
    differing by >12 degrees are rejected. See the official solver contract:
    https://docs.opencv.org/4.12.0/d5/d1f/calib3d_solvePnP.html
    """
    try:
        shape = tuple(image_shape)
        if len(shape) not in (2, 3) or any(
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, (int, np.integer)) or v <= 0
            for v in shape
        ):
            return None
        height, width = map(int, shape[:2])
        raw = np.asarray(landmarks)
        if raw.shape != (5, 2) or raw.dtype.kind not in "iuf":
            return None
        points = np.ascontiguousarray(raw, dtype=np.float64)
        if not np.isfinite(points).all():
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    eye_distance = _valid_geometry(points, height, width)
    if eye_distance is None:
        return None

    focal = float(max(height, width))
    camera = np.array([[focal, 0.0, width / 2.0],
                       [0.0, focal, height / 2.0],
                       [0.0, 0.0, 1.0]])
    try:
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            _MODEL_POINTS, points, camera, None, flags=cv2.SOLVEPNP_SQPNP
        )
    except cv2.error:
        return None
    if not count:
        return None

    candidates: list[tuple[float, HeadPose]] = []
    for rvec, tvec in zip(rvecs, tvecs):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                _MODEL_POINTS, points, camera, None, rvec.copy(), tvec.copy()
            )
            if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            camera_points = _MODEL_POINTS @ rotation.T + tvec.reshape(3)
            if np.any(camera_points[:, 2] <= 0):
                continue
            pose = _angles(rotation)
            if abs(pose.yaw) > 60 or abs(pose.pitch) > 45 or abs(pose.roll) > 60:
                continue
            projected, _ = cv2.projectPoints(_MODEL_POINTS, rvec, tvec, camera, None)
            residual = np.linalg.norm(projected.reshape(5, 2) - points, axis=1)
            rms = float(np.sqrt(np.mean(residual ** 2)) / eye_distance)
            # Downward views can leave larger residuals against the generic
            # five-point template even when the landmarks have valid geometry.
            rms_limit, point_limit = (0.11, 0.14) if classify_pose(pose) == "down" else (0.06, 0.12)
            if rms > rms_limit or float(residual.max()) / eye_distance > point_limit:
                continue
            candidates.append((rms, pose))
        except cv2.error:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda candidate: candidate[0])
    best_error, best = candidates[0]
    for error, other in candidates[1:]:
        if error <= best_error + 0.01 and max(
            abs(best.yaw - other.yaw), abs(best.pitch - other.pitch),
            abs(best.roll - other.roll),
        ) > 12.0:
            return None
    return best


def classify_pose(pose: HeadPose | None) -> str | None:
    """Return front/left/right/up/down, or None for uncertain/mixed poses.

    Labels left/right always refer to the USER'S ANATOMICAL side. Front needs
    |yaw|<=16 and |pitch|<=10; side turns need 18<=|yaw|<=50 and |pitch|<=12;
    up/down need 15<=|pitch|<=35 with |yaw|<=12 for up or <=14 for down.
    All need |roll|<=15.
    Gaps between these ranges deliberately prevent premature captures.
    This stateless function does not supply temporal stability/hysteresis.
    """
    if not isinstance(pose, HeadPose):
        return None
    values = np.asarray([pose.yaw, pose.pitch, pose.roll], dtype=np.float64)
    if not np.isfinite(values).all():
        return None
    yaw, pitch, roll = values
    if abs(roll) > MAX_CLASSIFIED_ROLL:
        return None
    if abs(yaw) <= FRONT_MAX_YAW and abs(pitch) <= FRONT_MAX_PITCH:
        return "front"
    if TURN_MIN_YAW <= abs(yaw) <= TURN_MAX_YAW and abs(pitch) <= TURN_MAX_PITCH:
        return "right" if yaw > 0 else "left"
    if -TILT_MAX_PITCH <= pitch <= -TILT_MIN_PITCH and abs(yaw) <= UP_MAX_YAW:
        return "up"
    if TILT_MIN_PITCH <= pitch <= TILT_MAX_PITCH and abs(yaw) <= DOWN_MAX_YAW:
        return "down"
    return None
