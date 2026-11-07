"""Independent perspective projections verify MTCNN order and anatomical signs."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from application.head_pose import HeadPose, classify_pose, estimate_head_pose


# Explicit independent fixture: do not import the estimator's model or camera.
# Row order is the actual decoded O-Net order, not an interleaved raw vector.
FACE = np.array([[-32., -28., 0.], [32., -28., 0.], [0., 0., -24.],
                 [-24., 32., -5.], [24., 32., -5.]])
SHAPE = (480, 640, 3)


def project(yaw=0., pitch=0., roll=0., *, shape=SHAPE, translation=(0., 0., 600.),
            focal_scale=1., face=FACE):
    """Generate ground truth with explicit Euler matrices and cv2.projectPoints."""
    y, p, r = np.deg2rad([yaw, pitch, roll])
    rx = np.array([[1., 0., 0.], [0., np.cos(p), -np.sin(p)],
                   [0., np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(y), 0., np.sin(y)], [0., 1., 0.],
                   [-np.sin(y), 0., np.cos(y)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0.],
                   [np.sin(r), np.cos(r), 0.], [0., 0., 1.]])
    rvec, _ = cv2.Rodrigues(rz @ ry @ rx)
    h, w = shape[:2]
    focal = max(h, w) * focal_scale
    camera = np.array([[focal, 0., w / 2], [0., focal, h / 2], [0., 0., 1.]])
    points, _ = cv2.projectPoints(face, rvec, np.array(translation), camera, None)
    return points.reshape(5, 2)


@pytest.mark.parametrize("yaw,pitch,roll,label", [
    (0, 0, 0, "front"), (-28, 0, 0, "left"), (28, 0, 0, "right"),
    (0, -24, 0, "up"), (0, 24, 0, "down"),
    (-30, 5, -8, "left"), (27, -4, 9, "right"),
    (5, -22, 7, "up"), (-5, 22, -7, "down"),
    (0, 0, 25, None), (0, 0, -25, None), (26, 24, 3, None),
])
def test_recover_projected_angles_and_classify(yaw, pitch, roll, label):
    pose = estimate_head_pose(project(yaw, pitch, roll), SHAPE)
    assert isinstance(pose, HeadPose)
    np.testing.assert_allclose([pose.yaw, pose.pitch, pose.roll],
                               [yaw, pitch, roll], atol=0.05)
    assert classify_pose(pose) == label


def test_signs_match_anatomical_instructions_in_unmirrored_camera():
    # Subject's right is viewer's left. A yaw sign bug reverses this test.
    right = project(yaw=28)
    left = project(yaw=-28)
    assert right[2, 0] < right[:2, 0].mean()
    assert left[2, 0] > left[:2, 0].mean()
    assert classify_pose(estimate_head_pose(right, SHAPE)) == "right"
    assert classify_pose(estimate_head_pose(left, SHAPE)) == "left"
    # Looking down moves the projecting nose down relative to the eye line.
    down, up = project(pitch=24), project(pitch=-24)
    assert down[2, 1] - down[:2, 1].mean() > up[2, 1] - up[:2, 1].mean()


@pytest.mark.parametrize("shape,translation", [
    ((720, 1280, 3), (90., -50., 850.)),
    ((800, 600), (-70., 60., 700.)),
    ((480, 640, 3), (0., 0., 1400.)),
])
def test_original_pixel_coordinates_translation_resolution_and_scale(shape, translation):
    points = project(-26, 4, 6, shape=shape, translation=translation)
    pose = estimate_head_pose(points, shape)
    assert pose is not None
    np.testing.assert_allclose([pose.yaw, pose.pitch, pose.roll], [-26, 4, 6], atol=.05)
    assert classify_pose(pose) == "left"


def test_accepts_mtcnn_float32_noncontiguous_array_and_does_not_modify_input():
    backing = np.zeros((5, 4), dtype=np.float32)
    backing[:, ::2] = project(28)
    points = backing[:, ::2]
    assert not points.flags.c_contiguous
    before = backing.copy()
    assert classify_pose(estimate_head_pose(points, SHAPE)) == "right"
    np.testing.assert_array_equal(backing, before)


@pytest.mark.parametrize("yaw,pitch,label", [
    (0, 0, "front"), (-30, 0, "left"), (30, 0, "right"),
    (0, -25, "up"), (0, 25, "down"),
])
def test_mild_landmark_noise_and_camera_approximation_keep_five_pose_labels(yaw, pitch, label):
    rng = np.random.default_rng(41)
    for focal_scale in (.85, 1.15):
        for _ in range(6):
            points = project(yaw, pitch, focal_scale=focal_scale)
            points += rng.normal(0, .35, size=(5, 2))
            pose = estimate_head_pose(points, SHAPE)
            assert pose is not None
            assert abs(pose.yaw - yaw) < 5
            assert abs(pose.pitch - pitch) < 5
            assert classify_pose(pose) == label


def test_mild_independent_face_proportion_change_does_not_force_frontal_turn():
    face = FACE.copy()
    face[2, 2] *= .9
    face[3:, 0] *= 1.05
    pose = estimate_head_pose(project(-30, face=face), SHAPE)
    assert pose is not None
    assert abs(pose.yaw + 30) < 6
    assert classify_pose(pose) == "left"


@pytest.mark.parametrize("points", [
    None, [], np.zeros((4, 2)), np.zeros((1, 5, 2)), np.zeros(10),
    np.zeros((5, 2)), np.ones((5, 2), dtype=bool),
    np.ones((5, 2), dtype=complex), np.full((5, 2), "1"),
    np.full((5, 2), np.nan), np.full((5, 2), np.inf),
    np.array([[100, 100], [150, 100], [125, 100], [110, 100], [140, 100]]),
])
def test_invalid_landmarks_return_none(points):
    assert estimate_head_pose(points, SHAPE) is None
    assert classify_pose(estimate_head_pose(points, SHAPE)) is None


@pytest.mark.parametrize("shape", [None, (), (480,), (0, 640), (-1, 640, 3),
                                  (480., 640), (True, 640), (480, 640, 0),
                                  (480, 640, 3, 1), "480x640"])
def test_invalid_image_shape_returns_none(shape):
    assert estimate_head_pose(project(), shape) is None


@pytest.mark.parametrize("change", [
    lambda p: p.__setitem__(0, p[1]),
    lambda p: p.__setitem__(2, p[3]),
    lambda p: p.__setitem__(2, [-1, 100]),
    lambda p: p.__setitem__(2, [640, 240]),
    lambda p: p.__setitem__(2, [320, 480]),
    lambda p: p.__setitem__([0, 1], p[[1, 0]]),
    lambda p: p.__setitem__([3, 4], p[[4, 3]]),
    lambda p: p.__setitem__(2, p[2] + [0, 70]),
    lambda p: p.__setitem__(4, p[4] + [0, -35]),
])
def test_bad_geometry_and_wrong_correspondences_rejected(change):
    points = project()
    change(points)
    assert estimate_head_pose(points, SHAPE) is None


def test_tiny_face_rejected_even_with_perfect_projection():
    assert estimate_head_pose(project(translation=(0., 0., 5000.)), SHAPE) is None


def test_reprojection_gate_rejects_plausible_but_inconsistent_geometry():
    points = project()
    points[3] += [14, -8]
    # Still has separated eyes, nose between eyes/mouth and ordered corners.
    assert estimate_head_pose(points, SHAPE) is None


def test_downward_view_accepts_plausible_face_proportions():
    face = FACE.copy()
    face[2, 1] += 20
    face[3:, 1] += 10
    pose = estimate_head_pose(project(yaw=8, pitch=20, face=face), SHAPE)
    assert classify_pose(pose) == "down"


@pytest.mark.parametrize("yaw,pitch", [(72, 0), (0, 55), (0, -55)])
def test_extreme_views_rejected(yaw, pitch):
    assert estimate_head_pose(project(yaw, pitch), SHAPE) is None


def test_mirrored_pixels_with_unchanged_correspondences_are_rejected():
    points = project(28)
    points[:, 0] = SHAPE[1] - 1 - points[:, 0]
    assert estimate_head_pose(points, SHAPE) is None


def test_solver_failure_does_not_become_front():
    with patch("application.head_pose.cv2.solvePnPGeneric", side_effect=cv2.error("failed")):
        assert estimate_head_pose(project(), SHAPE) is None


@pytest.mark.parametrize("pose,label", [
    (HeadPose(16, -10, 15), "front"),
    (HeadPose(-14.85, 6.1, -5), "front"),
    (HeadPose(-18, 12, -15), "left"), (HeadPose(50, -12, 15), "right"),
    (HeadPose(-12, -15, 15), "up"), (HeadPose(12, 35, -15), "down"),
    (HeadPose(14, 16, 0), "down"), (HeadPose(15, 16, 0), None),
    (HeadPose(13, -16, 0), None),
    (HeadPose(17, 0, 0), None), (HeadPose(0, 12, 0), None),
    (HeadPose(28, 20, 0), None), (HeadPose(51, 0, 0), None),
    (HeadPose(0, 36, 0), None), (HeadPose(0, 0, 16), None),
    (HeadPose(float("nan"), 0, 0), None), (HeadPose(0, float("inf"), 0), None),
    (HeadPose(0, 0, float("nan")), None), (None, None),
])
def test_classification_thresholds_gaps_and_invalid_angles(pose, label):
    assert classify_pose(pose) == label
