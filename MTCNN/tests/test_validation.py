"""Edge-case validation for the MTCNN -> FaceNet RGB uint8 interface."""

import numpy as np
import pytest

from MTCNN.mtcnn import MTCNN, DetectionResult, validate_detection_result
from MTCNN.preprocessing.face_alignment import TEMPLATE_112, align_face, validate_landmarks
from MTCNN.preprocessing.image_processing import crop_and_resize_rgb


def detector_without_models():
    # Extraction only, so no weights or GPU are required.
    return MTCNN.__new__(MTCNN)


def example_result():
    return DetectionResult(
        boxes=np.array([[80, 60, 240, 270]], dtype=np.float32),
        scores=np.array([0.95], dtype=np.float32),
        landmarks=(TEMPLATE_112 + [100, 100])[None].copy(),
    )


@pytest.mark.parametrize("points", [
    np.zeros((5, 2)),                 # both eyes at same pixel
    np.array([[0, 0], [10, 0], [4, 0], [2, 0], [8, 0]]),  # collinear
    np.ones((5, 2)) * np.nan,       # invalid numeric values
    np.ones((5, 2)) * np.inf,
    np.zeros((4, 2)),                # missing landmark
])
def test_invalid_landmarks_raise_value_error(points):
    with pytest.raises(ValueError):
        validate_landmarks(points)


def test_valid_landmarks_accept_rotated_face():
    angle = np.deg2rad(31)
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    points = TEMPLATE_112 @ rot.T + [90, 45]
    np.testing.assert_allclose(validate_landmarks(points), points)


def test_align_face_identity_keeps_rgb_dtype_and_shape():
    # Source landmarks already match 112x112 reference.
    image = np.random.default_rng(1).integers(0, 255, (112, 112, 3), dtype=np.uint8)
    result = align_face(image, TEMPLATE_112, size=112)
    assert result.dtype == np.uint8
    assert result.shape == image.shape
    np.testing.assert_allclose(result, image, atol=1)


def test_align_face_rejects_nearly_empty_warp():
    image = np.ones((100, 100, 3), dtype=np.uint8) * 150
    # Landmark coordinates belong far outside this image; warp would be mostly black.
    landmarks = TEMPLATE_112 + [400, 400]
    with pytest.raises(ValueError, match="padding"):
        align_face(image, landmarks)


@pytest.mark.parametrize("change", [
    lambda r: setattr(r, "scores", np.array([], dtype=np.float32)),
    lambda r: setattr(r, "landmarks", np.empty((0, 5, 2), dtype=np.float32)),
    lambda r: setattr(r, "scores", np.array([1.2], dtype=np.float32)),
    lambda r: r.landmarks.__setitem__((0, 0, 0), np.nan),
    lambda r: r.boxes.__setitem__((0, 2), r.boxes[0, 0] - 5),
])
def test_metadata_inconsistent_or_invalid_rejected(change):
    result = example_result()
    change(result)
    with pytest.raises(ValueError):
        validate_detection_result(result)
    with pytest.raises(ValueError):
        detector_without_models().extract_faces_rgb(
            np.ones((320, 320, 3), dtype=np.uint8), result, align=True
        )


def test_empty_metadata_must_still_be_consistent():
    malformed = DetectionResult(
        np.empty((0, 4), dtype=np.float32),
        np.array([0.8], dtype=np.float32),
        np.empty((0, 5, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError):
        detector_without_models().extract_faces_rgb(
            np.zeros((80, 80, 3), dtype=np.uint8), malformed
        )


def test_near_boundary_box_uses_padding_without_discarding_face():
    image = np.full((100, 100, 3), 170, dtype=np.uint8)
    box = np.array([[-10, -10, 50, 50]], dtype=np.float32)
    crops, actual = crop_and_resize_rgb(image, box, 24)
    assert crops.shape == (1, 24, 24, 3)
    assert crops.dtype == np.uint8
    assert (crops[0, 0, 0] == 0).all()  # black PIL padding
    assert (crops[0, -1, -1] == 170).all()
    assert (actual[0] == box[0]).all()
    # Public extraction accepts a partially visible face.
    result = DetectionResult(box, np.array([0.9]), np.zeros((1, 5, 2)))
    assert detector_without_models().extract_faces_rgb(image, result, align=False).shape == (1, 160, 160, 3)


def test_face_fully_outside_image_rejected_before_facenet():
    image = np.full((100, 100, 3), 100, dtype=np.uint8)
    result = DetectionResult(
        np.array([[300, 300, 390, 390]], dtype=np.float32),
        np.array([0.92], dtype=np.float32),
        (TEMPLATE_112 + [300, 300])[None],
    )
    for align in [False, True]:
        with pytest.raises(ValueError, match="ngoài ảnh"):
            detector_without_models().extract_faces_rgb(image, result, align=align)


def test_rgb_output_preserves_batch_order_and_uint8():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[:, :150] = (240, 10, 20)
    image[:, 150:] = (10, 200, 30)
    result = DetectionResult(
        np.array([[160, 40, 240, 120], [10, 40, 90, 120]], dtype=np.float32),
        np.array([0.88, 0.96], dtype=np.float32),
        np.zeros((2, 5, 2), dtype=np.float32),
    )
    output = detector_without_models().extract_faces_rgb(image, result)
    assert output.shape == (2, 160, 160, 3)
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(output[0, 80, 80], [10, 200, 30])
    np.testing.assert_array_equal(output[1, 80, 80], [240, 10, 20])


def test_ensure_no_second_normalization_from_rgb_extraction():
    image = np.full((320, 320, 3), 200, dtype=np.uint8)
    faces = detector_without_models().extract_faces_rgb(image, example_result(), align=True)
    assert faces.dtype == np.uint8
    # The aligned area of this constant image must retain 0-255 pixel values.
    assert faces[0, 80, 80, 0] == 200
