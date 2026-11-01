"""Unit tests for MTCNN alignment and FaceNet output contract."""
import numpy as np
import pytest
import torch

from MTCNN.mtcnn import MTCNN, DetectionResult
from MTCNN.preprocessing.face_alignment import (
    TEMPLATE_112, align_face, similarity_transform,
)
from MTCNN.preprocessing.image_processing import crop_and_resize_rgb


def test_similarity_transform_maps_rotated_landmarks():
    template = TEMPLATE_112.astype(np.float64)
    angle = np.deg2rad(19)
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    # Source comes from scaled, rotated and shifted target.
    src = template @ (1.7 * rot).T + [92, 135]
    matrix = similarity_transform(src, template)
    mapped = np.c_[src, np.ones(len(src))] @ matrix.T
    np.testing.assert_allclose(mapped, template, atol=1e-4)


def test_alignment_returns_rgb_uint8_and_validates_landmarks():
    image = np.full((280, 300, 3), 128, dtype=np.uint8)
    landmarks = TEMPLATE_112 + [85, 95]
    crop = align_face(image, landmarks, size=160)
    assert crop.dtype == np.uint8
    assert crop.shape == (160, 160, 3)
    with pytest.raises(ValueError):
        align_face(image, np.zeros((4, 2)))
    with pytest.raises(ValueError):
        align_face(image, np.zeros((5, 2)))


def test_rgb_crop_handles_out_of_bounds():
    image = np.ones((80, 80, 3), dtype=np.uint8) * 80
    crops, actual = crop_and_resize_rgb(
        image, np.array([[-5, -5, 30, 30]], dtype=np.float32), 24
    )
    assert crops.shape == (1, 24, 24, 3)
    assert crops.dtype == np.uint8
    assert actual.shape == (1, 4)


def test_empty_results_return_stable_shapes():
    detector = MTCNN.__new__(MTCNN)  # Only testing public extraction functions.
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = DetectionResult.empty()
    assert detector.extract_faces_rgb(image, result, align=True).shape == (0, 160, 160, 3)
    tensors = detector.extract_faces(image, result, align=True)
    assert isinstance(tensors, torch.Tensor)
    assert tensors.shape == (0, 3, 160, 160)
    assert tensors.dtype == torch.float32


def test_aligned_numpy_and_torch_outputs_match():
    detector = MTCNN.__new__(MTCNN)
    image = np.full((300, 320, 3), 172, dtype=np.uint8)
    landmarks = TEMPLATE_112 + [100, 80]
    result = DetectionResult(
        boxes=np.array([[90, 60, 230, 240]], dtype=np.float32),
        scores=np.array([0.99], dtype=np.float32),
        landmarks=landmarks[None],
    )
    faces_uint8 = detector.extract_faces_rgb(image, result, align=True)
    assert faces_uint8.shape == (1, 160, 160, 3)
    assert faces_uint8.dtype == np.uint8
    faces_tensor = detector.extract_faces(image, result, align=True)
    assert faces_tensor.shape == (1, 3, 160, 160)
    expected = (faces_uint8.astype(np.float32) - 127.5) / 128.0
    np.testing.assert_allclose(faces_tensor.numpy().transpose(0, 2, 3, 1), expected)


def test_unaligned_square_crop_not_stretched():
    detector = MTCNN.__new__(MTCNN)
    image = np.ones((150, 160, 3), dtype=np.uint8) * 127
    result = DetectionResult(
        boxes=np.array([[30, 20, 110, 120]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        landmarks=np.zeros((1, 5, 2), dtype=np.float32),
    )
    face = detector.extract_faces_rgb(image, result, align=False)
    assert face.shape == (1, 160, 160, 3)
    assert face.dtype == np.uint8
