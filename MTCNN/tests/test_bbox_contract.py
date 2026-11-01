"""Bounding-box and landmark mapping contract across MTCNN stages."""

import numpy as np

from MTCNN.detection.bounding_box import (
    calibrate_boxes, decode_landmarks, nms, square_boxes, valid_box_indices,
)


def test_nms_union_and_min_differ_for_nested_boxes():
    boxes = np.array([[0, 0, 100, 100], [20, 20, 70, 70]], dtype=np.float32)
    scores = np.array([0.95, 0.8], dtype=np.float32)
    np.testing.assert_array_equal(nms(boxes, scores, 0.7, mode='union'), [0, 1])
    np.testing.assert_array_equal(nms(boxes, scores, 0.7, mode='min'), [0])


def test_nms_empty_boxes():
    keep = nms(np.empty((0, 4)), np.empty((0,)), 0.7)
    assert keep.shape == (0,)
    assert keep.dtype == np.int64


def test_decode_landmarks_uses_x_first_then_y():
    box = np.array([[10, 20, 110, 220]], dtype=np.float32)
    # O-Net encoding: x1..x5, y1..y5, each relative to original crop.
    points = np.array([[0.1, 0.9, 0.5, 0.2, 0.8,
                        0.2, 0.2, 0.5, 0.8, 0.8]], dtype=np.float32)
    expected = np.array([[[20, 60], [100, 60], [60, 120],
                          [30, 180], [90, 180]]], dtype=np.float32)
    np.testing.assert_allclose(decode_landmarks(box, points), expected)


def test_box_calibration_and_square_geometry():
    boxes = np.array([[10, 20, 110, 80]], dtype=np.float32)
    offsets = np.array([[0.1, -0.1, -0.1, 0.1]], dtype=np.float32)
    calibrated = calibrate_boxes(boxes, offsets)
    np.testing.assert_allclose(calibrated, [[20, 14, 100, 86]])
    squared = square_boxes(calibrated)
    np.testing.assert_allclose(squared, [[20, 10, 100, 90]])
    assert np.isclose(squared[0, 2] - squared[0, 0], squared[0, 3] - squared[0, 1])


def test_valid_box_indices_filter_nan_and_inverted_boxes():
    boxes = np.array([[0, 0, 10, 20], [5, 5, 5, 20],
                      [9, 0, 1, 10], [np.nan, 0, 10, 10]], dtype=np.float32)
    np.testing.assert_array_equal(valid_box_indices(boxes), [0])
