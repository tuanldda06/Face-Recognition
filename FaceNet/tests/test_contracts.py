"""Hồi quy validator với notebook 01/10; tất cả dữ liệu lỗi chỉ ở RAM."""

from copy import deepcopy
import unittest

import numpy as np

from _support import SOURCE_ROOT, function_body, notebook_function_body, notebook_namespace
from FaceNet import contracts


CONTRACT_FUNCTIONS = (
    "validate_face", "validate_aligned_face", "validate_coordinate_sequence",
    "validate_detection_metadata", "validate_mtcnn_output",
)


class TestContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = notebook_namespace(
            "10_mtcnn_facenet_interface.ipynb", CONTRACT_FUNCTIONS,
            ("LANDMARK_NAMES", "DETECTION_KEYS", "NUMERIC_TYPES"),
        )

    def setUp(self):
        self.face = np.arange(160 * 160 * 3, dtype=np.uint8).reshape(160, 160, 3)
        self.landmarks = {
            "left_eye": [30.0, 50.0], "right_eye": [70.0, 50.0],
            "nose": [50.0, 75.0], "mouth_left": [35.0, 100.0],
            "mouth_right": [65.0, 100.0],
        }

    def assert_same_behavior(self, name, *args):
        outcomes = []
        for function in (self.original[name], getattr(contracts, name)):
            try:
                result = function(*args)
                outcomes.append((None, result))
            except (TypeError, ValueError) as error:
                outcomes.append((type(error), str(error)))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_validator_bodies_preserve_source(self):
        source = (SOURCE_ROOT / "FaceNet" / "contracts.py").read_text(encoding="utf-8")
        for name in CONTRACT_FUNCTIONS:
            with self.subTest(name=name):
                self.assertEqual(function_body(source, name), notebook_function_body("10_mtcnn_facenet_interface.ipynb", name))
        self.assertEqual(notebook_function_body("01_facenet_model.ipynb", "validate_face"), notebook_function_body("10_mtcnn_facenet_interface.ipynb", "validate_face"))

    def test_valid_image_is_not_changed(self):
        before = self.face.copy()
        self.assert_same_behavior("validate_face", self.face)
        self.assert_same_behavior("validate_aligned_face", self.face)
        np.testing.assert_array_equal(self.face, before)

    def test_image_rejections_match_notebook(self):
        cases = [
            None, [], np.empty((0, 0, 3), np.uint8),
            np.zeros((80, 80, 3), np.uint8), np.zeros((160, 160), np.uint8),
            np.zeros((160, 160, 4), np.uint8),
            np.zeros((160, 160, 3), np.float32), np.zeros((160, 160, 3), np.float64),
            np.full((160, 160, 3), np.nan), np.full((160, 160, 3), np.inf),
            np.zeros((160, 160, 3), np.int16),
        ]
        for index, image in enumerate(cases):
            for name in ("validate_face", "validate_aligned_face"):
                with self.subTest(case=index, validator=name):
                    self.assert_same_behavior(name, image)
                    with self.assertRaises((TypeError, ValueError)):
                        getattr(contracts, name)(image)

    def test_none_exceptions_remain_distinct(self):
        with self.assertRaises(TypeError):
            contracts.validate_face(None)
        with self.assertRaises(ValueError):
            contracts.validate_aligned_face(None)

    def test_valid_detection_lists_and_metadata_remain_unchanged(self):
        for confidence in (0.0, 1.0, np.float32(0.95)):
            metadata = {"bbox": [0, 0, 160.0, 160.0], "confidence": confidence,
                        "landmarks": deepcopy(self.landmarks)}
            detections = [{"face_rgb": self.face, **metadata}, {"face_rgb": self.face}]
            before = deepcopy(metadata)
            self.assert_same_behavior("validate_mtcnn_output", detections)
            self.assertEqual({k: v for k, v in detections[0].items() if k != "face_rgb"}, before)
            self.assertIs(detections[0]["face_rgb"], self.face)
        self.assert_same_behavior("validate_mtcnn_output", [])

    def test_invalid_container_and_required_fields(self):
        for value in (None, (), {}, [None], [self.face], [{}],
                      [{"face_rgb": None}], [{"face_rgb": self.face, "other": 1}]):
            with self.subTest(value_type=type(value).__name__):
                self.assert_same_behavior("validate_mtcnn_output", value)
                with self.assertRaises((TypeError, ValueError)):
                    contracts.validate_mtcnn_output(value)

    def test_invalid_bbox_is_rejected_without_clipping(self):
        cases = [None, [0, 0, 10], np.array([0, 0, 10, 10]),
                 [-1, 0, 10, 10], [0, 0, np.nan, 10], [0, 0, 10, np.inf],
                 [10, 0, 5, 10], [0, 0, 0, 10], [False, 0, 10, 10], [0, 0, "10", 10]]
        for bbox in cases:
            detection = {"face_rgb": self.face, "bbox": bbox}
            with self.subTest(bbox=str(bbox)):
                self.assert_same_behavior("validate_mtcnn_output", [detection])
                with self.assertRaises((TypeError, ValueError)):
                    contracts.validate_mtcnn_output([detection])
                self.assertIs(detection["bbox"], bbox)

    def test_invalid_confidence(self):
        for value in (None, True, np.bool_(False), "0.9", -.01, 1.01, np.nan, np.inf):
            with self.subTest(confidence=str(value)):
                detections = [{"face_rgb": self.face, "confidence": value}]
                self.assert_same_behavior("validate_mtcnn_output", detections)
                with self.assertRaises((TypeError, ValueError)):
                    contracts.validate_mtcnn_output(detections)

    def test_invalid_landmarks(self):
        cases = [None, [], {"nose": [1, 2]}, {**self.landmarks, "extra": [1, 2]}]
        for point in ([1], [1, 2, 3], [-1, 2], [np.nan, 2], [np.inf, 2], [True, 2], ["1", 2]):
            cases.append({**self.landmarks, "nose": point})
        for points in cases:
            with self.subTest(landmarks=str(points)):
                detections = [{"face_rgb": self.face, "landmarks": points}]
                self.assert_same_behavior("validate_mtcnn_output", detections)
                with self.assertRaises((TypeError, ValueError)):
                    contracts.validate_mtcnn_output(detections)

    def test_error_keeps_face_index_and_cause(self):
        with self.assertRaisesRegex(ValueError, "face_index 1") as context:
            contracts.validate_mtcnn_output([{"face_rgb": self.face}, {"face_rgb": self.face, "bbox": [-1, 0, 1, 1]}])
        self.assertIsInstance(context.exception.__cause__, ValueError)

    def test_no_implicit_original_image_bounds(self):
        # Giữ hành vi gốc: không có kích thước ảnh gốc để kiểm tra giới hạn trên.
        self.assert_same_behavior("validate_mtcnn_output", [{"face_rgb": self.face, "bbox": [1000, 1000, 1100, 1100]}])


if __name__ == "__main__":
    unittest.main()
