
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from FaceNet.contracts import validate_mtcnn_output
from integration import mtcnn_adapter as adapter


def inputs(count=2):
    image = np.full((240, 320, 3), [210, 80, 20], np.uint8)
    boxes = np.array([[10 + i * 20, 20, 110 + i * 20, 150]
                      for i in range(count)], np.float32).reshape(count, 4)
    scores = np.linspace(0.7, 0.95, count, dtype=np.float32)
    points = np.array([[30, 40], [60, 40], [45, 55], [35, 80], [55, 80]], np.float32)
    landmarks = np.stack([points + [i * 20, 0] for i in range(count)]).astype(np.float32) if count else np.empty((0, 5, 2), np.float32)
    result = SimpleNamespace(boxes=boxes, scores=scores, landmarks=landmarks)
    faces = np.empty((count, 160, 160, 3), np.uint8)
    for i in range(count):
        faces[i] = [10 + i * 30, 100, 220 - i * 20]
    return image, result, faces


class TestAdapter(unittest.TestCase):
    def test_one_detection_and_facenet_validator(self):
        image, result, faces = inputs(1)
        output = adapter.mtcnn_to_facenet_detections(image, result, faces)
        self.assertEqual(len(output), 1)
        self.assertEqual(set(output[0]), {"face_rgb", "bbox", "confidence", "landmarks"})
        self.assertEqual(output[0]["bbox"], result.boxes[0].tolist())
        self.assertEqual(output[0]["confidence"], float(result.scores[0]))
        validate_mtcnn_output(output)
        self.assertEqual(output[0]["face_rgb"].shape, (160, 160, 3))
        self.assertEqual(output[0]["face_rgb"].dtype, np.uint8)

    def test_many_faces_keep_order_without_score_sort(self):
        image, result, faces = inputs(3)
        output = adapter.mtcnn_to_facenet_detections(image, result, faces)
        self.assertEqual(len(output), 3)
        for i, item in enumerate(output):
            np.testing.assert_array_equal(item["face_rgb"], faces[i])
            self.assertEqual(item["confidence"], float(result.scores[i]))
            self.assertEqual(item["bbox"], result.boxes[i].tolist())

    def test_empty_still_validates_consistent_shapes(self):
        image, result, faces = inputs(0)
        self.assertEqual(adapter.mtcnn_to_facenet_detections(image, result, faces), [])
        result.scores = np.array([0.9], np.float32)
        with self.assertRaises(ValueError):
            adapter.mtcnn_to_facenet_detections(image, result, faces)

    def test_image_wrong_type_shape_and_dtype(self):
        _, result, faces = inputs()
        for image in (None, [], np.empty((0, 5, 3), np.uint8),
                      np.zeros((5, 5), np.uint8), np.zeros((5, 5, 4), np.uint8),
                      np.zeros((5, 5, 3), np.float32)):
            with self.subTest(shape=getattr(image, "shape", None)):
                with self.assertRaises((TypeError, ValueError)):
                    adapter.mtcnn_to_facenet_detections(image, result, faces)

    def test_faces_wrong_type_shape_count_and_dtype(self):
        image, result, _ = inputs()
        for faces in (None, [], np.empty((1, 160, 160, 3), np.uint8),
                      np.empty((2, 80, 80, 3), np.uint8),
                      np.empty((2, 160, 160, 3), np.float32),
                      np.empty((0, 160, 160, 3), np.uint8)):
            with self.subTest(shape=getattr(faces, "shape", None)):
                with self.assertRaises((TypeError, ValueError)):
                    adapter.mtcnn_to_facenet_detections(image, result, faces)

    def test_result_requires_all_attributes(self):
        image, _, faces = inputs()
        for result in (None, {}, SimpleNamespace(boxes=np.zeros((2, 4)))):
            with self.assertRaises(TypeError):
                adapter.mtcnn_to_facenet_detections(image, result, faces)

    def test_metadata_requires_numpy_real_arrays(self):
        for name in ("boxes", "scores", "landmarks"):
            for dtype in (bool, object, complex):
                image, result, faces = inputs()
                setattr(result, name, getattr(result, name).astype(dtype))
                with self.assertRaises(TypeError):
                    adapter.mtcnn_to_facenet_detections(image, result, faces)
            image, result, faces = inputs()
            setattr(result, name, getattr(result, name).tolist())
            with self.assertRaises(TypeError):
                adapter.mtcnn_to_facenet_detections(image, result, faces)

    def test_invalid_metadata_shapes(self):
        cases = [("boxes", np.zeros((2, 5))), ("boxes", np.zeros(4)),
                 ("scores", np.zeros((2, 1))), ("scores", np.zeros(1)),
                 ("landmarks", np.zeros((2, 4, 2))), ("landmarks", np.zeros((1, 5, 2)))]
        for name, value in cases:
            image, result, faces = inputs()
            setattr(result, name, value)
            with self.assertRaises(ValueError):
                adapter.mtcnn_to_facenet_detections(image, result, faces)

    def test_finite_real_float64_metadata_supported_without_downcast(self):
        image, result, faces = inputs(1)
        result.boxes = result.boxes.astype(np.float64)
        result.boxes[0, 0] = 10 + 2**-30
        result.scores = result.scores.astype(np.float64)
        result.landmarks = result.landmarks.astype(np.float64)
        output = adapter.mtcnn_to_facenet_detections(image, result, faces)
        self.assertEqual(output[0]["bbox"][0], result.boxes[0, 0])

    def test_negative_bbox_clipped_on_copy(self):
        image, result, faces = inputs(1)
        result.boxes[0] = [-10, -20, 100, 120]
        before = result.boxes.copy()
        output = adapter.mtcnn_to_facenet_detections(image, result, faces)
        self.assertEqual(output[0]["bbox"], [0, 0, 100, 120])
        np.testing.assert_array_equal(result.boxes, before)

    def test_overflow_bbox_clips_to_half_open_width_height(self):
        image, result, faces = inputs(1)
        result.boxes[0] = [10, 20, 400, 300]
        output = adapter.mtcnn_to_facenet_detections(image, result, faces)
        self.assertEqual(output[0]["bbox"], [10, 20, 320, 240])

    def test_empty_bbox_after_clip_has_face_index(self):
        for box in ([400, 300, 500, 400], [-30, -30, -10, -10],
                    [320, 0, 340, 10], [0, 240, 10, 260]):
            image, result, faces = inputs()
            result.boxes[1] = box
            with self.assertRaisesRegex(ValueError, "face_index 1: bbox sau clip"):
                adapter.mtcnn_to_facenet_detections(image, result, faces)

    def test_invalid_original_bbox_is_not_repaired(self):
        for box in ([20, 10, 20, 40], [30, 10, 20, 40], [0, 30, 20, 10]):
            image, result, faces = inputs(1)
            result.boxes[0] = box
            with self.assertRaisesRegex(ValueError, "face_index 0: bbox gốc"):
                adapter.mtcnn_to_facenet_detections(image, result, faces)

    def test_confidence_limits_and_no_detection_threshold(self):
        image, result, faces = inputs()
        result.scores[:] = [0, 1]
        output = adapter.mtcnn_to_facenet_detections(image, result, faces)
        self.assertEqual([r["confidence"] for r in output], [0.0, 1.0])

    def test_out_of_range_confidence_rejected(self):
        for value in (-0.01, 1.01):
            image, result, faces = inputs()
            result.scores[1] = value
            with self.assertRaisesRegex(ValueError, "face_index 1: confidence"):
                adapter.mtcnn_to_facenet_detections(image, result, faces)

    def test_nan_inf_rejected_in_all_metadata(self):
        for name in ("boxes", "scores", "landmarks"):
            for value in (np.nan, np.inf, -np.inf):
                image, result, faces = inputs()
                if name == "scores":
                    result.scores[1] = value
                else:
                    getattr(result, name)[1].flat[0] = value
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, "face_index 1: metadata"):
                        adapter.mtcnn_to_facenet_detections(image, result, faces)

    def test_landmark_order_and_exact_values(self):
        image, result, faces = inputs(1)
        output = adapter.mtcnn_to_facenet_detections(image, result, faces)
        self.assertEqual(tuple(output[0]["landmarks"]), adapter.LANDMARK_ORDER)
        for i, name in enumerate(adapter.LANDMARK_ORDER):
            self.assertEqual(output[0]["landmarks"][name], result.landmarks[0, i].tolist())

    def test_negative_landmarks_omitted_only_for_corresponding_face(self):
        image, result, faces = inputs(3)
        result.landmarks[1, 0, 0] = -0.01
        before = result.landmarks.copy()
        output = adapter.mtcnn_to_facenet_detections(image, result, faces)
        self.assertEqual(len(output), 3)
        self.assertEqual(sum("landmarks" not in r for r in output), 1)
        self.assertNotIn("landmarks", output[1])
        self.assertIn("landmarks", output[0])
        self.assertIn("landmarks", output[2])
        np.testing.assert_array_equal(output[1]["face_rgb"], faces[1])
        np.testing.assert_array_equal(result.landmarks, before)
        validate_mtcnn_output(output)

    def test_negative_landmark_does_not_hide_nan(self):
        image, result, faces = inputs(1)
        result.landmarks[0, 0, 0] = -1
        result.landmarks[0, 1, 0] = np.nan
        with self.assertRaises(ValueError):
            adapter.mtcnn_to_facenet_detections(image, result, faces)

    def test_inputs_unchanged_and_output_memory_independent(self):
        image, result, faces = inputs()
        snapshot = deepcopy((image, result, faces))
        output = adapter.mtcnn_to_facenet_detections(image, result, faces)
        for item, face in zip(output, faces):
            self.assertFalse(np.shares_memory(item["face_rgb"], faces))
            np.testing.assert_array_equal(item["face_rgb"], face)
        output[0]["face_rgb"][:] = 0
        output[0]["bbox"][0] = 999
        output[0]["landmarks"]["nose"][0] = 999
        for actual, expected in ((image, snapshot[0]), (faces, snapshot[2]),
                                  (result.boxes, snapshot[1].boxes),
                                  (result.scores, snapshot[1].scores),
                                  (result.landmarks, snapshot[1].landmarks)):
            np.testing.assert_array_equal(actual, expected)

    def test_noncontiguous_faces_keep_pixels(self):
        image, result, faces = inputs()
        faces = faces[:, :, ::-1, :]
        output = adapter.mtcnn_to_facenet_detections(image, result, faces)
        for i, item in enumerate(output):
            np.testing.assert_array_equal(item["face_rgb"], faces[i])

    def test_full_validation_before_output_and_no_partial_return(self):
        image, result, faces = inputs()
        result.scores[1] = 2
        sentinel = object()
        returned = sentinel
        with patch.object(adapter, "validate_mtcnn_output") as validate:
            with self.assertRaisesRegex(ValueError, "face_index 1"):
                returned = adapter.mtcnn_to_facenet_detections(image, result, faces)
            validate.assert_not_called()
        self.assertIs(returned, sentinel)

    def test_final_validation_uses_real_facenet_contract(self):
        image, result, faces = inputs()
        with patch.object(adapter, "validate_mtcnn_output", wraps=validate_mtcnn_output) as validate:
            output = adapter.mtcnn_to_facenet_detections(image, result, faces)
        validate.assert_called_once_with(output)

    def test_import_and_call_have_no_io_inference_or_sqlite_dependencies(self):
        code = textwrap.dedent("""
            import os, sys
            sys.dont_write_bytecode = True
            def audit(event, args):
                if event == "open":
                    mode, flags = args[1:3]
                    if str(args[0]).lower().endswith((".db", ".onnx", ".pt", ".png", ".jpg", ".ipynb")):
                        raise AssertionError("Asset I/O")
                    if ((isinstance(mode, str) and any(c in mode for c in "wax+")) or
                            (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
                        raise AssertionError("File write")
                if event in ("sqlite3.connect", "socket.connect", "os.mkdir"):
                    raise AssertionError("I/O")
            sys.addaudithook(audit)
        """) + f"\nsys.path[:0] = {[str(ROOT), str(ROOT / 'src')]!r}\n" + textwrap.dedent("""
            import numpy as np
            from types import SimpleNamespace
            from integration.mtcnn_adapter import mtcnn_to_facenet_detections
            image = np.zeros((100,100,3), np.uint8)
            result = SimpleNamespace(boxes=np.array([[0,0,50,50]],np.float32),
                scores=np.array([.9],np.float32), landmarks=np.ones((1,5,2),np.float32))
            output = mtcnn_to_facenet_detections(image,result,np.zeros((1,160,160,3),np.uint8))
            assert len(output)==1
            for name in ("torch","cv2","MTCNN.mtcnn","sqlite3",
                         "FaceNet.facenet_model","FaceNet.database",
                         "FaceNet.recognition","FaceNet.matching"):
                assert name not in sys.modules, name
            print("ADAPTER_NO_IO_OK")
        """)
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8",
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ADAPTER_NO_IO_OK", result.stdout)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "NOT TESTED: thiếu Torch")
class TestRealMTCNN(unittest.TestCase):
    def test_cpu_detection_alignment_and_adapter_only(self):
        from MTCNN.mtcnn import MTCNN
        from MTCNN.preprocessing.image_processing import read_image
        import cv2
        import sqlite3

        # Nguồn ảnh chỉ đọc từ project cũ, không copy dataset/demo output.
        sample_paths = [Path("D:/MTCNN/MTCNN/face.jpg"),
                        Path("D:/MTCNN/aligned_faces/face_1.png")]
        if not all(p.is_file() for p in sample_paths):
            self.skipTest("NOT TESTED: thiếu ảnh mẫu local")
        self.assertTrue(Path(sys.modules["MTCNN.mtcnn"].__file__).resolve().is_relative_to(ROOT))
        detector = MTCNN(device="cpu")
        reports = []
        with patch.object(cv2.dnn, "readNetFromONNX", side_effect=AssertionError("Không load FaceNet")), \
                patch.object(sqlite3, "connect", side_effect=AssertionError("Không mở SQLite")):
            images = [(p.name, read_image(p)) for p in sample_paths]
            images.append(("black_in_RAM", np.zeros((240, 320, 3), np.uint8)))
            for name, image in images:
                before = image.copy()
                start = time.perf_counter()
                result = detector.detect(image)
                detection_seconds = time.perf_counter() - start
                snapshot = deepcopy(result)
                start = time.perf_counter()
                faces = detector.extract_faces_rgb(image, result, size=160, align=True)
                alignment_seconds = time.perf_counter() - start
                start = time.perf_counter()
                output = adapter.mtcnn_to_facenet_detections(image, result, faces)
                mapping_seconds = time.perf_counter() - start
                validate_mtcnn_output(output)
                self.assertEqual(len(output), len(result.boxes))
                self.assertEqual(faces.shape, (len(output), 160, 160, 3))
                if name == "face.jpg":
                    self.assertGreaterEqual(len(output), 2)
                elif name == "face_1.png":
                    self.assertEqual(len(output), 1)
                else:
                    self.assertEqual(output, [])
                for i, item in enumerate(output):
                    np.testing.assert_array_equal(item["face_rgb"], faces[i])
                    self.assertEqual(item["confidence"], float(result.scores[i]))
                np.testing.assert_array_equal(image, before)
                for field in ("boxes", "scores", "landmarks"):
                    np.testing.assert_array_equal(getattr(result, field), getattr(snapshot, field))
                reports.append({"sample": name, "detections": len(result.boxes),
                    "aligned_faces": len(faces), "outputs": len(output),
                    "landmarks_omitted": sum("landmarks" not in r for r in output),
                    "boxes_clipped": sum(not np.array_equal(r["bbox"], result.boxes[i]) for i, r in enumerate(output)),
                    "validation_errors": 0, "detection_seconds": detection_seconds,
                    "alignment_seconds": alignment_seconds, "adapter_seconds": mapping_seconds})
        print("ADAPTER_REAL_MTCNN " + json.dumps(reports, sort_keys=True))

    def test_alignment_failure_propagates_and_adapter_is_not_called(self):
        from MTCNN.mtcnn import MTCNN, DetectionResult
        from MTCNN.preprocessing.face_alignment import TEMPLATE_112
        image = np.ones((300, 300, 3), np.uint8)
        result = DetectionResult(
            np.array([[30,30,220,240], [30,30,220,240]], np.float32),
            np.array([.9,.9],np.float32),
            np.stack([TEMPLATE_112+[60,60], np.zeros((5,2))]).astype(np.float32))
        detector = MTCNN.__new__(MTCNN)
        mapping = Mock(wraps=adapter.mtcnn_to_facenet_detections)
        with self.assertRaises(ValueError):
            faces = detector.extract_faces_rgb(image, result, size=160, align=True)
            mapping(image, result, faces)
        mapping.assert_not_called()


if __name__ == "__main__":
    unittest.main()
