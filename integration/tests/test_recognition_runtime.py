
from contextlib import contextmanager
import hashlib
import inspect
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from integration import recognition_runtime as runtime
from integration import recognition_pipeline as pipeline
from integration.enrollment_pipeline import build_model_version, initialize_enrollment_registry
from FaceNet import database as db, matching, recognition as core
from FaceNet import facenet_model as model


class TestRuntimeSelection(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((160, 160, 3), np.uint8)
        self.detector = object()
        self.net = object()
        self.path = Path(tempfile.gettempdir()) / "not_opened.sqlite"
        self.version = "caller-owned-profile"
        self.args = (self.image, self.detector, self.net, self.path, self.version)

    def forwarded(self, **kwargs):
        output = [{"status": "unknown", "person_id": None}]
        with patch.object(pipeline, "recognize_faces_from_image", return_value=output) as call:
            result = runtime.recognize_image(*self.args, **kwargs)
        call.assert_called_once()
        values = call.call_args.kwargs
        self.assertEqual(set(values), {"image_rgb", "detector", "facenet_net", "db_path", "model_version", "threshold"})
        for name, expected in zip(("image_rgb", "detector", "facenet_net", "db_path", "model_version"), self.args):
            self.assertIs(values[name], expected)
        self.assertIs(result, output)
        return values["threshold"]

    def test_default_V2_and_forwarded_objects(self):
        self.assertEqual(self.forwarded().hex(), "0x1.54be2dd42d71dp-1")

    def test_explicit_V1_rollback(self):
        self.assertEqual(self.forwarded(threshold_profile="v1").hex(), "0x1.d9df8d3362a3cp-1")

    def test_explicit_threshold_override(self):
        self.assertEqual(self.forwarded(threshold=.8), .8)

    def test_override_takes_precedence_over_valid_profile(self):
        self.assertEqual(self.forwarded(threshold_profile="v1", threshold=.7), .7)

    def test_exact_binary64_configuration(self):
        self.assertEqual(runtime.V2_CONTROLLED_THRESHOLD, 0.665513450766159)
        self.assertEqual(runtime.V1_ROLLBACK_THRESHOLD, 0.925533688089963)
        self.assertEqual(runtime.V2_CONTROLLED_THRESHOLD.hex(), "0x1.54be2dd42d71dp-1")
        self.assertEqual(runtime.V1_ROLLBACK_THRESHOLD.hex(), "0x1.d9df8d3362a3cp-1")
        self.assertEqual(runtime.DEFAULT_THRESHOLD_PROFILE, "v2")

    def test_numeric_override_domain_and_numpy_scalars(self):
        for value in (-1, 0, 1, np.float32(.5), np.float64(.8), np.int64(0)):
            with self.subTest(value=value):
                actual = self.forwarded(threshold=value)
                self.assertIs(type(actual), float)
                self.assertEqual(actual, matching.validate_threshold(value))

    def test_explicit_None_is_invalid_not_default(self):
        with patch.object(pipeline, "recognize_faces_from_image") as call:
            with self.assertRaises(TypeError):
                runtime.recognize_image(*self.args, threshold=None)
            call.assert_not_called()

    def test_invalid_override_types_never_call_pipeline(self):
        for value in (True, False, np.bool_(True), ".8", [], object()):
            with self.subTest(value=type(value).__name__), patch.object(pipeline, "recognize_faces_from_image") as call:
                with self.assertRaises(TypeError):
                    runtime.recognize_image(*self.args, threshold=value)
                call.assert_not_called()

    def test_invalid_override_values_never_call_pipeline(self):
        for value in (math.nan, math.inf, -math.inf, -1.01, 1.01):
            with self.subTest(value=value), patch.object(pipeline, "recognize_faces_from_image") as call:
                with self.assertRaises(ValueError):
                    runtime.recognize_image(*self.args, threshold=value)
                call.assert_not_called()

    def test_invalid_profile_types(self):
        for value in (None, True, 1, []):
            with self.subTest(value=value), patch.object(pipeline, "recognize_faces_from_image") as call:
                with self.assertRaises(TypeError):
                    runtime.recognize_image(*self.args, threshold_profile=value)
                call.assert_not_called()

    def test_invalid_profiles_even_with_override(self):
        for value in ("", "V1", "v3", " v2"):
            with self.subTest(value=value), patch.object(pipeline, "recognize_faces_from_image") as call:
                with self.assertRaises(ValueError):
                    runtime.recognize_image(*self.args, threshold_profile=value, threshold=.8)
                call.assert_not_called()

    def test_pipeline_exception_propagates_unchanged(self):
        error = RuntimeError("caller registry/model failure")
        with patch.object(pipeline, "recognize_faces_from_image", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                runtime.recognize_image(*self.args)
        self.assertIs(caught.exception, error)

    def test_model_reuse_without_loading(self):
        with patch.object(model, "load_facenet_model", side_effect=AssertionError("Unexpected model load")) as loader:
            self.forwarded()
            self.forwarded(threshold_profile="v1")
        loader.assert_not_called()

    def test_existing_public_apis_still_require_threshold(self):
        for fn in (pipeline.recognize_faces_from_image, core.recognize_aligned_face,
                   core.recognize_detected_faces, matching.detect_unknown):
            self.assertIs(inspect.signature(fn).parameters["threshold"].default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            inspect.signature(pipeline.recognize_faces_from_image).bind(*self.args)
        inspect.signature(runtime.recognize_image).bind(*self.args)

    def test_rollback_does_not_change_next_call_default(self):
        self.forwarded(threshold_profile="v1")
        self.assertEqual(self.forwarded().hex(), "0x1.54be2dd42d71dp-1")

    def test_fresh_import_has_no_business_IO_or_model_load(self):
        code = textwrap.dedent("""
            import sys,os
            sys.dont_write_bytecode=True
            def guard(event,args):
                if event=="open":
                    mode,flags=args[1:3]
                    if str(args[0]).lower().endswith((".onnx",".pt",".db",".sqlite",".png",".jpg",".ipynb")):
                        raise AssertionError("Import accessed business assets")
                    if ((isinstance(mode,str) and any(c in mode for c in "wax+")) or
                        (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))):
                        raise AssertionError("Import wrote a file")
                if event in ("sqlite3.connect","os.mkdir","socket.connect"):
                    raise AssertionError("Unexpected import I/O")
            sys.addaudithook(guard)
        """) + f"\nsys.path[:0]={[str(ROOT)]!r}\n" + textwrap.dedent("""
            from integration.recognition_runtime import recognize_image
            assert "torch" not in sys.modules
            assert "MTCNN.mtcnn" not in sys.modules
            print("RUNTIME_IMPORT_OK")
        """)
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(tempfile.gettempdir()),
                                capture_output=True, text=True, encoding="utf-8",
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("RUNTIME_IMPORT_OK", result.stdout)


def unit():
    vector = np.zeros(512, np.float32)
    vector[0] = 1
    return vector


def fake_detector():
    detected = SimpleNamespace(
        boxes=np.array([[0, 0, 160, 160]], np.float32),
        scores=np.array([.99], np.float32),
        landmarks=np.array([[[30, 40], [60, 40], [45, 55], [35, 80], [55, 80]]], np.float32),
    )
    return SimpleNamespace(
        min_face_size=20, factor=.709, thresholds=(.6, .7, .7), batch_size=256, device="cpu",
        detect=Mock(return_value=detected),
        extract_faces_rgb=Mock(return_value=np.zeros((1, 160, 160, 3), np.uint8)),
    )


class TestRuntimePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.version = build_model_version(fake_detector())

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="runtime_wrapper_", dir=Path(tempfile.gettempdir()))
        self.folder = Path(self.temporary.name).resolve()
        self.addCleanup(self.safe_cleanup)
        self.path = self.folder / "registry.sqlite"
        initialize_enrollment_registry(self.path)
        db.add_person(self.path, "unit_person", unit(), 3, self.version)
        self.image = np.zeros((160, 160, 3), np.uint8)
        self.detector = fake_detector()
        self.net = object()
        self.query = unit()
        self.query[0], self.query[1] = np.float32(.8), np.float32(.6)
        self.args = (self.image, self.detector, self.net, self.path, self.version)

    def safe_cleanup(self):
        if not self.folder.is_relative_to((Path(tempfile.gettempdir())).resolve()) or not self.folder.name.startswith("runtime_wrapper_"):
            raise AssertionError("Unsafe temporary registry cleanup")
        self.temporary.cleanup()
        self.assertFalse(self.folder.exists())

    @contextmanager
    def synthetic_embedding(self):
        with patch.object(core, "extract_embedding", return_value=self.query), \
             patch.object(pipeline, "extract_embedding", return_value=self.query):
            yield

    def test_same_threshold_has_identical_pipeline_results(self):
        for threshold in (-1., .8, runtime.V1_ROLLBACK_THRESHOLD, runtime.V2_CONTROLLED_THRESHOLD, 1.):
            with self.subTest(threshold=threshold), self.synthetic_embedding():
                expected = pipeline.recognize_faces_from_image(*self.args, threshold)
                actual = runtime.recognize_image(*self.args, threshold=threshold)
                self.assertEqual(actual, expected)

    def test_default_accepts_and_V1_rolls_back_to_rejection(self):
        with self.synthetic_embedding():
            current = runtime.recognize_image(*self.args)[0]
            rollback = runtime.recognize_image(*self.args, threshold_profile="v1")[0]
        self.assertEqual(current["status"], "recognized")
        self.assertEqual(rollback["status"], "unknown")
        self.assertEqual(current["similarity"], rollback["similarity"])
        self.assertEqual(current["best_match_id"], rollback["best_match_id"])

    def test_invalid_threshold_precedes_detector_and_registry_IO(self):
        with patch.object(db, "connect_database", side_effect=AssertionError("Unexpected registry access")):
            with self.assertRaises(ValueError):
                runtime.recognize_image(*self.args, threshold=math.nan)
        self.detector.detect.assert_not_called()

    def test_missing_registry_error_is_not_converted_to_unknown(self):
        args = (self.image, self.detector, self.net, self.folder / "missing.sqlite", self.version)
        with self.assertRaises(FileNotFoundError):
            runtime.recognize_image(*args)
        self.detector.detect.assert_not_called()

    def test_no_face_preserves_empty_result(self):
        self.detector.detect.return_value = SimpleNamespace(
            boxes=np.empty((0, 4), np.float32), scores=np.empty(0, np.float32),
            landmarks=np.empty((0, 5, 2), np.float32))
        self.detector.extract_faces_rgb.return_value = np.empty((0, 160, 160, 3), np.uint8)
        with patch.object(core, "extract_embedding", side_effect=AssertionError("No faces to embed")):
            self.assertEqual(runtime.recognize_image(*self.args), [])

    def test_existing_greater_equal_boundary_preserved(self):
        score = matching.cosine_similarity(self.query, unit())
        for threshold, accepted in ((math.nextafter(score, -math.inf), True),
                                    (score, True), (math.nextafter(score, math.inf), False)):
            with self.subTest(threshold=threshold), self.synthetic_embedding():
                row = runtime.recognize_image(*self.args, threshold=threshold)[0]
                self.assertEqual(row["status"] == "recognized", accepted)

    def test_registry_and_input_unchanged(self):
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        children = sorted(p.name for p in self.folder.iterdir())
        image_before = self.image.copy()
        with self.synthetic_embedding():
            runtime.recognize_image(*self.args)
            runtime.recognize_image(*self.args, threshold_profile="v1")
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), children)
        np.testing.assert_array_equal(self.image, image_before)


if __name__ == "__main__":
    unittest.main()
