"""Kiểm thử CPU và hồi quy giá trị với hàm gốc; không chạy toàn bộ notebook."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from _support import (
    PROJECT_ROOT, SOURCE_ROOT, function_body, notebook_function_body,
    notebook_namespace, protected_hashes,
)
from FaceNet import facenet_model as model


MODEL_PATH = SOURCE_ROOT / "FaceNet/models/facenet.onnx"
ORIGINAL_FUNCTIONS = ("load_facenet_model", "validate_face", "preprocess_face", "extract_embedding", "extract_embeddings")


class TestModelInference(unittest.TestCase):
    def setUp(self):
        self.face = np.arange(160 * 160 * 3, dtype=np.uint8).reshape(160, 160, 3)

    def test_preprocess_formula_layout_and_input_preservation(self):
        before = self.face.copy()
        actual = model.preprocess_face(self.face)
        expected = ((self.face.astype(np.float32) - np.float32(127.5)) / np.float32(128.0)).transpose(2, 0, 1)[None]
        self.assertEqual(actual.shape, (1, 3, 160, 160))
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(actual.flags.c_contiguous)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(self.face, before)

    def test_noncontiguous_input(self):
        face = self.face[:, ::-1, :]
        self.assertFalse(face.flags.c_contiguous)
        actual = model.preprocess_face(face)
        self.assertTrue(actual.flags.c_contiguous)
        np.testing.assert_array_equal(actual[0].transpose(1, 2, 0), (face.astype(np.float32) - 127.5) / 128.0)

    def test_invalid_image_does_not_reach_model(self):
        net = Mock()
        for image in (None, np.zeros((80, 80, 3), np.uint8), self.face.astype(np.float32)):
            with self.subTest(shape=getattr(image, "shape", None)):
                with self.assertRaises((TypeError, ValueError)):
                    model.extract_embedding(image, net)
        net.setInput.assert_not_called()
        net.forward.assert_not_called()

    def test_alias_and_empty_batch(self):
        self.assertIs(model.get_face_embedding, model.extract_embedding)
        net = Mock()
        for value in ([], ()):
            output = model.extract_embeddings(value, net)
            self.assertEqual(output.shape, (0, 512))
            self.assertEqual(output.dtype, np.float32)
        net.setInput.assert_not_called()
        for value in (None, {}, self.face):
            with self.assertRaises(TypeError):
                model.extract_embeddings(value, net)

    def test_batch_order_and_indexed_error(self):
        first = np.zeros(512, np.float32); first[0] = 1
        second = np.zeros(512, np.float32); second[1] = 1
        with patch.object(model, "get_face_embedding", side_effect=[first, second]):
            np.testing.assert_array_equal(model.extract_embeddings([self.face, self.face], None), np.stack([first, second]))
        with patch.object(model, "get_face_embedding", side_effect=[first, ValueError("invalid face")]):
            with self.assertRaisesRegex(ValueError, "chỉ số 1") as error:
                model.extract_embeddings([self.face, self.face], None)
            self.assertIsInstance(error.exception.__cause__, ValueError)

    def test_explicit_absolute_model_path(self):
        with self.assertRaisesRegex(ValueError, "tuyệt đối"):
            model.load_facenet_model("models/facenet.onnx")
        with self.assertRaises(FileNotFoundError):
            model.load_facenet_model(PROJECT_ROOT / "models" / "missing_phase1.onnx")

    def test_loader_sets_cpu_and_reports_opencv_errors(self):
        net = Mock(); net.empty.return_value = False
        with patch.object(Path, "is_file", return_value=True), patch.object(model.cv2.dnn, "readNetFromONNX", return_value=net):
            self.assertIs(model.load_facenet_model(MODEL_PATH), net)
        net.setPreferableBackend.assert_called_once_with(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget.assert_called_once_with(cv2.dnn.DNN_TARGET_CPU)
        with patch.object(Path, "is_file", return_value=True), patch.object(model.cv2.dnn, "readNetFromONNX", side_effect=cv2.error("bad ONNX")):
            with self.assertRaises(RuntimeError) as error:
                model.load_facenet_model(MODEL_PATH)
            self.assertIsInstance(error.exception.__cause__, cv2.error)
        net.empty.return_value = True
        with patch.object(Path, "is_file", return_value=True), patch.object(model.cv2.dnn, "readNetFromONNX", return_value=net):
            with self.assertRaisesRegex(RuntimeError, "rỗng"):
                model.load_facenet_model(MODEL_PATH)

    def test_output_validation_matches_original(self):
        original = notebook_namespace("01_facenet_model.ipynb", ("validate_face", "preprocess_face", "extract_embedding"))
        cases = [[], np.zeros(512, np.float32), np.zeros((2, 512), np.float32),
                 np.zeros((1, 256), np.float32), np.ones((1, 512), np.float64),
                 np.full((1, 512), np.nan, np.float32), np.full((1, 512), np.inf, np.float32),
                 np.zeros((1, 512), np.float32), np.full((1, 512), 1e-20, np.float32)]
        for index, output in enumerate(cases):
            outcomes = []
            for fn in (original["extract_embedding"], model.extract_embedding):
                net = Mock(); net.forward.return_value = output
                try:
                    fn(self.face, net)
                except (TypeError, ValueError, RuntimeError) as error:
                    outcomes.append((type(error), str(error)))
            with self.subTest(case=index):
                self.assertEqual(len(outcomes), 2)
                self.assertEqual(outcomes[0], outcomes[1])

    def test_inference_error_is_chained(self):
        for operation in ("setInput", "forward"):
            net = Mock(); getattr(net, operation).side_effect = cv2.error("inference failed")
            with self.subTest(operation=operation), self.assertRaises(RuntimeError) as error:
                model.extract_embedding(self.face, net)
            self.assertIsInstance(error.exception.__cause__, cv2.error)

    def test_numeric_bodies_match_notebooks(self):
        source = (SOURCE_ROOT / "FaceNet" / "facenet_model.py").read_text(encoding="utf-8")
        for name, notebook in [("preprocess_face", "01_facenet_model.ipynb"),
                               ("extract_embedding", "01_facenet_model.ipynb"),
                               ("extract_embeddings", "02_embedding_extraction.ipynb")]:
            with self.subTest(function=name):
                self.assertEqual(function_body(source, name), notebook_function_body(notebook, name))


class TestImportAndWorkingDirectory(unittest.TestCase):
    def run_child(self, body, cwd):
        # Chỉ tiến trình kiểm thử cấu hình src; không cài package hoặc sửa môi trường.
        guard = textwrap.dedent('''
            import os, sys
            sys.dont_write_bytecode = True
            def readonly(event, args):
                if event == "open":
                    mode, flags = args[1:3]
                    if ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                            or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
                        raise PermissionError("Read-only test")
                if event in ("os.mkdir", "os.remove", "os.rmdir", "os.rename", "socket.__new__", "subprocess.Popen"):
                    raise PermissionError("Read-only test")
            sys.addaudithook(readonly)
        ''')
        code = guard + f"\nsys.path.insert(0, {str(SOURCE_ROOT)!r})\n" + textwrap.dedent(body)
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=cwd,
                                capture_output=True, text=True, encoding="utf-8",
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_fresh_import_has_no_model_or_data_access(self):
        output = self.run_child('''
            def reject_assets(event, args):
                if event == "open" and str(args[0]).lower().endswith((".onnx", ".db", ".png", ".ipynb")):
                    raise AssertionError("Import attempted asset access")
            sys.addaudithook(reject_assets)
            import FaceNet
            assert "cv2" not in sys.modules and "numpy" not in sys.modules
            import cv2
            from unittest.mock import patch
            with patch.object(cv2.dnn, "readNetFromONNX", side_effect=AssertionError("Import loaded model")) as loader:
                from FaceNet import contracts, facenet_model
                loader.assert_not_called()
            assert "torch" not in sys.modules
            assert "sqlite3" not in sys.modules
            assert "IPython" not in sys.modules
            print("IMPORT_OK")
        ''', PROJECT_ROOT / "notebooks")
        self.assertIn("IMPORT_OK", output)

    def test_absolute_model_load_from_two_working_directories(self):
        code = f'''
            from FaceNet.facenet_model import load_facenet_model
            net = load_facenet_model({str(MODEL_PATH)!r})
            assert not net.empty()
            print("LOAD_OK")
        '''
        for cwd in (PROJECT_ROOT, PROJECT_ROOT / "notebooks"):
            with self.subTest(cwd=str(cwd)):
                self.assertIn("LOAD_OK", self.run_child(code, cwd))


class TestNotebookRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hashes_before = protected_hashes()
        cls.original = notebook_namespace("02_embedding_extraction.ipynb", ORIGINAL_FUNCTIONS, ("get_face_embedding",))
        cls.original_net = cls.original["load_facenet_model"](MODEL_PATH)
        cls.module_net = model.load_facenet_model(MODEL_PATH)

    @classmethod
    def tearDownClass(cls):
        if protected_hashes() != cls.hashes_before:
            raise AssertionError("Asset hoặc tài liệu gốc bị thay đổi trong kiểm thử")

    def test_real_images_match_original_values(self):
        paths = sorted((PROJECT_ROOT / "data" / "processed_faces_imm").rglob("*.png"))
        paths += sorted((PROJECT_ROOT / "data" / "processed_unknown_faces_imm").rglob("*.png"))
        self.assertEqual(len(paths), 71)
        max_preprocess = max_embedding = max_norm_error = 0.0
        for path in paths:
            with self.subTest(image=str(path.relative_to(PROJECT_ROOT))):
                bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                self.assertIsNotNone(bgr)
                face = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                before = face.copy()
                expected_pre = self.original["preprocess_face"](face)
                actual_pre = model.preprocess_face(face)
                np.testing.assert_allclose(actual_pre, expected_pre, atol=0, rtol=0)
                expected = self.original["extract_embedding"](face, self.original_net)
                actual = model.extract_embedding(face, self.module_net)
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
                self.assertEqual(actual.shape, (512,))
                self.assertEqual(actual.dtype, np.float32)
                self.assertTrue(np.isfinite(actual).all())
                norm_error = abs(float(np.linalg.norm(actual.astype(np.float64))) - 1.0)
                self.assertLessEqual(norm_error, 1e-6)
                max_preprocess = max(max_preprocess, float(np.max(np.abs(actual_pre - expected_pre))))
                max_embedding = max(max_embedding, float(np.max(np.abs(actual - expected))))
                max_norm_error = max(max_norm_error, norm_error)
                np.testing.assert_array_equal(face, before)
        print("REGRESSION " + json.dumps({"images": len(paths), "preprocessing_max_abs_difference": max_preprocess,
              "embedding_max_abs_difference": max_embedding, "max_L2_norm_error": max_norm_error,
              "embedding_atol": 1e-6, "embedding_rtol": 1e-5}, sort_keys=True))

    def test_real_batch_order_and_repeat(self):
        paths = sorted((PROJECT_ROOT / "data" / "processed_faces_imm").rglob("*.png"))[:3]
        faces = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]
        expected = self.original["extract_embeddings"](faces, self.original_net)
        actual = model.extract_embeddings(tuple(faces), self.module_net)
        self.assertEqual(actual.shape, (3, 512))
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
        repeated = model.extract_embeddings(faces, self.module_net)
        np.testing.assert_allclose(repeated, actual, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
