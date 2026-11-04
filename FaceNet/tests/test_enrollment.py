"""Mean/L2 enrollment: unit tests trong RAM và đối chiếu hàm notebook 05/06."""

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import unittest

import cv2
import numpy as np

from _support import (
    PROJECT_ROOT, SOURCE_ROOT, function_body, notebook_function_body,
    notebook_namespace, protected_hashes,
)
from FaceNet import database
from FaceNet.enrollment import representative_from_embeddings
from FaceNet.facenet_model import extract_embeddings, load_facenet_model


def unit(index=0):
    result = np.zeros(512, np.float32)
    result[index] = 1
    return result


class TestEnrollment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.originals = [notebook_namespace(name, ("representative_from_embeddings",))
                         for name in ("05_face_enrollment.ipynb", "06_face_database.ipynb")]

    def assert_error_matches_source(self, embeddings):
        outcomes = []
        for function in [source["representative_from_embeddings"] for source in self.originals] + [representative_from_embeddings]:
            try:
                function(embeddings)
            except (TypeError, ValueError) as error:
                outcomes.append((type(error), str(error)))
        self.assertEqual(len(outcomes), 3, "Hai notebook và module đều phải từ chối")
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0], outcomes[2])

    def test_body_preserves_notebook05_and06(self):
        source = (SOURCE_ROOT / "FaceNet/enrollment.py").read_text(encoding="utf-8")
        body = function_body(source, "representative_from_embeddings")
        for notebook in ("05_face_enrollment.ipynb", "06_face_database.ipynb"):
            self.assertEqual(body, notebook_function_body(notebook, "representative_from_embeddings"))

    def test_three_different_embeddings_shape_dtype_and_l2(self):
        embeddings = np.stack([unit(0), unit(1), unit(2)])
        result = representative_from_embeddings(embeddings)
        expected = np.zeros(512, np.float32); expected[:3] = np.float32(1 / np.sqrt(3))
        np.testing.assert_allclose(result, expected, atol=1e-7, rtol=0)
        self.assertEqual(result.shape, (512,))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(float(np.linalg.norm(result.astype(np.float64))), 1.0, places=6)

    def test_identical_embeddings(self):
        embeddings = np.stack([unit(4)] * 3)
        np.testing.assert_array_equal(representative_from_embeddings(embeddings), unit(4))

    def test_counts_other_than_three_are_supported(self):
        for count in (1, 2, 4, 5, 7):
            with self.subTest(count=count):
                embeddings = np.stack([unit(3)] * count)
                result = representative_from_embeddings(embeddings)
                np.testing.assert_array_equal(result, unit(3))
                for source in self.originals:
                    np.testing.assert_array_equal(result, source["representative_from_embeddings"](embeddings))

    def test_empty_shapes(self):
        for shape in ((0, 512), (0,), (0, 0)):
            with self.subTest(shape=shape):
                self.assert_error_matches_source(np.empty(shape, np.float32))

    def test_unsupported_container_and_mixed_input(self):
        for value in (None, [], [unit(), unit(1)], (unit(), unit(1)),
                      [unit(), np.zeros(511, np.float32)]):
            with self.subTest(type=type(value).__name__):
                self.assert_error_matches_source(value)
        ragged = np.empty(2, dtype=object)
        ragged[0], ragged[1] = unit(), np.zeros(511, np.float32)
        self.assert_error_matches_source(ragged)

    def test_invalid_shapes(self):
        for shape in ((), (512,), (3, 511), (3, 513), (1, 3, 512), (512, 3)):
            with self.subTest(shape=shape):
                self.assert_error_matches_source(np.zeros(shape, np.float32))

    def test_invalid_dtypes(self):
        for dtype in (np.float64, np.float16, np.int32, np.bool_, object):
            with self.subTest(dtype=str(dtype)):
                self.assert_error_matches_source(np.stack([unit()] * 3).astype(dtype))

    def test_nan_and_inf(self):
        for value in (np.nan, np.inf, -np.inf):
            embeddings = np.stack([unit()] * 3)
            embeddings[1, 7] = value
            self.assert_error_matches_source(embeddings)

    def test_invalid_row_norm_and_heterogeneous_rows(self):
        for row in (np.zeros(512, np.float32), unit() * 2, np.ones(512, np.float32)):
            embeddings = np.stack([unit(), row, unit(1)])
            before = embeddings.copy()
            self.assert_error_matches_source(embeddings)
            np.testing.assert_array_equal(embeddings, before)

    def test_zero_mean_is_rejected(self):
        self.assert_error_matches_source(np.stack([unit(), -unit()]))

    def test_near_zero_mean_is_rejected(self):
        opposite = -unit(); opposite[1] = np.float32(1e-12)
        self.assert_error_matches_source(np.stack([unit(), opposite]))

    def test_small_mean_above_original_cutoff_is_accepted(self):
        opposite = -unit(); opposite[1] = np.float32(4e-12)
        embeddings = np.stack([unit(), opposite])
        result = representative_from_embeddings(embeddings)
        np.testing.assert_array_equal(result, unit(1))
        for source in self.originals:
            np.testing.assert_array_equal(result, source["representative_from_embeddings"](embeddings))

    def test_mean_is_float64_not_float32(self):
        # Float32 mất epsilon khi cộng 1 + epsilon - 1; float64 giữ thành phần này.
        middle = unit(1); middle[0] = np.float32(2**-25)
        embeddings = np.stack([unit(), middle, -unit()])
        actual = representative_from_embeddings(embeddings)
        mean64 = np.mean(embeddings, axis=0, dtype=np.float64)
        expected = (mean64 / np.linalg.norm(mean64)).astype(np.float32)
        mean32 = np.mean(embeddings, axis=0, dtype=np.float32)
        wrong = (mean32 / np.linalg.norm(mean32)).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(float(actual[0]), 0.0)
        self.assertFalse(np.array_equal(actual, wrong))

    def test_rows_are_not_normalized_again(self):
        embeddings = np.stack([unit(0) * np.float32(1 + 5e-7), unit(1) * np.float32(1 - 5e-7)])
        mean64 = np.mean(embeddings, axis=0, dtype=np.float64)
        expected = (mean64 / np.linalg.norm(mean64)).astype(np.float32)
        actual = representative_from_embeddings(embeddings)
        np.testing.assert_array_equal(actual, expected)
        renormalized = embeddings.astype(np.float64)
        renormalized /= np.linalg.norm(renormalized, axis=1, keepdims=True)
        wrong_mean = renormalized.mean(axis=0)
        wrong = (wrong_mean / np.linalg.norm(wrong_mean)).astype(np.float32)
        self.assertFalse(np.array_equal(actual, wrong))

    def test_noncontiguous_input_unchanged_and_result_independent(self):
        base = np.stack([unit(0), unit(1), unit(2), unit(3)])
        embeddings = base[::2]
        self.assertFalse(embeddings.flags.c_contiguous)
        before = base.copy()
        result = representative_from_embeddings(embeddings)
        np.testing.assert_array_equal(base, before)
        self.assertFalse(np.shares_memory(result, base))
        result[:] = 0
        np.testing.assert_array_equal(base, before)

    def test_algorithm_and_registry_count_policies_remain_distinct(self):
        for count in (1, 2, 3, 5):
            representative = representative_from_embeddings(np.stack([unit()] * count))
            # Chỉ tạo tuple bản ghi trong RAM, không mở hoặc ghi SQLite.
            database.validate_record_metadata("ram_only", count, "test_v1")
            blob = database.embedding_to_blob(representative)
            rows = [("ram_only", blob, count, "test_v1", hashlib.sha256(blob).hexdigest())]
            if count == 3:
                self.assertEqual(list(database.registered_from_rows(rows, "test_v1")), ["ram_only"])
            else:
                with self.assertRaisesRegex(ValueError, "3 ảnh/người"):
                    database.registered_from_rows(rows, "test_v1")

    def test_fresh_import_has_no_io_or_inference_dependencies(self):
        code = textwrap.dedent('''
            import os, sys
            sys.dont_write_bytecode = True
            def readonly(event, args):
                if event == "open":
                    mode, flags = args[1:3]
                    if str(args[0]).lower().endswith((".db", ".onnx", ".png", ".jpg", ".ipynb")):
                        raise AssertionError("Import accessed asset")
                    if ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                            or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
                        raise AssertionError("Import wrote a file")
                if event in ("sqlite3.connect", "os.mkdir", "socket.__new__"):
                    raise AssertionError("Import performed I/O")
            sys.addaudithook(readonly)
        ''') + f"\nsys.path.insert(0, {str(SOURCE_ROOT)!r})\n" + textwrap.dedent('''
            from FaceNet.enrollment import representative_from_embeddings
            for name in ("cv2", "torch", "sqlite3", "IPython", "MTCNN",
                         "FaceNet.facenet_model", "FaceNet.database"):
                assert name not in sys.modules, name
            print("ENROLLMENT_IMPORT_OK")
        ''')
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT / "notebooks",
                                capture_output=True, text=True, encoding="utf-8",
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ENROLLMENT_IMPORT_OK", result.stdout)


class TestEnrollmentNotebookRegression(unittest.TestCase):
    def test_real_enrollment_embeddings_match_notebook(self):
        before = protected_hashes()
        source05 = notebook_namespace("05_face_enrollment.ipynb", ("representative_from_embeddings",))
        source06 = notebook_namespace("06_face_database.ipynb", ("representative_from_embeddings",))
        manifest = json.loads((PROJECT_ROOT / "data/processed_faces_imm/manifest.json").read_text(encoding="utf-8"))
        grouped = {}
        for record in manifest["records"]:
            if record["status"] == "saved":
                grouped.setdefault(record["person_id"], []).append(record)
        self.assertEqual(len(grouped), 5)
        net = load_facenet_model(SOURCE_ROOT / "FaceNet/models/facenet.onnx")
        maximum_difference = maximum_norm_error = 0.0
        set_count = comparisons = 0
        used_paths = set()
        for person_id, records in sorted(grouped.items()):
            ordered = sorted(records, key=lambda r: r["processed_image"])
            self.assertEqual(len(ordered), 5)
            training, heldout = ordered[:3], ordered[3:]
            faces = []
            for record in training:
                path = PROJECT_ROOT / record["processed_image"]
                bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                self.assertIsNotNone(bgr)
                faces.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                used_paths.add(record["processed_image"])
            self.assertTrue(used_paths.isdisjoint(r["processed_image"] for r in heldout))
            embeddings = extract_embeddings(faces, net)
            # N=5 chỉ lặp embedding enrollment trong RAM, không dùng hai ảnh heldout.
            sets = {1: embeddings[:1], 2: embeddings[:2], 3: embeddings,
                    5: np.concatenate([embeddings, embeddings[:2]], axis=0)}
            for count, matrix in sets.items():
                with self.subTest(person_id=person_id, count=count):
                    snapshot = matrix.copy()
                    actual = representative_from_embeddings(matrix)
                    maximum_norm_error = max(maximum_norm_error, abs(float(np.linalg.norm(actual.astype(np.float64))) - 1.0))
                    for source in (source05, source06):
                        expected = source["representative_from_embeddings"](matrix)
                        difference = float(np.max(np.abs(actual - expected)))
                        maximum_difference = max(maximum_difference, difference)
                        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
                        comparisons += 1
                    np.testing.assert_array_equal(matrix, snapshot)
                    self.assertEqual(actual.shape, (512,))
                    self.assertEqual(actual.dtype, np.float32)
                    self.assertTrue(np.isfinite(actual).all())
                    self.assertLessEqual(abs(float(np.linalg.norm(actual.astype(np.float64))) - 1.0), 1e-6)
                    set_count += 1
        self.assertEqual(len(used_paths), 15)
        self.assertEqual(protected_hashes(), before)
        print("ENROLLMENT_REGRESSION " + json.dumps({"identities": 5, "enrollment_images": 15,
              "heldout_images_used": 0, "embedding_counts": [1, 2, 3, 5], "embedding_sets": set_count,
              "notebook_comparisons": comparisons, "representative_max_abs_difference": maximum_difference,
              "max_L2_norm_error": maximum_norm_error, "atol": 1e-7, "rtol": 1e-6}, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
