"""Matching thuần và hồi quy notebook 07–10; SQLite chỉ đọc, không enrollment."""

from contextlib import closing
import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
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
from FaceNet import matching
from FaceNet.facenet_model import extract_embedding, load_facenet_model


COMMON_FUNCTIONS = ("validate_person_id", "validate_database_embedding", "cosine_similarity", "match_embedding")
ALL_FUNCTIONS = COMMON_FUNCTIONS + ("validate_threshold", "detect_unknown")
NOTEBOOKS = (
    "07_face_matching.ipynb", "08_unknown_detection.ipynb",
    "09_open_set_evaluation.ipynb", "10_mtcnn_facenet_interface.ipynb",
)


def unit_vector(index=0):
    vector = np.zeros(512, np.float32)
    vector[index] = 1
    return vector


class TestMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = notebook_namespace(NOTEBOOKS[-1], ALL_FUNCTIONS, ("EMBEDDING_SIZE",))

    def setUp(self):
        self.query = unit_vector()
        self.registry = {"same": self.query.copy(), "orthogonal": unit_vector(1), "opposite": -self.query}

    def assert_same_error(self, name, *args):
        outcomes = []
        for fn in (self.original[name], getattr(matching, name)):
            try:
                fn(*args)
            except (TypeError, ValueError) as error:
                outcomes.append((type(error), str(error),
                                 type(error.__cause__) if error.__cause__ else None))
        self.assertEqual(len(outcomes), 2, "Cả hai phiên bản phải từ chối input")
        self.assertEqual(outcomes[0], outcomes[1])

    def test_bodies_preserve_all_notebook_versions(self):
        source = (SOURCE_ROOT / "FaceNet" / "matching.py").read_text(encoding="utf-8")
        for notebook in NOTEBOOKS:
            names = COMMON_FUNCTIONS if notebook.startswith("07") else ALL_FUNCTIONS
            for name in names:
                with self.subTest(notebook=notebook, function=name):
                    self.assertEqual(function_body(source, name), notebook_function_body(notebook, name))

    def test_identical_orthogonal_and_opposite_cosine(self):
        self.assertEqual(matching.cosine_similarity(self.query, self.query), 1.0)
        self.assertEqual(matching.cosine_similarity(self.query, unit_vector(1)), 0.0)
        self.assertEqual(matching.cosine_similarity(self.query, -self.query), -1.0)
        mixed = (self.query + unit_vector(1)) / np.float32(np.sqrt(2))
        self.assertAlmostEqual(matching.cosine_similarity(self.query, mixed), 1 / np.sqrt(2), places=12)

    def test_near_unit_norm_is_not_mutated(self):
        vector = self.query * np.float32(1 + 5e-7)
        before = vector.copy()
        matching.validate_database_embedding(vector)
        self.assertEqual(matching.cosine_similarity(vector, self.query), 1.0)
        np.testing.assert_array_equal(vector, before)

    def test_nonunit_embedding_is_rejected_not_normalized(self):
        vector = self.query * 2
        before = vector.copy()
        self.assert_same_error("validate_database_embedding", vector)
        np.testing.assert_array_equal(vector, before)

    def test_ranking_order_schema_and_all_candidates(self):
        result = matching.match_embedding(self.query, self.registry)
        self.assertEqual([r["person_id"] for r in result], ["same", "orthogonal", "opposite"])
        self.assertEqual([r["rank"] for r in result], [1, 2, 3])
        self.assertEqual([r["similarity"] for r in result], [1.0, 0.0, -1.0])
        self.assertTrue(all(set(r) == {"person_id", "similarity", "rank"} for r in result))

    def test_ties_use_person_id_not_insertion_order(self):
        for ids in (("z", "a", "m"), ("m", "z", "a")):
            registry = {name: self.query.copy() for name in ids}
            result = matching.match_embedding(self.query, registry)
            self.assertEqual([r["person_id"] for r in result], ["a", "m", "z"])

    def test_known_result_fields(self):
        result = matching.detect_unknown(self.query, self.registry, 0.48)
        self.assertEqual(set(result), {"predicted_id", "best_match_id", "similarity", "threshold", "is_known", "ranking"})
        self.assertEqual(result["predicted_id"], "same")
        self.assertEqual(result["best_match_id"], "same")
        self.assertIs(result["is_known"], True)
        self.assertEqual(result["threshold"], 0.48)

    def test_unknown_preserves_best_candidate(self):
        result = matching.detect_unknown(self.query, {"nearest": unit_vector(1)}, 0.48)
        self.assertEqual(result["predicted_id"], "Unknown")
        self.assertEqual(result["best_match_id"], "nearest")
        self.assertIs(result["is_known"], False)
        self.assertEqual(result["similarity"], 0.0)
        self.assertEqual(result["ranking"][0]["person_id"], "nearest")

    def test_equality_and_adjacent_thresholds(self):
        registry = {"candidate": unit_vector(1)}
        for threshold, expected in ((0.0, True), (np.nextafter(0.0, -np.inf), True),
                                    (np.nextafter(0.0, np.inf), False)):
            result = matching.detect_unknown(self.query, registry, threshold)
            self.assertIs(result["is_known"], expected)
            self.assertEqual(result, self.original["detect_unknown"](self.query, registry, threshold))
        self.assertIs(matching.detect_unknown(self.query, {"same": self.query}, 1.0)["is_known"], True)

    def test_no_default_threshold_or_calibration(self):
        signature = inspect.signature(matching.detect_unknown)
        self.assertIs(signature.parameters["threshold"].default, inspect.Parameter.empty)
        for threshold in (-1, 0, 1, np.float32(.5), np.float64(.48)):
            self.assertEqual(matching.validate_threshold(threshold), self.original["validate_threshold"](threshold))

    def test_empty_registry_keeps_original_error(self):
        self.assert_same_error("match_embedding", self.query, {})
        self.assert_same_error("detect_unknown", self.query, {}, 0.48)

    def test_invalid_registry_types(self):
        for registry in (None, [], (), [("a", self.query)]):
            with self.subTest(registry=type(registry).__name__):
                self.assert_same_error("match_embedding", self.query, registry)

    def test_invalid_embeddings(self):
        invalid = [None, [], np.empty(0, np.float32), np.zeros((1, 512), np.float32),
                   np.zeros(511, np.float32), np.zeros(513, np.float32),
                   self.query.astype(np.float64), self.query.astype(np.int32),
                   np.zeros(512, np.float32), np.full(512, np.nan, np.float32),
                   np.full(512, np.inf, np.float32), np.ones(512, np.float32)]
        for index, value in enumerate(invalid):
            with self.subTest(case=index):
                self.assert_same_error("cosine_similarity", value, self.query)
                self.assert_same_error("cosine_similarity", self.query, value)
                self.assert_same_error("match_embedding", value, self.registry)
                self.assert_same_error("match_embedding", self.query, {"bad": value})

    def test_candidate_error_keeps_id_and_cause(self):
        with self.assertRaisesRegex(TypeError, "Embedding đăng ký của bad") as error:
            matching.match_embedding(self.query, {"bad": self.query.astype(np.float64)})
        self.assertIsInstance(error.exception.__cause__, TypeError)

    def test_invalid_ids(self):
        for person_id in (None, 1, "", " ", " leading", "trailing "):
            with self.subTest(person_id=person_id):
                self.assert_same_error("match_embedding", self.query, {person_id: self.query})

    def test_reserved_unknown_behavior_is_unchanged(self):
        # Validator/ranking cũ cho phép tên này; decision mới là nơi báo lỗi.
        matching.validate_person_id("Unknown")
        self.assertEqual(matching.match_embedding(self.query, {"Unknown": self.query})[0]["person_id"], "Unknown")
        self.assert_same_error("detect_unknown", self.query, {"Unknown": self.query}, 0.48)

    def test_invalid_thresholds(self):
        for threshold in (None, True, np.bool_(False), "0.48", [], np.nan, np.inf, -np.inf, -1.01, 1.01):
            with self.subTest(threshold=str(threshold)):
                self.assert_same_error("validate_threshold", threshold)
                self.assert_same_error("detect_unknown", self.query, self.registry, threshold)

    def test_validation_order_is_unchanged(self):
        self.assert_same_error("detect_unknown", None, {}, "invalid")
        self.assert_same_error("match_embedding", None, {})

    def test_inputs_remain_unchanged(self):
        before = {key: value.copy() for key, value in self.registry.items()}
        query_before = self.query.copy()
        ids_before = list(self.registry)
        matching.match_embedding(self.query, self.registry)
        matching.detect_unknown(self.query, self.registry, .5)
        np.testing.assert_array_equal(self.query, query_before)
        self.assertEqual(list(self.registry), ids_before)
        for key, vector in before.items():
            np.testing.assert_array_equal(self.registry[key], vector)

    def test_fresh_import_has_no_business_io_or_heavy_dependencies(self):
        code = textwrap.dedent('''
            import os, sys
            sys.dont_write_bytecode = True
            def readonly(event, args):
                if event == "open":
                    mode, flags = args[1:3]
                    if str(args[0]).lower().endswith((".onnx", ".db", ".png", ".jpg", ".ipynb")):
                        raise AssertionError("Import accessed asset")
                    if ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                            or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
                        raise PermissionError("Read-only import test")
                if event in ("os.mkdir", "os.remove", "os.rmdir", "os.rename", "socket.__new__", "sqlite3.connect"):
                    raise AssertionError("Import performed I/O")
            sys.addaudithook(readonly)
        ''') + f"\nsys.path.insert(0, {str(SOURCE_ROOT)!r})\n" + textwrap.dedent('''
            from FaceNet.matching import cosine_similarity, match_embedding, detect_unknown
            for name in ("cv2", "torch", "sqlite3", "IPython", "MTCNN", "FaceNet.facenet_model"):
                assert name not in sys.modules, name
            print("MATCHING_IMPORT_OK")
        ''')
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT / "notebooks",
                                capture_output=True, text=True, encoding="utf-8",
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("MATCHING_IMPORT_OK", result.stdout)


class TestMatchingNotebookRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hashes_before = protected_hashes()
        cls.originals = {
            notebook: notebook_namespace(notebook,
                COMMON_FUNCTIONS if notebook.startswith("07") else ALL_FUNCTIONS,
                ("EMBEDDING_SIZE",))
            for notebook in NOTEBOOKS
        }
        cls.db_path = PROJECT_ROOT / "data" / "face_database_test.db"
        cls.db_hash = hashlib.sha256(cls.db_path.read_bytes()).hexdigest()
        with closing(sqlite3.connect(cls.db_path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as connection:
            connection.execute("PRAGMA query_only=ON")
            if connection.execute("PRAGMA query_only").fetchone() != (1,):
                raise AssertionError("SQLite phải chỉ đọc")
            rows = connection.execute(
                "SELECT person_id, embedding, num_images, model_version, embedding_sha256 "
                "FROM FaceDatabase ORDER BY person_id"
            ).fetchall()
        cls.registry = {}
        versions = set()
        for person_id, blob, num_images, version, checksum in rows:
            if len(blob) != 2048 or hashlib.sha256(blob).hexdigest() != checksum or num_images != 3:
                raise AssertionError("Bản ghi baseline không hợp lệ")
            vector = np.frombuffer(blob, dtype="<f4").astype(np.float32, copy=True)
            matching.validate_database_embedding(vector)
            cls.registry[person_id] = vector
            versions.add(version)
        if len(cls.registry) != 5 or len(versions) != 1:
            raise AssertionError("Baseline phải có 5 danh tính cùng version")

        known = json.loads((PROJECT_ROOT / "data/processed_faces_imm/manifest.json").read_text(encoding="utf-8"))
        grouped = {}
        for record in known["records"]:
            if record["status"] == "saved":
                grouped.setdefault(record["person_id"], []).append(record)
        if set(grouped) != set(cls.registry):
            raise AssertionError("Danh tính Known không khớp SQLite")
        cls.records = []
        for person_id, records in sorted(grouped.items()):
            ordered = sorted(records, key=lambda r: r["processed_image"])
            if len(ordered) != 5:
                raise AssertionError("Split baseline phải có 5 ảnh/người")
            # Chỉ dùng hai query cũ; không trích xuất hoặc cập nhật enrollment.
            cls.records.extend((person_id, record) for record in ordered[3:])
        unknown = json.loads((PROJECT_ROOT / "data/processed_unknown_faces_imm/manifest.json").read_text(encoding="utf-8"))
        cls.records.extend(("Unknown", record) for record in sorted(
            (r for r in unknown["records"] if r["status"] == "saved" and r["split"] == "evaluation"),
            key=lambda r: (r["person_id"], r["processed_image"])))
        if len(cls.records) != 32:
            raise AssertionError("Hồi quy cần đúng 10 Known + 22 Unknown evaluation")
        cls.net = load_facenet_model(SOURCE_ROOT / "FaceNet/models/facenet.onnx")

    @classmethod
    def tearDownClass(cls):
        if protected_hashes() != cls.hashes_before:
            raise AssertionError("Asset/tài liệu gốc thay đổi trong hồi quy matching")
        if hashlib.sha256(cls.db_path.read_bytes()).hexdigest() != cls.db_hash:
            raise AssertionError("SQLite đã bị thay đổi")

    def test_real_queries_match_notebook_cosine_rank_and_decisions(self):
        stats = {
            "queries": len(self.records), "known_queries": 10, "unknown_evaluation_queries": 22,
            "candidates_per_query": len(self.registry), "unique_query_candidate_pairs": 160,
            "cosine_comparisons": 0, "ranking_comparisons": 0, "decision_comparisons": 0,
            "fixed_threshold_decision_comparisons": 0, "boundary_decision_comparisons": 0,
            "similarity_max_abs_difference": 0.0, "ranking_mismatches": 0,
            "best_match_mismatches": 0, "known_unknown_mismatches": 0,
            "predicted_id_mismatches": 0, "thresholds": [0.48, 0.5],
        }
        query_input_hashes = []
        registry_before = {key: vector.copy() for key, vector in self.registry.items()}
        for actual_id, record in self.records:
            path = PROJECT_ROOT / record["processed_image"]
            bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(bgr)
            face_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            query = extract_embedding(face_rgb, self.net)
            query_before = query.copy()
            query_input_hashes.append(hashlib.sha256(face_rgb.tobytes()).hexdigest())
            actual_ranking = matching.match_embedding(query, self.registry)
            for notebook, source in self.originals.items():
                with self.subTest(image=path.name, notebook=notebook):
                    for candidate in self.registry.values():
                        expected_score = source["cosine_similarity"](query, candidate)
                        actual_score = matching.cosine_similarity(query, candidate)
                        difference = abs(actual_score - expected_score)
                        stats["similarity_max_abs_difference"] = max(stats["similarity_max_abs_difference"], difference)
                        stats["cosine_comparisons"] += 1
                        np.testing.assert_allclose(actual_score, expected_score, atol=1e-12, rtol=0)
                    expected_ranking = source["match_embedding"](query, self.registry)
                    mismatch = actual_ranking != expected_ranking
                    stats["ranking_comparisons"] += 1
                    stats["ranking_mismatches"] += int(mismatch)
                    self.assertEqual(actual_ranking, expected_ranking)
                    if notebook.startswith("07"):
                        continue
                    # Score làm threshold ở ca biên chỉ kiểm tra toán tử >=,
                    # không chọn ngưỡng vận hành hoặc calibration từ evaluation.
                    for threshold in (0.48, 0.5, actual_ranking[0]["similarity"]):
                        expected = source["detect_unknown"](query, self.registry, threshold)
                        actual = matching.detect_unknown(query, self.registry, threshold)
                        stats["decision_comparisons"] += 1
                        if threshold in (0.48, 0.5):
                            stats["fixed_threshold_decision_comparisons"] += 1
                        else:
                            stats["boundary_decision_comparisons"] += 1
                            self.assertIs(actual["is_known"], True)
                        stats["best_match_mismatches"] += int(actual["best_match_id"] != expected["best_match_id"])
                        stats["known_unknown_mismatches"] += int(actual["is_known"] != expected["is_known"])
                        stats["predicted_id_mismatches"] += int(actual["predicted_id"] != expected["predicted_id"])
                        self.assertEqual(actual, expected)
            np.testing.assert_array_equal(query, query_before)
        for key, vector in registry_before.items():
            np.testing.assert_array_equal(self.registry[key], vector)
        self.assertEqual(len(set(query_input_hashes)), 32)
        print("MATCHING_REGRESSION " + json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
