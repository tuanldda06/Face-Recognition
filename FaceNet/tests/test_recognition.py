"""Orchestration notebook 10: kiểm tra hợp đồng và hồi quy ONNX/SQLite chỉ đọc."""

from contextlib import closing
from copy import deepcopy
import hashlib
import json
import os
import sqlite3
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
from FaceNet import database, recognition
from FaceNet.facenet_model import load_facenet_model


NOTEBOOK = "10_mtcnn_facenet_interface.ipynb"
FUNCTIONS = (
    "validate_face", "validate_aligned_face", "validate_coordinate_sequence",
    "validate_detection_metadata", "validate_mtcnn_output", "preprocess_face",
    "extract_embedding", "validate_person_id", "validate_database_embedding",
    "cosine_similarity", "match_embedding", "validate_threshold", "detect_unknown",
    "recognize_aligned_face", "recognize_detected_faces", "blob_to_embedding",
    "validate_record_metadata", "row_to_person", "check_readonly_schema",
    "registered_from_rows", "load_registered_embeddings",
)


def original_namespace():
    # Chỉ lấy định nghĩa/constant; không chạy code setup, demo hoặc Run All.
    source = notebook_namespace(NOTEBOOK, FUNCTIONS, (
        "LANDMARK_NAMES", "DETECTION_KEYS", "NUMERIC_TYPES", "EMBEDDING_SIZE",
        "BLOB_BYTES", "ENROLLMENT_IMAGES_PER_PERSON", "EXPECTED_SCHEMA_SQL",
        "expected_model_sha256", "embedding_config",
    ))
    source.update(deepcopy=deepcopy, closing=closing, sqlite3=sqlite3,
                  hashlib=hashlib, project_root=PROJECT_ROOT)
    return source


def unit(index=0):
    result = np.zeros(512, np.float32)
    result[index] = 1
    return result


def metadata():
    return {"bbox": [10, 20, 180, 190], "confidence": np.float32(0.9),
            "landmarks": {name: [30 + i, 40 + i] for i, name in enumerate(
                ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right"))}}


class TestRecognition(unittest.TestCase):
    def setUp(self):
        self.source = original_namespace()
        self.face = np.arange(160 * 160 * 3, dtype=np.uint8).reshape(160, 160, 3)
        self.registry = {"person_b": unit(1), "person_a": unit()}
        self.net = object()

    def compare_error(self, detections, registry=None, threshold=0.5):
        """Đối chiếu loại lỗi, nội dung và số inference với notebook độc lập."""
        registry = self.registry if registry is None else registry
        expected_extract = Mock(return_value=unit())
        self.source["extract_embedding"] = expected_extract
        outcomes = []
        for function in (self.source["recognize_detected_faces"], recognition.recognize_detected_faces):
            with patch.object(recognition, "extract_embedding", return_value=unit()) as actual_extract:
                try:
                    function(detections, self.net, registry, threshold)
                except (TypeError, ValueError, RuntimeError) as error:
                    outcomes.append((type(error), str(error)))
                else:
                    self.fail("Notebook và module phải cùng từ chối input")
                if function is recognition.recognize_detected_faces:
                    self.assertEqual(actual_extract.call_count, expected_extract.call_count)
        self.assertEqual(outcomes[0], outcomes[1])
        return outcomes[0], expected_extract.call_count

    def test_both_bodies_preserve_notebook10(self):
        source = (SOURCE_ROOT / "FaceNet/recognition.py").read_text(encoding="utf-8")
        for name in ("recognize_aligned_face", "recognize_detected_faces"):
            self.assertEqual(function_body(source, name), notebook_function_body(NOTEBOOK, name))

    def test_one_face_known_and_exact_output_keys(self):
        with patch.object(recognition, "extract_embedding", return_value=unit()) as extract:
            result = recognition.recognize_detected_faces([{"face_rgb": self.face}], self.net, self.registry, 0.5)
        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0]), {"face_index", "predicted_id", "best_match_id", "similarity",
                                          "threshold", "is_known", "ranking"})
        self.assertEqual(result[0]["predicted_id"], "person_a")
        self.assertIs(result[0]["is_known"], True)
        self.assertEqual(result[0]["similarity"], 1.0)
        extract.assert_called_once()
        self.assertIs(extract.call_args.args[0], self.face)
        self.assertIs(extract.call_args.args[1], self.net)

    def test_unknown_retains_best_candidate_and_full_ranking(self):
        with patch.object(recognition, "extract_embedding", return_value=unit(2)):
            result = recognition.recognize_aligned_face(self.face, self.net, self.registry, 0.5)
        self.assertEqual(result["predicted_id"], "Unknown")
        self.assertEqual(result["best_match_id"], "person_a")
        self.assertIs(result["is_known"], False)
        self.assertEqual([r["person_id"] for r in result["ranking"]], ["person_a", "person_b"])
        self.assertEqual([r["rank"] for r in result["ranking"]], [1, 2])

    def test_similarity_equal_threshold_is_known(self):
        with patch.object(recognition, "extract_embedding", return_value=unit(2)):
            result = recognition.recognize_aligned_face(self.face, self.net, self.registry, 0.0)
        self.assertEqual(result["similarity"], result["threshold"])
        self.assertIs(result["is_known"], True)

    def test_multiple_faces_keep_order_and_unknown(self):
        faces = [self.face, self.face.copy(), self.face.copy()]
        with patch.object(recognition, "extract_embedding", side_effect=[unit(1), unit(2), unit()]) as extract:
            results = recognition.recognize_detected_faces(
                [{"face_rgb": face} for face in faces], self.net, self.registry, 0.5)
        self.assertEqual([r["face_index"] for r in results], [0, 1, 2])
        self.assertEqual([r["predicted_id"] for r in results], ["person_b", "Unknown", "person_a"])
        for call, face in zip(extract.call_args_list, faces):
            self.assertIs(call.args[0], face)

    def test_empty_returns_before_threshold_registry_or_inference(self):
        for threshold in (None, "invalid", float("nan"), 0.5):
            with patch.object(recognition, "extract_embedding") as extract:
                actual = recognition.recognize_detected_faces([], None, None, threshold)
                self.assertEqual(actual, self.source["recognize_detected_faces"]([], None, None, threshold))
                self.assertEqual(actual, [])
                extract.assert_not_called()

    def test_all_optional_metadata_and_independent_copy(self):
        detection = {"face_rgb": self.face, **metadata()}
        with patch.object(recognition, "extract_embedding", return_value=unit()):
            result = recognition.recognize_detected_faces([detection], self.net, self.registry, 0.5)[0]
        for name in metadata():
            self.assertEqual(result[name], detection[name])
            if isinstance(detection[name], (dict, list)):
                self.assertIsNot(result[name], detection[name])
        result["bbox"][0] = 999
        result["landmarks"]["left_eye"][0] = 999
        self.assertEqual(detection["bbox"][0], 10)
        self.assertEqual(detection["landmarks"]["left_eye"][0], 30)

    def test_absent_metadata_is_not_added(self):
        for names in ((), ("bbox",), ("confidence",), ("landmarks",)):
            detection = {"face_rgb": self.face, **{k: v for k, v in metadata().items() if k in names}}
            with patch.object(recognition, "extract_embedding", return_value=unit()):
                result = recognition.recognize_detected_faces([detection], self.net, self.registry, 0.5)[0]
            self.assertEqual(set(result) & set(metadata()), set(names))

    def test_tuple_and_numpy_scalar_metadata_preserve_types(self):
        detection = {"face_rgb": self.face, "bbox": (np.int64(1), 2, 9999, 9999),
                     "landmarks": {name: (np.float64(2000), 2000) for name in metadata()["landmarks"]}}
        with patch.object(recognition, "extract_embedding", return_value=unit()):
            result = recognition.recognize_detected_faces([detection], self.net, self.registry, 0.5)[0]
        # Không clip theo kích thước crop 160x160 hoặc tự chuyển kiểu metadata.
        self.assertEqual(result["bbox"], detection["bbox"])
        self.assertIsInstance(result["bbox"], tuple)
        self.assertIsInstance(result["bbox"][0], np.int64)
        self.assertEqual(result["landmarks"], detection["landmarks"])

    def test_invalid_metadata_matches_notebook(self):
        cases = [
            {"bbox": None}, {"bbox": [1, 2, 3]}, {"bbox": [1, 2, 1, 4]},
            {"bbox": [-1, 0, 10, 10]}, {"bbox": [True, 0, 10, 10]},
            {"bbox": [0, 0, np.inf, 10]}, {"confidence": None},
            {"confidence": True}, {"confidence": -0.1}, {"confidence": 1.1},
            {"confidence": np.nan}, {"landmarks": None}, {"landmarks": {}},
            {"landmarks": {**metadata()["landmarks"], "nose": [False, 2]}},
            {"landmarks": {**metadata()["landmarks"], "nose": [1, np.inf]}},
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                error, count = self.compare_error([{"face_rgb": self.face, **fields}])
                self.assertIn("face_index 0:", error[1])
                self.assertEqual(count, 0)

    def test_extra_keys_are_rejected(self):
        error, count = self.compare_error([{"face_rgb": self.face, "person_id": "not_allowed"}])
        self.assertIn("khóa ngoài hợp đồng", error[1])
        self.assertEqual(count, 0)

    def test_invalid_detection_container_and_missing_face(self):
        for value in (None, {}, (), [{"bbox": [0, 0, 10, 10]}], [None], [self.face]):
            with self.subTest(type=type(value).__name__):
                _, count = self.compare_error(value)
                self.assertEqual(count, 0)

    def test_invalid_face_matches_notebook_and_has_index(self):
        for face in (None, [], np.zeros((0, 0, 3), np.uint8),
                     np.zeros((80, 80, 3), np.uint8), self.face.astype(np.float32)):
            error, count = self.compare_error([{"face_rgb": self.face}, {"face_rgb": face}])
            self.assertIn("face_index 1:", error[1])
            self.assertEqual(count, 0)

    def test_later_invalid_metadata_prevents_all_inference(self):
        # Ảnh đầu hợp lệ cũng không được inference khi metadata ảnh cuối lỗi.
        _, count = self.compare_error([
            {"face_rgb": self.face}, {"face_rgb": self.face, "confidence": 2}])
        self.assertEqual(count, 0)

    def test_invalid_threshold_prevents_inference(self):
        for threshold in (None, True, np.bool_(False), "0.5", np.nan, np.inf, -1.01, 1.01):
            _, count = self.compare_error([{"face_rgb": self.face}], threshold=threshold)
            self.assertEqual(count, 0)

    def test_face_validation_precedes_threshold(self):
        error, count = self.compare_error([{"face_rgb": None}], threshold="invalid")
        self.assertIn("face_rgb không được là None", error[1])
        self.assertEqual(count, 0)

    def test_empty_registry_matches_error_after_one_inference(self):
        error, count = self.compare_error([{"face_rgb": self.face}], registry={})
        self.assertIn("face_index 0: Không có danh tính", error[1])
        self.assertEqual(count, 1)

    def test_invalid_registry_uses_matching_behavior(self):
        cases = [[], {"Unknown": unit()}, {" bad": unit()}, {"bad": np.zeros(512, np.float32)},
                 {"bad": unit().astype(np.float64)}]
        for registry in cases:
            _, count = self.compare_error([{"face_rgb": self.face}], registry=registry)
            self.assertEqual(count, 1)

    def test_inference_errors_keep_type_message_index_and_cause(self):
        for exception in (TypeError, ValueError, RuntimeError):
            outcomes = []
            for function in (self.source["recognize_detected_faces"], recognition.recognize_detected_faces):
                self.source["extract_embedding"] = Mock(side_effect=[unit(), exception("inference failed")])
                with patch.object(recognition, "extract_embedding", side_effect=[unit(), exception("inference failed")]):
                    with self.assertRaises(exception) as error:
                        function([{"face_rgb": self.face}] * 2, self.net, self.registry, 0.5)
                outcomes.append(str(error.exception))
                self.assertIsInstance(error.exception.__cause__, exception)
            self.assertEqual(outcomes, ["face_index 1: inference failed"] * 2)

    def test_call_order_reuses_validation_extraction_and_matching(self):
        events = []
        validate_all = recognition.validate_mtcnn_output
        extract = Mock(side_effect=lambda face, net: events.append("extract") or unit())
        decide = recognition.detect_unknown
        with patch.object(recognition, "validate_mtcnn_output", side_effect=lambda ds: events.append("validate_all") or validate_all(ds)), \
                patch.object(recognition, "extract_embedding", extract), \
                patch.object(recognition, "detect_unknown", side_effect=lambda *args: events.append("decide") or decide(*args)):
            recognition.recognize_detected_faces([{"face_rgb": self.face}] * 2, self.net, self.registry, 0.5)
        self.assertEqual(events, ["validate_all", "extract", "decide", "extract", "decide"])

    def test_input_pixels_metadata_and_registry_unchanged(self):
        face = self.face[:, ::-1, :]  # view không contiguous vẫn giữ nguyên.
        detections = [{"face_rgb": face, **metadata()}]
        before = deepcopy(detections)
        registered_before = deepcopy(self.registry)
        with patch.object(recognition, "extract_embedding", return_value=unit()) as extract:
            recognition.recognize_detected_faces(detections, self.net, self.registry, 0.5)
        self.assertIs(extract.call_args.args[0], face)
        np.testing.assert_array_equal(detections[0]["face_rgb"], before[0]["face_rgb"])
        for name in metadata():
            self.assertEqual(detections[0][name], before[0][name])
        for key in self.registry:
            np.testing.assert_array_equal(self.registry[key], registered_before[key])

    def test_single_face_validation_and_exceptions_match_notebook(self):
        for face, threshold in ((None, 0.5), (self.face.astype(np.float32), 0.5), (self.face, True)):
            outcomes = []
            for function in (self.source["recognize_aligned_face"], recognition.recognize_aligned_face):
                with patch.object(recognition, "extract_embedding") as extract:
                    try:
                        function(face, self.net, self.registry, threshold)
                    except (TypeError, ValueError) as error:
                        outcomes.append((type(error), str(error)))
                    extract.assert_not_called()
            self.assertEqual(len(outcomes), 2)
            self.assertEqual(outcomes[0], outcomes[1])

    def test_threshold_is_required_for_both_apis(self):
        for function, face_input in ((recognition.recognize_aligned_face, self.face),
                                     (recognition.recognize_detected_faces, [])):
            with self.assertRaises(TypeError):
                function(face_input, self.net, self.registry)

    def test_import_does_not_access_assets_load_model_or_connect_sqlite(self):
        code = textwrap.dedent('''
            import os, sys
            sys.dont_write_bytecode = True
            def readonly(event, args):
                if event == "open":
                    mode, flags = args[1:3]
                    if str(args[0]).lower().endswith((".db", ".onnx", ".png", ".jpg", ".ipynb", ".json")):
                        raise AssertionError("Import accessed asset")
                    if ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                            or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
                        raise AssertionError("Import wrote a file")
                if event in ("sqlite3.connect", "os.mkdir", "socket.__new__"):
                    raise AssertionError("Import performed I/O")
            sys.addaudithook(readonly)
        ''') + f"\nsys.path.insert(0, {str(SOURCE_ROOT)!r})\n" + textwrap.dedent('''
            import cv2
            from unittest.mock import patch
            with patch.object(cv2.dnn, "readNetFromONNX", side_effect=AssertionError("Model load")):
                from FaceNet import recognition
            for name in ("sqlite3", "torch", "IPython", "MTCNN", "FaceNet.database"):
                assert name not in sys.modules, name
            print("RECOGNITION_IMPORT_OK")
        ''')
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT / "notebooks",
                                capture_output=True, text=True, encoding="utf-8",
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("RECOGNITION_IMPORT_OK", result.stdout)


class TestRecognitionNotebookRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = protected_hashes()
        cls.source = original_namespace()
        cls.version = json.dumps(cls.source["embedding_config"], sort_keys=True, separators=(",", ":"))
        cls.db_path = PROJECT_ROOT / "data/face_database_test.db"
        cls.registry = database.load_registered_embeddings(cls.db_path, cls.version)
        cls.original_registry = cls.source["load_registered_embeddings"](cls.db_path, cls.version)
        if set(cls.registry) != set(cls.original_registry) or len(cls.registry) != 5:
            raise AssertionError("Registry cần đúng 5 danh tính của baseline")
        for key in cls.registry:
            np.testing.assert_array_equal(cls.registry[key], cls.original_registry[key])
        known = json.loads((PROJECT_ROOT / "data/processed_faces_imm/manifest.json").read_text(encoding="utf-8"))
        grouped = {}
        for record in known["records"]:
            if record["status"] == "saved":
                grouped.setdefault(record["person_id"], []).append(record)
        cls.records = []
        for person_id, records in sorted(grouped.items()):
            ordered = sorted(records, key=lambda r: r["processed_image"])
            if len(ordered) != 5:
                raise AssertionError("Split Known phải giữ 3 enrollment + 2 query")
            cls.records.extend((person_id, record) for record in ordered[3:])
        unknown = json.loads((PROJECT_ROOT / "data/processed_unknown_faces_imm/manifest.json").read_text(encoding="utf-8"))
        cls.records.extend(("Unknown", record) for record in sorted(
            (r for r in unknown["records"] if r["status"] == "saved" and r["split"] == "evaluation"),
            key=lambda r: (r["person_id"], r["processed_image"])))
        if len(cls.records) != 32:
            raise AssertionError("Hồi quy cần đúng 10 Known + 22 Unknown evaluation")
        cls.detections = []
        for i, (_, record) in enumerate(cls.records):
            bgr = cv2.imread(str(PROJECT_ROOT / record["processed_image"]), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise AssertionError("Không đọc được PNG hồi quy")
            detection = {"face_rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)}
            # Metadata hợp lệ chỉ phục vụ kiểm tra hợp đồng, không giả làm MTCNN.
            if i % 4 == 0:
                detection.update(metadata())
            elif i % 4 == 1:
                detection["bbox"] = (1, 2, 200, 220)
            elif i % 4 == 2:
                detection["confidence"] = 0.8
            cls.detections.append(detection)
        cls.net = load_facenet_model(SOURCE_ROOT / "FaceNet/models/facenet.onnx")

    @classmethod
    def tearDownClass(cls):
        if protected_hashes() != cls.before:
            raise AssertionError("Asset/tài liệu gốc thay đổi khi chạy recognition")
        if list(cls.db_path.parent.glob(cls.db_path.name + "-*")):
            raise AssertionError("Có SQLite sidecar ngoài dự kiến")

    def test_real_onnx_complete_outputs_match_notebook10(self):
        snapshot = deepcopy(self.detections)
        registry_before = deepcopy(self.registry)
        stats = {"unique_faces": 32, "known_queries": 10, "unknown_evaluation_queries": 22,
                 "single_face_list_calls": 0, "multi_face_list_calls": 0, "aligned_face_calls": 0,
                 "face_output_comparisons": 0, "complete_output_mismatches": 0,
                 "metadata_mismatches": 0, "ranking_mismatches": 0,
                 "known_unknown_mismatches": 0, "similarity_max_abs_difference": 0.0,
                 "thresholds": [0.48, 0.5], "known_decisions": 0, "unknown_decisions": 0}

        def compare(actual, expected):
            stats["face_output_comparisons"] += 1
            stats["complete_output_mismatches"] += int(actual != expected)
            stats["metadata_mismatches"] += int(
                {k: actual[k] for k in metadata() if k in actual} !=
                {k: expected[k] for k in metadata() if k in expected})
            stats["ranking_mismatches"] += int(actual["ranking"] != expected["ranking"])
            stats["known_unknown_mismatches"] += int(actual["is_known"] != expected["is_known"])
            stats["similarity_max_abs_difference"] = max(stats["similarity_max_abs_difference"],
                                                         abs(actual["similarity"] - expected["similarity"]))
            self.assertEqual(actual, expected)  # toàn bộ key, giá trị và metadata.

        # Không mock extraction hoặc decision trong hồi quy này: cả hai chạy ONNX.
        with patch.object(database.sqlite3, "connect", side_effect=AssertionError("Registry RAM không mở DB")):
            for threshold in stats["thresholds"]:
                for detection in self.detections:
                    expected = self.source["recognize_detected_faces"](
                        [detection], self.net, self.original_registry, threshold)
                    actual = recognition.recognize_detected_faces([detection], self.net, self.registry, threshold)
                    self.assertEqual(len(actual), 1)
                    compare(actual[0], expected[0])
                    stats["single_face_list_calls"] += 1
                    stats["known_decisions" if actual[0]["is_known"] else "unknown_decisions"] += 1
            # Đảo thứ tự và trộn Known/Unknown để kiểm tra face_index theo input.
            order = list(range(31, -1, -1))
            detections = [self.detections[i] for i in order]
            expected = self.source["recognize_detected_faces"](detections, self.net, self.original_registry, 0.48)
            actual = recognition.recognize_detected_faces(detections, self.net, self.registry, 0.48)
            self.assertEqual(len(actual), 32)
            self.assertEqual([r["face_index"] for r in actual], list(range(32)))
            for a, e in zip(actual, expected):
                compare(a, e)
            stats["multi_face_list_calls"] += 1
            for detection in self.detections:
                expected = self.source["recognize_aligned_face"](
                    detection["face_rgb"], self.net, self.original_registry, 0.48)
                actual = recognition.recognize_aligned_face(detection["face_rgb"], self.net, self.registry, 0.48)
                compare(actual, expected)
                stats["aligned_face_calls"] += 1
            self.assertEqual(recognition.recognize_detected_faces([], None, None, None),
                             self.source["recognize_detected_faces"]([], None, None, None))
        self.assertGreater(stats["known_decisions"], 0)
        self.assertGreater(stats["unknown_decisions"], 0)
        for actual, before in zip(self.detections, snapshot):
            np.testing.assert_array_equal(actual["face_rgb"], before["face_rgb"])
            self.assertEqual({k: v for k, v in actual.items() if k != "face_rgb"},
                             {k: v for k, v in before.items() if k != "face_rgb"})
        for key in registry_before:
            np.testing.assert_array_equal(self.registry[key], registry_before[key])
        print("RECOGNITION_REGRESSION " + json.dumps(stats, sort_keys=True))

    def test_sqlite_registry_composition_is_readonly_without_initialization(self):
        real_connect = sqlite3.connect
        opened = []
        statements = []

        def guarded_connect(database_uri, *args, **kwargs):
            self.assertIn("?mode=ro", database_uri)
            self.assertIs(kwargs.get("uri"), True)
            opened.append(database_uri)
            connection = real_connect(database_uri, *args, **kwargs)
            connection.set_trace_callback(statements.append)
            return connection

        with patch.object(database.sqlite3, "connect", side_effect=guarded_connect), \
                patch.object(database, "initialize_database", side_effect=AssertionError("No initialization")):
            registry = database.load_registered_embeddings(self.db_path, self.version)
            original = self.source["load_registered_embeddings"](self.db_path, self.version)
        self.assertEqual(len(opened), 2)
        self.assertTrue(any("query_only=ON" in sql for sql in statements))
        for sql in statements:
            self.assertTrue(sql.lstrip().upper().startswith(("SELECT", "PRAGMA")), sql)
        for person_id in original:
            np.testing.assert_array_equal(registry[person_id], original[person_id])
        detection = self.detections[0]
        actual = recognition.recognize_detected_faces([detection], self.net, registry, 0.48)
        expected = self.source["recognize_detected_faces"]([detection], self.net, original, 0.48)
        self.assertEqual(actual, expected)

    def test_model_version_mismatch_stops_before_inference(self):
        outcomes = []
        with patch.object(recognition, "extract_embedding") as extract:
            for loader in (self.source["load_registered_embeddings"], database.load_registered_embeddings):
                try:
                    registry = loader(self.db_path, self.version + "_incompatible_MTCNN")
                    recognition.recognize_detected_faces(self.detections, self.net, registry, 0.48)
                except ValueError as error:
                    outcomes.append(str(error))
            extract.assert_not_called()
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertIn("model_version không tương thích", outcomes[0])


if __name__ == "__main__":
    unittest.main()
