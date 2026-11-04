"""SQLite persistence: chỉ ghi DB tạm ngoài project; baseline luôn read-only."""

from contextlib import closing, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from _support import (
    PROJECT_ROOT, SOURCE_ROOT, function_body, notebook_function_body,
    notebook_namespace, protected_hashes,
)
from FaceNet import database as db


BASELINE = PROJECT_ROOT / "data" / "face_database_test.db"
VERSION = "phase3_test_v1"
SQL_COLUMNS = "person_id, embedding, num_images, model_version, embedding_sha256"
SOURCE_FUNCTIONS = (
    "validate_database_path", "check_schema", "connect_database", "initialize_database",
    "validate_database_embedding", "validate_person_id", "embedding_to_blob", "blob_to_embedding",
    "validate_record_metadata", "row_to_person", "add_person", "get_person", "list_people",
    "update_person", "delete_person",
)


def vector(index=0):
    result = np.zeros(512, np.float32)
    result[index] = 1
    return result


def original_database(project_root):
    """Giữ source notebook, chỉ cung cấp globals thay cho việc chạy cell setup."""
    source = notebook_namespace("06_face_database.ipynb", SOURCE_FUNCTIONS,
                                ("EMBEDDING_SIZE", "BLOB_BYTES", "CREATE_TABLE_SQL"))
    source.update(sqlite3=sqlite3, closing=closing, hashlib=hashlib,
                  project_root=project_root, MODEL_VERSION=VERSION)
    return source


def raw_rows(path):
    with closing(sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        return connection.execute(f"SELECT {SQL_COLUMNS} FROM FaceDatabase ORDER BY person_id").fetchall()


class TemporaryDatabaseCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="facenet_phase3_")
        self.folder = Path(self.temporary.name).resolve()
        self.addCleanup(self.cleanup_temporary)
        self.assert_safe_temporary()
        self.path = self.folder / "module" / "test.db"
        db.initialize_database(self.path)

    def assert_safe_temporary(self):
        # Xác minh target trước cleanup đệ quy của TemporaryDirectory trên Windows.
        if (not self.folder.is_relative_to(Path(tempfile.gettempdir()).resolve())
                or not self.folder.name.startswith("facenet_phase3_")
                or self.folder.is_relative_to(PROJECT_ROOT.resolve())):
            raise AssertionError("Thư mục test phải nằm trong vùng tạm ngoài project")

    def cleanup_temporary(self):
        self.assert_safe_temporary()
        self.temporary.cleanup()

    def add(self, person_id="person", embedding=None, num_images=3, version=VERSION):
        db.add_person(self.path, person_id, vector() if embedding is None else embedding, num_images, version)

    def assert_people_equal(self, actual, expected):
        if actual is None or expected is None:
            self.assertIs(actual, expected)
            return
        self.assertEqual(set(actual), set(expected))
        np.testing.assert_array_equal(actual["embedding"], expected["embedding"])
        for key in ("person_id", "num_images", "model_version"):
            self.assertEqual(actual[key], expected[key])


class TestDatabase(TemporaryDatabaseCase):
    def test_core_bodies_match_notebook(self):
        source = (SOURCE_ROOT / "FaceNet/database.py").read_text(encoding="utf-8")
        for name in ("check_schema", "embedding_to_blob", "blob_to_embedding", "validate_record_metadata",
                     "row_to_person", "add_person", "update_person", "delete_person"):
            with self.subTest(function=name):
                self.assertEqual(function_body(source, name), notebook_function_body("06_face_database.ipynb", name))
        self.assertEqual(function_body(source, "registered_from_rows"), notebook_function_body("07_face_matching.ipynb", "registered_from_rows"))

    def test_schema_initialization_and_existing_database_preservation(self):
        self.add()
        before = self.path.read_bytes()
        self.assertIsNone(db.initialize_database(self.path))
        self.assertEqual(self.path.read_bytes(), before)
        with closing(db.connect_database(self.path, readonly=True)) as connection:
            db.check_schema(connection)
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone(), ("ok",))
            self.assertEqual(connection.execute("SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall(), [("table", "FaceDatabase")])

    def test_connect_missing_does_not_create_database(self):
        missing = self.folder / "missing.db"
        with self.assertRaises(FileNotFoundError):
            db.connect_database(missing)
        self.assertFalse(missing.exists())

    def test_absolute_paths_and_baseline_write_guard(self):
        for path in (None, 1):
            with self.assertRaises(TypeError):
                db.validate_database_path(path)
        with self.assertRaisesRegex(ValueError, "tuyệt đối"):
            db.validate_database_path("relative.db")
        with self.assertRaisesRegex(ValueError, "không phải file"):
            db.validate_database_path(self.folder)
        # Không gọi initialize/CRUD trên baseline; guard phải chặn trước SQLite.
        with patch.object(db.sqlite3, "connect", side_effect=AssertionError("Không được mở baseline để ghi")) as opener:
            with self.assertRaises(PermissionError):
                db.validate_database_path(BASELINE, for_write=True)
            with self.assertRaises(PermissionError):
                db.connect_database(BASELINE)
            opener.assert_not_called()

    def test_initialization_stops_before_io_when_path_guard_rejects(self):
        with patch.object(db, "validate_database_path", side_effect=PermissionError("blocked")), patch.object(db.sqlite3, "connect") as opener:
            with self.assertRaises(PermissionError):
                db.initialize_database(self.folder / "blocked.db")
            opener.assert_not_called()

    def test_encode_decode_is_exact_little_endian_and_independent(self):
        original = vector(4); before = original.copy()
        blob = db.embedding_to_blob(original)
        self.assertIsInstance(blob, bytes)
        self.assertEqual(len(blob), 2048)
        self.assertEqual(blob, original.astype("<f4").tobytes(order="C"))
        for value in (blob, bytearray(blob), memoryview(blob)):
            decoded = db.blob_to_embedding(value)
            self.assertEqual(decoded.dtype, np.float32)
            np.testing.assert_array_equal(decoded, original)
            decoded[4] = 0
            self.assertEqual(blob, db.embedding_to_blob(original))
        np.testing.assert_array_equal(original, before)

    def test_invalid_embeddings_match_notebook(self):
        source = original_database(self.folder / "reference")
        invalid = [None, [], np.zeros((1, 512), np.float32), np.zeros(511, np.float32),
                   vector().astype(np.float64), vector().astype(np.int32), np.zeros(512, np.float32),
                   vector() * 2, np.full(512, np.nan, np.float32), np.full(512, np.inf, np.float32)]
        for index, value in enumerate(invalid):
            outcomes = []
            for fn in (source["embedding_to_blob"], db.embedding_to_blob):
                try:
                    fn(value)
                except (TypeError, ValueError) as error:
                    outcomes.append((type(error), str(error)))
            with self.subTest(case=index):
                self.assertEqual(len(outcomes), 2)
                self.assertEqual(outcomes[0], outcomes[1])

    def test_invalid_blob_types_sizes_and_norm(self):
        for value in (None, "blob", np.zeros(512, np.float32)):
            with self.assertRaises(TypeError):
                db.blob_to_embedding(value)
        for value in (b"", bytes(2047), bytes(2049)):
            with self.assertRaises(ValueError):
                db.blob_to_embedding(value)
        for value in (np.zeros(512, np.float32), np.full(512, np.nan, np.float32)):
            with self.assertRaises(ValueError):
                db.blob_to_embedding(value.astype("<f4").tobytes())

    def test_record_metadata_validation_matches_notebook(self):
        source = original_database(self.folder / "reference")
        cases = [(None, 3, VERSION), ("", 3, VERSION), (" leading", 3, VERSION),
                 ("a", True, VERSION), ("a", 0, VERSION), ("a", -1, VERSION),
                 ("a", 3.0, VERSION), ("a", 2**63, VERSION),
                 ("a", 3, None), ("a", 3, ""), ("a", 3, " v1")]
        for values in cases:
            outcomes = []
            for fn in (source["validate_record_metadata"], db.validate_record_metadata):
                try:
                    fn(*values)
                except (TypeError, ValueError) as error:
                    outcomes.append((type(error), str(error)))
            self.assertEqual(len(outcomes), 2)
            self.assertEqual(outcomes[0], outcomes[1])
        db.validate_record_metadata("valid", np.int64(3), VERSION)

    def test_checksum_blob_and_version_are_verified(self):
        self.add()
        row = raw_rows(self.path)[0]
        wrong_checksum = (*row[:4], "0" * 64)
        with self.assertRaisesRegex(ValueError, "Checksum"):
            db.row_to_person(wrong_checksum, VERSION)
        with self.assertRaisesRegex(ValueError, "model_version"):
            db.row_to_person(row, "other_version")
        with self.assertRaises(TypeError):
            db.row_to_person((row[0], memoryview(row[1]), *row[2:]), VERSION)
        zero_blob = bytes(2048)
        with self.assertRaisesRegex(ValueError, "norm"):
            db.row_to_person((row[0], zero_blob, 3, VERSION, hashlib.sha256(zero_blob).hexdigest()))

    def test_crud_order_and_return_values(self):
        self.assertIsNone(db.add_person(self.path, "z", vector(), 3, VERSION))
        self.add("a", vector(1)); self.add("m", vector(2))
        self.assertEqual([r["person_id"] for r in db.list_people(self.path)], ["a", "m", "z"])
        np.testing.assert_array_equal(db.get_person(self.path, "a", VERSION)["embedding"], vector(1))
        self.assertTrue(db.update_person(self.path, "a", vector(3), 3, VERSION))
        np.testing.assert_array_equal(db.get_person(self.path, "a", VERSION)["embedding"], vector(3))
        self.assertTrue(db.delete_person(self.path, "m"))
        self.assertIsNone(db.get_person(self.path, "m", VERSION))
        self.assertEqual(list(db.load_registered_embeddings(self.path, VERSION)), ["a", "z"])

    def test_duplicate_insert_and_missing_update_delete_preserve_rows(self):
        self.add(); before = raw_rows(self.path)
        with self.assertRaises(ValueError) as error:
            self.add(embedding=vector(1))
        self.assertIsInstance(error.exception.__cause__, sqlite3.IntegrityError)
        with self.assertRaises(KeyError):
            db.update_person(self.path, "missing", vector(), 3, VERSION)
        with self.assertRaises(KeyError):
            db.delete_person(self.path, "missing")
        self.assertEqual(raw_rows(self.path), before)

    def test_empty_database(self):
        self.assertEqual(db.list_people(self.path), [])
        self.assertIsNone(db.get_person(self.path, "missing", VERSION))
        with self.assertRaisesRegex(ValueError, "Database rỗng"):
            db.load_registered_embeddings(self.path, VERSION)

    def test_parameterized_id_does_not_execute_sql(self):
        person_id = "x'; DROP TABLE FaceDatabase; --"
        self.add(person_id)
        self.assertEqual(db.get_person(self.path, person_id, VERSION)["person_id"], person_id)
        self.assertTrue(db.delete_person(self.path, person_id))

    def test_readonly_connection_blocks_writes(self):
        with closing(db.connect_database(self.path, readonly=True)) as connection:
            self.assertEqual(connection.execute("PRAGMA query_only").fetchone(), (1,))
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM FaceDatabase")
        self.assertEqual(raw_rows(self.path), [])

    def test_transaction_rolls_back_all_statements(self):
        blob = db.embedding_to_blob(vector()); checksum = hashlib.sha256(blob).hexdigest()
        with closing(db.connect_database(self.path)) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                with connection:
                    connection.execute(f"INSERT INTO FaceDatabase ({SQL_COLUMNS}) VALUES (?,?,?,?,?)", ("a", blob, 3, VERSION, checksum))
                    connection.execute(f"INSERT INTO FaceDatabase ({SQL_COLUMNS}) VALUES (?,?,?,?,?)", ("a", blob, 3, VERSION, checksum))
        self.assertEqual(raw_rows(self.path), [])

    def test_corrupt_checksum_is_rejected_on_read_and_reinitialization(self):
        self.add()
        with closing(db.connect_database(self.path)) as connection:
            with connection:
                connection.execute("UPDATE FaceDatabase SET embedding_sha256=?", ("0" * 64,))
        before = self.path.read_bytes()
        for action in (lambda: db.get_person(self.path, "person", VERSION),
                       lambda: db.load_registered_embeddings(self.path, VERSION),
                       lambda: db.initialize_database(self.path)):
            with self.assertRaisesRegex(ValueError, "Checksum"):
                action()
        self.assertEqual(self.path.read_bytes(), before)
        # list_people chỉ đọc metadata, đúng phạm vi kiểm tra notebook 06.
        self.assertEqual(len(db.list_people(self.path)), 1)

    def test_existing_wrong_schema_is_not_migrated(self):
        for index, sql in enumerate(("CREATE TABLE Other (id TEXT)", db.CREATE_TABLE_SQL.replace("2048", "1024"))):
            path = self.folder / f"invalid_schema_{index}.db"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(sql); connection.commit()
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                db.initialize_database(path)
            self.assertEqual(path.read_bytes(), before)

    def test_connection_closes_if_schema_check_fails(self):
        connection = sqlite3.connect(self.path)
        with patch.object(db.sqlite3, "connect", return_value=connection), patch.object(db, "check_schema", side_effect=ValueError("bad schema")):
            with self.assertRaises(ValueError):
                db.connect_database(self.path)
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_registry_checks_count_version_duplicate_and_row_index(self):
        self.add(); row = raw_rows(self.path)[0]
        for rows in ([row, row], [(row[0], row[1], 2, *row[3:])], [(*row[:4], "0" * 64)]):
            with self.assertRaisesRegex(ValueError, "Bản ghi SQLite tại chỉ số"):
                db.registered_from_rows(rows, VERSION)
        with self.assertRaises(ValueError):
            db.load_registered_embeddings(self.path, "wrong_version")
        for value in (None, "", " "):
            with self.assertRaises((TypeError, ValueError)):
                db.get_person(self.path, "person", value)

    def test_import_has_no_asset_access_or_connection(self):
        code = textwrap.dedent('''
            import os, sys
            sys.dont_write_bytecode = True
            def readonly(event, args):
                if event == "open":
                    mode, flags = args[1:3]
                    if str(args[0]).lower().endswith((".db", ".onnx", ".png", ".ipynb")):
                        raise AssertionError("Import accessed asset")
                    if ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                            or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
                        raise AssertionError("Import wrote file")
                if event in ("sqlite3.connect", "os.mkdir", "socket.__new__"):
                    raise AssertionError("Import performed I/O")
            sys.addaudithook(readonly)
        ''') + f"\nsys.path.insert(0, {str(SOURCE_ROOT)!r})\n" + textwrap.dedent('''
            from FaceNet import database
            for name in ("cv2", "torch", "IPython", "MTCNN", "FaceNet.facenet_model"):
                assert name not in sys.modules, name
            print("DATABASE_IMPORT_OK")
        ''')
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT / "notebooks",
                                capture_output=True, text=True, encoding="utf-8",
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("DATABASE_IMPORT_OK", result.stdout)


class TestBaselineRegression(unittest.TestCase):
    def test_readonly_baseline_matches_notebook06_and_reader07(self):
        before = protected_hashes(); baseline_before = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
        source = original_database(PROJECT_ROOT)
        def readonly_reference(path):
            self.assertEqual(Path(path).resolve(), BASELINE.resolve())
            connection = sqlite3.connect(BASELINE.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
            connection.execute("PRAGMA query_only=ON")
            try:
                source["check_schema"](connection)
            except Exception:
                connection.close(); raise
            return connection
        # Chỉ thay provider kết nối cho các SELECT của notebook: không có mode=rw.
        source["connect_database"] = readonly_reference
        with closing(readonly_reference(BASELINE)) as connection:
            rows = connection.execute(f"SELECT {SQL_COLUMNS} FROM FaceDatabase ORDER BY person_id").fetchall()
            schema = connection.execute("SELECT sql FROM sqlite_master WHERE name='FaceDatabase'").fetchone()[0]
        self.assertEqual(len(rows), 5)
        version = rows[0][3]; source["MODEL_VERSION"] = version
        expected_people = source["list_people"](BASELINE)
        self.assertEqual(db.list_people(BASELINE), expected_people)
        source_reader = notebook_namespace("07_face_matching.ipynb",
            ("validate_person_id", "validate_database_embedding", "blob_to_embedding", "validate_record_metadata", "row_to_person", "registered_from_rows"),
            ("EMBEDDING_SIZE", "BLOB_BYTES", "ENROLLMENT_IMAGES_PER_PERSON"))
        source_reader["hashlib"] = hashlib
        expected_registry = source_reader["registered_from_rows"](rows, version)
        actual_registry = db.load_registered_embeddings(BASELINE, version)
        self.assertEqual(list(actual_registry), list(expected_registry))
        max_difference = 0.0
        for row in rows:
            person_id, blob, count, actual_version, checksum = row
            original = source["get_person"](BASELINE, person_id)
            actual = db.get_person(BASELINE, person_id, version)
            self.assertEqual((actual["person_id"], actual["num_images"], actual["model_version"]), (person_id, count, actual_version))
            np.testing.assert_array_equal(actual["embedding"], original["embedding"])
            np.testing.assert_array_equal(actual_registry[person_id], expected_registry[person_id])
            self.assertEqual(db.embedding_to_blob(actual["embedding"]), source["embedding_to_blob"](original["embedding"]))
            self.assertEqual(db.embedding_to_blob(actual["embedding"]), blob)
            self.assertEqual(hashlib.sha256(blob).hexdigest(), checksum)
            max_difference = max(max_difference, float(np.max(np.abs(actual["embedding"] - original["embedding"]))))
        with closing(db.connect_database(BASELINE, readonly=True)) as connection:
            db.check_schema(connection)
            self.assertEqual(connection.execute("SELECT sql FROM sqlite_master WHERE name='FaceDatabase'").fetchone()[0], schema)
        self.assertEqual(hashlib.sha256(BASELINE.read_bytes()).hexdigest(), baseline_before)
        self.assertEqual(protected_hashes(), before)
        print("DATABASE_BASELINE_REGRESSION " + json.dumps({"identities": len(rows), "schema_mismatches": 0,
              "id_order_mismatches": 0, "metadata_mismatches": 0, "blob_mismatches": 0,
              "checksum_mismatches": 0, "embedding_max_abs_difference": max_difference,
              "baseline_sha256_before": baseline_before, "baseline_sha256_after": hashlib.sha256(BASELINE.read_bytes()).hexdigest()}, sort_keys=True))


class TestCrudNotebookRegression(TemporaryDatabaseCase):
    def test_crud_errors_and_rollback_on_two_independent_temp_databases(self):
        reference_root = self.folder / "notebook"
        reference_path = reference_root / "data" / "face_database_test.db"
        source = original_database(reference_root)
        before_assets = protected_hashes()
        stats = {"temporary_databases": 2, "operations_compared": 0, "raw_row_mismatches": 0,
                 "result_mismatches": 0, "exception_mismatches": 0, "rollback_cases": 0}

        def compare(label, original_action, module_action, person_result=False):
            values = []; errors = []
            for action in (original_action, module_action):
                try:
                    with redirect_stdout(io.StringIO()):
                        values.append(action())
                    errors.append(None)
                except Exception as error:
                    values.append(None)
                    errors.append((type(error).__name__, str(error), type(error.__cause__).__name__ if error.__cause__ else None))
            with self.subTest(operation=label):
                self.assertEqual(errors[0], errors[1])
                if errors[0] is None:
                    if person_result:
                        self.assert_people_equal(values[1], values[0])
                    else:
                        self.assertEqual(values[0], values[1])
                self.assertEqual(raw_rows(reference_path), raw_rows(self.path))
            stats["operations_compared"] += 1

        compare("initialize", lambda: source["initialize_database"](reference_path), lambda: db.initialize_database(self.path))
        compare("initialize_existing", lambda: source["initialize_database"](reference_path), lambda: db.initialize_database(self.path))
        for person_id, index in (("z", 0), ("a", 1), ("m", 2)):
            compare("insert_" + person_id,
                    lambda person_id=person_id, index=index: source["add_person"](reference_path, person_id, vector(index), 3, VERSION),
                    lambda person_id=person_id, index=index: db.add_person(self.path, person_id, vector(index), 3, VERSION))
        compare("duplicate_insert", lambda: source["add_person"](reference_path, "a", vector(3), 3, VERSION), lambda: db.add_person(self.path, "a", vector(3), 3, VERSION))
        compare("get_existing", lambda: source["get_person"](reference_path, "a"), lambda: db.get_person(self.path, "a", VERSION), True)
        compare("get_missing", lambda: source["get_person"](reference_path, "missing"), lambda: db.get_person(self.path, "missing", VERSION), True)
        compare("list_sorted", lambda: source["list_people"](reference_path), lambda: db.list_people(self.path))
        compare("update_existing", lambda: source["update_person"](reference_path, "a", vector(3), 4, VERSION), lambda: db.update_person(self.path, "a", vector(3), 4, VERSION))
        compare("update_missing", lambda: source["update_person"](reference_path, "missing", vector(), 3, VERSION), lambda: db.update_person(self.path, "missing", vector(), 3, VERSION))
        source["MODEL_VERSION"] = "wrong_version"
        compare("get_wrong_version", lambda: source["get_person"](reference_path, "a"), lambda: db.get_person(self.path, "a", "wrong_version"), True)
        source["MODEL_VERSION"] = VERSION
        compare("delete_existing", lambda: source["delete_person"](reference_path, "m"), lambda: db.delete_person(self.path, "m"))
        compare("delete_missing", lambda: source["delete_person"](reference_path, "missing"), lambda: db.delete_person(self.path, "missing"))
        compare("invalid_embedding", lambda: source["add_person"](reference_path, "bad", vector() * 2, 3, VERSION), lambda: db.add_person(self.path, "bad", vector() * 2, 3, VERSION))
        compare("invalid_metadata", lambda: source["update_person"](reference_path, "a", vector(), True, VERSION), lambda: db.update_person(self.path, "a", vector(), True, VERSION))

        class FailAfterMutation(sqlite3.Connection):
            def execute(self, sql, parameters=()):
                cursor = super().execute(sql, parameters)
                if sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
                    raise sqlite3.OperationalError("forced failure after mutation")
                return cursor

        reference_connector = source["connect_database"]
        def failing_connection(path):
            # Factory chỉ nhận hai database tạm; tuyệt đối không mở baseline.
            candidate = Path(path).resolve()
            if candidate not in (reference_path.resolve(), self.path.resolve()):
                raise AssertionError("Rollback test chỉ được dùng database tạm")
            return sqlite3.connect(candidate.as_uri() + "?mode=rw", uri=True, factory=FailAfterMutation)
        for operation in ("add_person", "update_person", "delete_person"):
            before_rows = raw_rows(self.path)
            args = ("rollback_new", vector(), 3, VERSION) if operation == "add_person" else (("a", vector(4), 3, VERSION) if operation == "update_person" else ("a",))
            source["connect_database"] = failing_connection
            try:
                with patch.object(db, "connect_database", side_effect=failing_connection):
                    compare("rollback_" + operation,
                            lambda operation=operation, args=args: source[operation](reference_path, *args),
                            lambda operation=operation, args=args: getattr(db, operation)(self.path, *args))
            finally:
                source["connect_database"] = reference_connector
            self.assertEqual(raw_rows(self.path), before_rows)
            stats["rollback_cases"] += 1
        self.assertEqual(protected_hashes(), before_assets)
        print("DATABASE_CRUD_REGRESSION " + json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
