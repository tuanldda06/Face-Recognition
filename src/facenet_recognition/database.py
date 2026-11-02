
from __future__ import annotations

from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3

import numpy as np

from .matching import EMBEDDING_SIZE, validate_database_embedding, validate_person_id


BLOB_BYTES = EMBEDDING_SIZE * np.dtype("<f4").itemsize
ENROLLMENT_IMAGES_PER_PERSON = 3
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS FaceDatabase (
    person_id TEXT PRIMARY KEY NOT NULL,
    embedding BLOB NOT NULL CHECK (typeof(embedding) = 'blob' AND length(embedding) = 2048),
    num_images INTEGER NOT NULL CHECK (typeof(num_images) = 'integer' AND num_images > 0),
    model_version TEXT NOT NULL CHECK (length(trim(model_version)) > 0),
    embedding_sha256 TEXT NOT NULL CHECK (length(embedding_sha256) = 64)
);
"""


def _is_project_baseline(path: Path) -> bool:
    """Tìm baseline khi có lời gọi tường minh, không resolve asset lúc import."""
    baseline = Path(__file__).resolve().parents[2] / "data" / "face_database_test.db"
    return path.resolve() == baseline.resolve() or (
        path.exists() and baseline.exists() and path.samefile(baseline)
    )


def validate_database_path(path: str | Path, *, for_write: bool = False) -> Path:
    """Kiểm tra path tuyệt đối, từ chối symlink và ghi vào baseline.

    Thay allowlist một file của notebook bằng path tường minh để dùng được DB
    tạm/DB riêng. Baseline (kể cả hardlink cùng file) luôn cấm ghi.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("Database path phải là str hoặc pathlib.Path.")
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("Database path phải là đường dẫn tuyệt đối.")
    if candidate.is_symlink() or any(parent.is_symlink() for parent in candidate.parents):
        raise ValueError("Database path không được đi qua symlink.")
    if candidate.exists() and not candidate.is_file():
        raise ValueError("Database path không phải file.")
    resolved = candidate.resolve()
    if for_write and _is_project_baseline(resolved):
        raise PermissionError("Database baseline chỉ được mở read-only; hãy dùng database riêng để ghi.")
    return resolved


def check_schema(connection: sqlite3.Connection) -> None:
    """Giữ schema check notebook 06; không migrate, sửa hoặc xóa schema cũ."""
    objects = connection.execute(
        "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
    ).fetchall()
    if objects != [("table", "FaceDatabase")]:
        raise ValueError("Database phải chỉ có bảng FaceDatabase, không có trigger/view lạ.")
    actual = [(r[1], r[2], r[3], r[5])
              for r in connection.execute("PRAGMA table_info(FaceDatabase)")]
    expected = [
        ("person_id", "TEXT", 1, 1), ("embedding", "BLOB", 1, 0),
        ("num_images", "INTEGER", 1, 0), ("model_version", "TEXT", 1, 0),
        ("embedding_sha256", "TEXT", 1, 0),
    ]
    stored_sql = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
        ("table", "FaceDatabase"),
    ).fetchone()[0]
    normalize = lambda sql: " ".join(sql.upper().replace("IF NOT EXISTS ", "").split()).rstrip(";")
    if actual != expected or normalize(stored_sql) != normalize(CREATE_TABLE_SQL):
        raise ValueError("Schema khác dự kiến; không tự migrate hoặc xóa dữ liệu.")
    if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
        raise ValueError("SQLite quick_check phát hiện database không hợp lệ.")


def connect_database(path: str | Path, *, readonly: bool = False) -> sqlite3.Connection:
    """Mở file đã có bằng mode=rw hoặc ro; bên gọi chịu trách nhiệm close.

    Reader bật query_only. Baseline cố định dùng thêm immutable=1 để tránh
    sidecar; không áp immutable lên các database riêng đang được cập nhật.
    """
    path = validate_database_path(path, for_write=not readonly)
    if not path.is_file():
        raise FileNotFoundError("Hãy gọi initialize_database trước khi CRUD.")
    mode = "?mode=ro" if readonly else "?mode=rw"
    if readonly and _is_project_baseline(path):
        mode += "&immutable=1"
    connection = sqlite3.connect(path.as_uri() + mode, uri=True, timeout=5)
    try:
        if readonly:
            connection.execute("PRAGMA query_only=ON")
            if connection.execute("PRAGMA query_only").fetchone() != (1,):
                raise RuntimeError("Kết nối chưa ở chế độ query_only.")
        check_schema(connection)
    except Exception:
        connection.close()
        raise
    return connection


def initialize_database(db_path: str | Path) -> None:
    """Tạo schema trên DB riêng chưa tồn tại; DB đã có chỉ được xác minh.

    Cấm baseline ngay trước mọi mkdir/connect. Giữ return None của notebook,
    bỏ các print dùng cho demo. Không tự sửa schema/database không hợp lệ.
    """
    path = validate_database_path(db_path, for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with closing(connect_database(path, readonly=True)) as connection:
            rows = connection.execute(
                "SELECT person_id, embedding, num_images, model_version, embedding_sha256 "
                "FROM FaceDatabase ORDER BY person_id"
            ).fetchall()
            for row in rows:
                row_to_person(row)
        return
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.execute(CREATE_TABLE_SQL)
        check_schema(connection)


def embedding_to_blob(embedding: np.ndarray) -> bytes:
    """Mã hóa float32 đơn vị (512,) thành đúng 2048 byte little-endian."""
    validate_database_embedding(embedding)
    blob = embedding.astype("<f4", copy=False).tobytes(order="C")
    if len(blob) != BLOB_BYTES:
        raise ValueError("BLOB phải có đúng 2048 bytes.")
    return blob


def blob_to_embedding(blob: bytes | bytearray | memoryview) -> np.ndarray:
    """Giải mã thành float32 (512,) có bộ nhớ riêng, không tự sửa norm."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("BLOB phải là dữ liệu bytes.")
    blob = bytes(blob)
    if len(blob) != BLOB_BYTES:
        raise ValueError("BLOB phải có đúng 2048 bytes.")
    embedding = np.frombuffer(blob, dtype="<f4").astype(np.float32, copy=True)
    validate_database_embedding(embedding)
    return embedding


def validate_record_metadata(person_id: str, num_images: int, model_version: str) -> None:
    """Giữ validation ID, số ảnh INTEGER dương và version của notebook 06."""
    validate_person_id(person_id)
    if isinstance(num_images, (bool, np.bool_)) or not isinstance(num_images, (int, np.integer)):
        raise TypeError("num_images phải là số nguyên dương, không phải bool.")
    if not 0 < int(num_images) <= 2**63 - 1:
        raise ValueError("num_images phải nguyên dương trong giới hạn INTEGER của SQLite.")
    if not isinstance(model_version, str):
        raise TypeError("model_version phải là chuỗi.")
    if not model_version.strip() or model_version != model_version.strip():
        raise ValueError("model_version không rỗng và không có khoảng trắng đầu/cuối.")


def row_to_person(row: tuple, expected_model_version: str | None = None) -> dict[str, object]:
    """Kiểm tra metadata, checksum, BLOB và version tùy chọn trước khi trả bản ghi."""
    person_id, blob, num_images, model_version, checksum = row
    validate_record_metadata(person_id, num_images, model_version)
    if not isinstance(blob, bytes):
        raise TypeError("Cột embedding không phải BLOB.")
    if hashlib.sha256(blob).hexdigest() != checksum:
        raise ValueError("Checksum BLOB không khớp; không sử dụng embedding.")
    embedding = blob_to_embedding(blob)
    if expected_model_version is not None and model_version != expected_model_version:
        raise ValueError("model_version không tương thích cấu hình embedding hiện tại.")
    return {"person_id": person_id, "embedding": embedding,
            "num_images": num_images, "model_version": model_version}


def add_person(db_path: str | Path, person_id: str, embedding: np.ndarray,
               num_images: int, model_version: str) -> None:
    """INSERT riêng; ID trùng báo ValueError, không upsert/ghi đè."""
    validate_record_metadata(person_id, num_images, model_version)
    blob = embedding_to_blob(embedding)
    checksum = hashlib.sha256(blob).hexdigest()
    with closing(connect_database(db_path)) as connection:
        try:
            with connection:
                connection.execute(
                    "INSERT INTO FaceDatabase "
                    "(person_id, embedding, num_images, model_version, embedding_sha256) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (person_id, blob, int(num_images), model_version, checksum),
                )
        except sqlite3.IntegrityError as error:
            if connection.execute(
                "SELECT 1 FROM FaceDatabase WHERE person_id = ?", (person_id,)
            ).fetchone():
                raise ValueError(f"person_id đã tồn tại: {person_id}; không ghi đè.") from error
            raise


def _validate_expected_version(expected_model_version: str) -> None:
    if not isinstance(expected_model_version, str):
        raise TypeError("expected_model_version phải là chuỗi.")
    if not expected_model_version.strip():
        raise ValueError("expected_model_version không được rỗng.")


def get_person(db_path: str | Path, person_id: str,
               expected_model_version: str) -> dict[str, object] | None:
    """Đọc một người hoặc None; version tường minh thay MODEL_VERSION toàn cục."""
    validate_person_id(person_id)
    _validate_expected_version(expected_model_version)
    with closing(connect_database(db_path, readonly=True)) as connection:
        row = connection.execute(
            "SELECT person_id, embedding, num_images, model_version, embedding_sha256 "
            "FROM FaceDatabase WHERE person_id = ?", (person_id,)
        ).fetchone()
    return None if row is None else row_to_person(row, expected_model_version=expected_model_version)


def list_people(db_path: str | Path) -> list[dict[str, object]]:
    """Liệt kê metadata theo person_id; giữ việc không giải mã BLOB của notebook."""
    with closing(connect_database(db_path, readonly=True)) as connection:
        rows = connection.execute(
            "SELECT person_id, num_images, model_version FROM FaceDatabase ORDER BY person_id"
        ).fetchall()
    results = []
    for person_id, num_images, model_version in rows:
        validate_record_metadata(person_id, num_images, model_version)
        results.append({"person_id": person_id, "num_images": num_images,
                        "model_version": model_version})
    return results


def update_person(db_path: str | Path, person_id: str, embedding: np.ndarray,
                  num_images: int, model_version: str) -> bool:
    """UPDATE một ID đã tồn tại; thiếu ID báo KeyError, không tự INSERT."""
    validate_record_metadata(person_id, num_images, model_version)
    blob = embedding_to_blob(embedding)
    with closing(connect_database(db_path)) as connection:
        with connection:
            cursor = connection.execute(
                "UPDATE FaceDatabase SET embedding = ?, num_images = ?, "
                "model_version = ?, embedding_sha256 = ? WHERE person_id = ?",
                (blob, int(num_images), model_version, hashlib.sha256(blob).hexdigest(), person_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Không tìm thấy person_id để cập nhật: {person_id}")
    return True


def delete_person(db_path: str | Path, person_id: str) -> bool:
    """DELETE một ID đã tồn tại; thiếu ID báo KeyError."""
    validate_person_id(person_id)
    with closing(connect_database(db_path)) as connection:
        with connection:
            cursor = connection.execute(
                "DELETE FROM FaceDatabase WHERE person_id = ?", (person_id,)
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Không tìm thấy person_id để xóa: {person_id}")
    return True


def registered_from_rows(rows: list[tuple], expected_model_version: str) -> dict[str, np.ndarray]:
    """Giữ reader notebook 07: cùng version, đúng 3 ảnh/người và không trùng ID."""
    if not isinstance(expected_model_version, str):
        raise TypeError("expected_model_version phải là chuỗi.")
    if not expected_model_version.strip():
        raise ValueError("expected_model_version không được rỗng.")
    if not rows:
        raise ValueError("Database rỗng; không có danh tính để matching.")
    registered = {}
    for index, row in enumerate(rows):
        try:
            person = row_to_person(row, expected_model_version=expected_model_version)
            person_id = person["person_id"]
            if person_id in registered:
                raise ValueError(f"person_id bị lặp: {person_id}")
            if person["num_images"] != ENROLLMENT_IMAGES_PER_PERSON:
                raise ValueError("Số ảnh enrollment không khớp cấu hình 3 ảnh/người.")
        except (TypeError, ValueError) as error:
            raise type(error)(f"Bản ghi SQLite tại chỉ số {index}: {error}") from error
        registered[person_id] = person["embedding"]
    return registered


def load_registered_embeddings(db_path: str | Path,
                               expected_model_version: str) -> dict[str, np.ndarray]:
    """Đọc toàn bộ registry theo ID, xác minh schema/checksum/norm/version, không ghi."""
    path = validate_database_path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file SQLite: {path}")
    with closing(connect_database(path, readonly=True)) as connection:
        rows = connection.execute(
            "SELECT person_id, embedding, num_images, model_version, embedding_sha256 "
            "FROM FaceDatabase ORDER BY person_id"
        ).fetchall()
    return registered_from_rows(rows, expected_model_version)
