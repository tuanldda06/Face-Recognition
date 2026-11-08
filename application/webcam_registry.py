from contextlib import closing
import hashlib
import sqlite3

from FaceNet import database as db
from .errors import ApplicationError


NAME_SQL = ("CREATE TABLE IF NOT EXISTS EnrollmentNames ("
            "person_id TEXT PRIMARY KEY NOT NULL, display_name TEXT NOT NULL)")


def metadata_path(path):
    candidate = path.with_name(path.name + ".names.sqlite")
    result = db.validate_database_path(candidate, for_write=True)
    if result.exists() and result.stat().st_nlink != 1:
        raise ValueError("Hardlinked name registry")
    return result


def _check_names(connection, schema="main", *, allow_empty=False):
    objects = connection.execute(
        f"SELECT type,name FROM {schema}.sqlite_master WHERE name NOT LIKE 'sqlite_%'"
    ).fetchall()
    # ATTACH may have created an empty sidecar before a failed first commit.
    # It contains no identity/name and is safe to treat as uninitialized.
    if allow_empty and not objects:
        return False
    columns = connection.execute(f"PRAGMA {schema}.table_info(EnrollmentNames)").fetchall()
    if (objects != [("table", "EnrollmentNames")]
            or [(r[1], r[2], r[3], r[5]) for r in columns] !=
            [("person_id", "TEXT", 1, 1), ("display_name", "TEXT", 1, 0)]):
        raise ValueError("Unexpected names schema")
    sql = connection.execute(
        f"SELECT sql FROM {schema}.sqlite_master WHERE name='EnrollmentNames'").fetchone()[0]
    canonical = lambda s: " ".join(s.upper().replace("IF NOT EXISTS ", "").split())
    if canonical(sql) != canonical(NAME_SQL):
        raise ValueError("Unexpected names SQL")
    return True


def read_names(path):
    path = metadata_path(path)
    if not path.exists():
        return {}
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        if not _check_names(connection, allow_empty=True):
            return {}
        return dict(connection.execute("SELECT person_id,display_name FROM EnrollmentNames"))


def _attach_names(connection, path):
    # Never silently change journal modes on a user's existing database.
    if connection.execute("PRAGMA main.journal_mode").fetchone()[0].lower() != "delete":
        raise ValueError("Atomic enrollment requires DELETE journal mode")
    connection.execute("ATTACH DATABASE ? AS names", (str(metadata_path(path)),))
    if connection.execute("PRAGMA names.journal_mode").fetchone()[0].lower() != "delete":
        raise ValueError("Atomic name storage requires DELETE journal mode")
    if (connection.execute("PRAGMA main.synchronous").fetchone()[0] < 2
            or connection.execute("PRAGMA names.synchronous").fetchone()[0] < 2):
        raise ValueError("Atomic enrollment requires FULL synchronous mode")


def commit_identity(path, person_id, display_name, embedding, version, *, num_images):
    db.validate_record_metadata(person_id, num_images, version)
    if person_id == "Unknown":
        raise ValueError("Reserved identity")
    blob = db.embedding_to_blob(embedding)
    with closing(db.connect_database(path)) as connection:
        _attach_names(connection, path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            db.check_schema(connection)
            rows = connection.execute(
                "SELECT person_id,embedding,num_images,model_version,embedding_sha256 FROM FaceDatabase"
            ).fetchall()
            for row in rows:
                person = db.row_to_person(row, expected_model_version=version)
                if person["person_id"] == "Unknown":
                    raise ValueError("Registry contains reserved identity")
                if person["person_id"] == person_id:
                    raise ApplicationError("duplicate_person", "person_id đã tồn tại; không ghi đè.", 409)
            connection.execute(NAME_SQL.replace("EnrollmentNames", "names.EnrollmentNames", 1))
            _check_names(connection, "names")
            connection.execute("INSERT INTO FaceDatabase VALUES (?,?,?,?,?)",
                               (person_id, blob, num_images, version, hashlib.sha256(blob).hexdigest()))
            connection.execute("INSERT INTO names.EnrollmentNames VALUES (?,?)", (person_id, display_name))
            connection.commit()
        except BaseException:
            connection.rollback()
            raise


def delete_identity(path, person_id):
    """Remove names with the identity, including ID reuse after deletion."""
    if not metadata_path(path).exists():
        return db.delete_person(path, person_id)
    with closing(db.connect_database(path)) as connection:
        _attach_names(connection, path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            has_names = _check_names(connection, "names", allow_empty=True)
            cursor = connection.execute("DELETE FROM FaceDatabase WHERE person_id=?", (person_id,))
            if cursor.rowcount != 1:
                raise KeyError(person_id)
            if has_names:
                connection.execute("DELETE FROM names.EnrollmentNames WHERE person_id=?", (person_id,))
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
