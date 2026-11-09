# Lưu vector đặc trưng và tên ở hai file SQLite; giao dịch giúp cập nhật chúng đồng bộ.
# Lấy closing từ contextlib để quản lý tài nguyên bằng with, yield và thao tác dọn tự động.
from contextlib import closing
# Nạp hashlib để tính mã băm để kiểm tra nội dung file/ảnh/vector.
import hashlib
# Nạp sqlite3 để đọc, ghi và quản lý giao dịch SQLite.
import sqlite3

# Lấy database với bí danh db từ FaceNet.
from FaceNet import database as db
# Lấy ApplicationError từ .errors; dấu . chỉ module cùng package.
from .errors import ApplicationError


# Gán cho NAME_SQL: chuỗi 'CREATE TABLE IF NOT EXISTS EnrollmentNames (person_id TEXT PRIM….
NAME_SQL = ("CREATE TABLE IF NOT EXISTS EnrollmentNames ("
            # Tiếp chuỗi/dữ liệu của câu lệnh trên; Python tự nối các chuỗi liền kề trong ngoặc.
            "person_id TEXT PRIMARY KEY NOT NULL, display_name TEXT NOT NULL)")


# Tạo đường dẫn file .names.sqlite cạnh kho vector và kiểm tra đường dẫn hợp lệ.
# Định nghĩa hàm metadata_path(path).
def metadata_path(path):
    # Gán cho candidate: tạo đường dẫn cùng thư mục nhưng đổi tên file qua path.with_name(...); dữ liệu/tham số: path.name + '.names.sqlite'.
    candidate = path.with_name(path.name + ".names.sqlite")
    # Gán cho result: gọi db.validate_database_path(...) với candidate; dữ liệu/tham số: candidate, for_write=True.
    result = db.validate_database_path(candidate, for_write=True)
    # Chỉ chạy khối bên dưới khi result.exists() and result.stat().st_nlink != 1 cho kết quả đúng.
    if result.exists() and result.stat().st_nlink != 1:
        # Phát sinh lỗi: gọi ValueError(...) với 'Hardlinked name registry'; dữ liệu/tham số: 'Hardlinked name registry'.
        raise ValueError("Hardlinked name registry")
    # Kết thúc hàm và trả result (kết quả xử lý).
    return result


# Kiểm tra bảng tên đúng cấu trúc dự kiến trước khi đọc hoặc cập nhật.
# Định nghĩa hàm _check_names(connection, schema). Mặc định: schema='main'. Sau dấu * phải truyền theo tên: allow_empty=False.
def _check_names(connection, schema="main", *, allow_empty=False):
    # Gán cho objects: lấy toàn bộ hàng kết quả SQL qua connection.execute(f"SELECT type,name FROM {schema}.sqlite_master WHERE name NOT LIKE 'sq…(...).
    objects = connection.execute(
        # Tiếp phần biểu thức nhiều dòng: chuỗi f-string: chèn giá trị biểu thức nằm trong {...} vào nội dung.
        f"SELECT type,name FROM {schema}.sqlite_master WHERE name NOT LIKE 'sqlite_%'"
    # Kết thúc lời gọi SQL rồi lấy toàn bộ các hàng kết quả bằng fetchall().
    ).fetchall()
    # ATTACH có thể đã tạo file tên rỗng trước khi lần commit đầu bị lỗi.
    # File chưa chứa ID/tên nên có thể coi như chưa được khởi tạo.
    # Chỉ chạy khối bên dưới khi allow_empty and (not objects) cho kết quả đúng.
    if allow_empty and not objects:
        # Kết thúc hàm và trả False (sai/tắt).
        return False
    # Gán cho columns: lấy toàn bộ hàng kết quả SQL qua connection.execute(f'PRAGMA {schema}.table_info(EnrollmentNames)').fetchall(...).
    columns = connection.execute(f"PRAGMA {schema}.table_info(EnrollmentNames)").fetchall()
    # Chỉ chạy khối bên dưới khi objects != [('table', 'EnrollmentNames')] or [(r[1], r[2], r[3], r[5]) for r in columns] != [('person_id', 'TEXT', 1, 1), ('… cho kết quả đúng.
    if (objects != [("table", "EnrollmentNames")]
            # Bổ sung trường hợp được chấp nhận bằng or: or [(r[1], r[2], r[3], r[5]) for r in columns] !=.
            or [(r[1], r[2], r[3], r[5]) for r in columns] !=
            # Tiếp phần biểu thức nhiều dòng: danh sách [('person_id', 'TEXT', 1, 1), ('display_name', 'TEXT', 1, 0)] ([] giữ các phần tử theo thứ tự).
            [("person_id", "TEXT", 1, 1), ("display_name", "TEXT", 1, 0)]):
        # Phát sinh lỗi: gọi ValueError(...) với 'Unexpected names schema'; dữ liệu/tham số: 'Unexpected names schema'.
        raise ValueError("Unexpected names schema")
    # Gán cho sql: lấy phần tử/lát cắt connection.execute(f"SELECT sql FROM {schema}.sqlite_master WHERE name='EnrollmentNames'"… từ connection.execute(f"SELECT sql FROM {schema}.sqlite_master WHERE name='EnrollmentNames'"….
    sql = connection.execute(
        # Tiếp phần biểu thức nhiều dòng: chuỗi f-string: chèn giá trị biểu thức nằm trong {...} vào nội dung.
        f"SELECT sql FROM {schema}.sqlite_master WHERE name='EnrollmentNames'").fetchone()[0]
    # Gán cho canonical: hàm ngắn lambda nhận s và trả ' '.join(s.upper().replace('IF NOT EXISTS ', '').split()).
    canonical = lambda s: " ".join(s.upper().replace("IF NOT EXISTS ", "").split())
    # Chỉ chạy khối bên dưới khi canonical(sql) != canonical(NAME_SQL) cho kết quả đúng.
    if canonical(sql) != canonical(NAME_SQL):
        # Phát sinh lỗi: gọi ValueError(...) với 'Unexpected names SQL'; dữ liệu/tham số: 'Unexpected names SQL'.
        raise ValueError("Unexpected names SQL")
    # Kết thúc hàm và trả True (đúng/bật).
    return True


# Đọc ánh xạ ID → tên bằng kết nối chỉ đọc; chưa có file tên thì trả dict rỗng.
# Định nghĩa hàm read_names(path).
def read_names(path):
    # Gán cho path: xác định file SQLite chứa tên đi kèm bằng metadata_path(...); dữ liệu/tham số: path.
    path = metadata_path(path)
    # Chỉ chạy khối bên dưới khi not path.exists() cho kết quả đúng.
    if not path.exists():
        # Kết thúc hàm và trả từ điển {} (ánh xạ khóa → giá trị).
        return {}
    # Mở khối with để bảo đảm tài nguyên được close khi kết thúc khối with bằng closing(...); dữ liệu/tham số: sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) (mở kết nối cơ sở dữ liệu SQLite bằng sqlite3.connect(...)); gọi tài nguyên là connection; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        # Chỉ chạy khối bên dưới khi not _check_names(connection, allow_empty=True) cho kết quả đúng.
        if not _check_names(connection, allow_empty=True):
            # Kết thúc hàm và trả từ điển {} (ánh xạ khóa → giá trị).
            return {}
        # Kết thúc hàm và trả tạo từ điển khóa–giá trị bằng dict(...); dữ liệu/tham số: connection.execute('SELECT person_id,display_name FROM EnrollmentNames') (thực hiện câu lệnh SQL và truyền các tham số đi kèm qua connection.execute(...)).
        return dict(connection.execute("SELECT person_id,display_name FROM EnrollmentNames"))


# Gắn kho tên vào kết nối hiện tại; yêu cầu chế độ ghi hỗ trợ giao dịch hai file.
# Định nghĩa hàm _attach_names(connection, path).
def _attach_names(connection, path):
    # Không tự đổi chế độ nhật ký trên cơ sở dữ liệu sẵn có của người dùng.
    # Chỉ chạy khối bên dưới khi connection.execute('PRAGMA main.journal_mode').fetchone()[0].lower() != 'delete' cho kết quả đúng.
    if connection.execute("PRAGMA main.journal_mode").fetchone()[0].lower() != "delete":
        # Phát sinh lỗi: gọi ValueError(...) với 'Atomic enrollment requires DELETE…; dữ liệu/tham số: 'Atomic enrollment requires DELETE journal mode'.
        raise ValueError("Atomic enrollment requires DELETE journal mode")
    # Thực hiện câu lệnh SQL và truyền các tham số đi kèm qua connection.execute(...); dữ liệu/tham số: 'ATTACH DATABASE ? AS names', (str(metadata_path(path)),).
    connection.execute("ATTACH DATABASE ? AS names", (str(metadata_path(path)),))
    # Chỉ chạy khối bên dưới khi connection.execute('PRAGMA names.journal_mode').fetchone()[0].lower() != 'delete' cho kết quả đúng.
    if connection.execute("PRAGMA names.journal_mode").fetchone()[0].lower() != "delete":
        # Phát sinh lỗi: gọi ValueError(...) với 'Atomic name storage requires DELE…; dữ liệu/tham số: 'Atomic name storage requires DELETE journal mode'.
        raise ValueError("Atomic name storage requires DELETE journal mode")
    # Chỉ chạy khối bên dưới khi connection.execute('PRAGMA main.synchronous').fetchone()[0] < 2 or connection.execute('PRAGMA names.synchronous').fetchone()… cho kết quả đúng.
    if (connection.execute("PRAGMA main.synchronous").fetchone()[0] < 2
            # Bổ sung trường hợp được chấp nhận bằng or: or connection.execute("PRAGMA names.synchronous").fetchone()[0] < 2).
            or connection.execute("PRAGMA names.synchronous").fetchone()[0] < 2):
        # Phát sinh lỗi: gọi ValueError(...) với 'Atomic enrollment requires FULL s…; dữ liệu/tham số: 'Atomic enrollment requires FULL synchronous mode'.
        raise ValueError("Atomic enrollment requires FULL synchronous mode")


# Ghi vector và tên trong cùng giao dịch; bất kỳ bước nào lỗi thì hoàn tác cả hai.
# Định nghĩa hàm commit_identity(path, person_id, display_name, embedding, version). Sau dấu * phải truyền theo tên: num_images (bắt buộc).
def commit_identity(path, person_id, display_name, embedding, version, *, num_images):
    # Gọi db.validate_record_metadata(...) với person_id, num_images, version; dữ liệu/tham số: person_id, num_images, version.
    db.validate_record_metadata(person_id, num_images, version)
    # Chỉ chạy khối bên dưới khi person_id == 'Unknown' cho kết quả đúng.
    if person_id == "Unknown":
        # Phát sinh lỗi: gọi ValueError(...) với 'Reserved identity'; dữ liệu/tham số: 'Reserved identity'.
        raise ValueError("Reserved identity")
    # Kiểm tra và mã hóa vector embedding thành byte để ghi cột BLOB trong SQLite.
    blob = db.embedding_to_blob(embedding)
    # Mở khối with để bảo đảm tài nguyên được close khi kết thúc khối with bằng closing(...); dữ liệu/tham số: db.connect_database(path) (gọi db.connect_database(...) với path); gọi tài nguyên là connection; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with closing(db.connect_database(path)) as connection:
        # Gắn kho tên vào kết nối của kho vector bằng _attach_names(...); dữ liệu/tham số: connection, path.
        _attach_names(connection, path)
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Lấy quyền ghi ngay từ đầu; các lệnh sau thuộc cùng một giao dịch.
            # Bắt đầu giao dịch SQLite và giữ quyền ghi ngay; các lệnh INSERT/DELETE sau đó cùng thành công hoặc cùng bị hoàn tác.
            connection.execute("BEGIN IMMEDIATE")
            # Gọi db.check_schema(...) với connection; dữ liệu/tham số: connection.
            db.check_schema(connection)
            # Gán cho rows: lấy toàn bộ hàng kết quả SQL qua connection.execute('SELECT person_id,embedding,num_images,model_version,embedding_sha256 …(...).
            rows = connection.execute(
                # Tiếp phần biểu thức nhiều dòng: chuỗi 'SELECT person_id,embedding,num_images,model_version,embedding_s….
                "SELECT person_id,embedding,num_images,model_version,embedding_sha256 FROM FaceDatabase"
            # Kết thúc lời gọi SQL rồi lấy toàn bộ các hàng kết quả bằng fetchall().
            ).fetchall()
            # Lặp qua rows; mỗi lượt gán phần tử cho row.
            for row in rows:
                # Gán cho person: gọi db.row_to_person(...) với row; dữ liệu/tham số: row, expected_model_version=version.
                person = db.row_to_person(row, expected_model_version=version)
                # Chỉ chạy khối bên dưới khi person['person_id'] == 'Unknown' cho kết quả đúng.
                if person["person_id"] == "Unknown":
                    # Phát sinh lỗi: gọi ValueError(...) với 'Registry contains reserved identi…; dữ liệu/tham số: 'Registry contains reserved identity'.
                    raise ValueError("Registry contains reserved identity")
                # Chỉ chạy khối bên dưới khi person['person_id'] == person_id cho kết quả đúng.
                if person["person_id"] == person_id:
                    # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'duplicate_person', 'person_id đã tồn tại; không ghi đè.', 409.
                    raise ApplicationError("duplicate_person", "person_id đã tồn tại; không ghi đè.", 409)
            # Thực hiện câu lệnh SQL và truyền các tham số đi kèm qua connection.execute(...); dữ liệu/tham số: NAME_SQL.replace('EnrollmentNames', 'names.EnrollmentNames', 1) (thay chuỗi hoặc thay file đích tùy đối tượng qua NAME_SQL.replace(...)).
            connection.execute(NAME_SQL.replace("EnrollmentNames", "names.EnrollmentNames", 1))
            # Kiểm tra cấu trúc bảng tên bằng _check_names(...); dữ liệu/tham số: connection, 'names'.
            _check_names(connection, "names")
            # Thực hiện câu lệnh SQL và truyền các tham số đi kèm qua connection.execute(...); dữ liệu/tham số: 'INSERT INTO FaceDatabase VALUES (?,?,?,?,?)', (person_id, blob, num_images, version, hashlib.sha256(blob).hexdigest()).
            connection.execute("INSERT INTO FaceDatabase VALUES (?,?,?,?,?)",
                               # Tiếp phần biểu thức nhiều dòng: bộ giá trị (person_id, blob, num_images, version, hashlib.sha256(blob).hexdigest()) (tuple giữ thứ tự, không gán lại từng phần tử).
                               (person_id, blob, num_images, version, hashlib.sha256(blob).hexdigest()))
            # Thực hiện câu lệnh SQL và truyền các tham số đi kèm qua connection.execute(...); dữ liệu/tham số: 'INSERT INTO names.EnrollmentNames VALUES (?,?)', (person_id, display_name).
            connection.execute("INSERT INTO names.EnrollmentNames VALUES (?,?)", (person_id, display_name))
            # Commit xác nhận toàn bộ thay đổi; nếu chưa commit thì chưa coi là lưu thành công.
            # Xác nhận toàn bộ thay đổi trong giao dịch; sau dòng này dữ liệu mới được coi là đã lưu thành công.
            connection.commit()
        # Bắt lỗi BaseException để xử lý trong khối bên dưới.
        except BaseException:
            # Rollback hoàn tác toàn bộ giao dịch khi có lỗi, kể cả lỗi ở file tên đi kèm.
            # Hoàn tác giao dịch do có lỗi; hủy cả thay đổi vector và tên thay vì giữ hồ sơ dở dang.
            connection.rollback()
            # Ném lại lỗi đang xử lý để phía gọi biết thao tác chưa thành công.
            raise


# Xóa vector và tên đồng bộ để ID có thể dùng lại sau khi xóa.
# Định nghĩa hàm delete_identity(path, person_id).
def delete_identity(path, person_id):
    # Chuỗi tài liệu (docstring) mô tả cách dùng; Python lưu nội dung này trong __doc__.
    """Remove names with the identity, including ID reuse after deletion."""
    # Chỉ chạy khối bên dưới khi not metadata_path(path).exists() cho kết quả đúng.
    if not metadata_path(path).exists():
        # Kết thúc hàm và trả gọi db.delete_person(...) với path, person_id; dữ liệu/tham số: path, person_id.
        return db.delete_person(path, person_id)
    # Mở khối with để bảo đảm tài nguyên được close khi kết thúc khối with bằng closing(...); dữ liệu/tham số: db.connect_database(path) (gọi db.connect_database(...) với path); gọi tài nguyên là connection; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with closing(db.connect_database(path)) as connection:
        # Gắn kho tên vào kết nối của kho vector bằng _attach_names(...); dữ liệu/tham số: connection, path.
        _attach_names(connection, path)
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Lấy quyền ghi ngay từ đầu; các lệnh sau thuộc cùng một giao dịch.
            # Bắt đầu giao dịch SQLite và giữ quyền ghi ngay; các lệnh INSERT/DELETE sau đó cùng thành công hoặc cùng bị hoàn tác.
            connection.execute("BEGIN IMMEDIATE")
            # Gán cho has_names: kiểm tra cấu trúc bảng tên bằng _check_names(...); dữ liệu/tham số: connection, 'names', allow_empty=True.
            has_names = _check_names(connection, "names", allow_empty=True)
            # Gán cho cursor: thực hiện câu lệnh SQL và truyền các tham số đi kèm qua connection.execute(...); dữ liệu/tham số: 'DELETE FROM FaceDatabase WHERE person_id=?', (person_id,).
            cursor = connection.execute("DELETE FROM FaceDatabase WHERE person_id=?", (person_id,))
            # Chỉ chạy khối bên dưới khi cursor.rowcount != 1 cho kết quả đúng.
            if cursor.rowcount != 1:
                # Phát sinh lỗi: gọi KeyError(...) với person_id; dữ liệu/tham số: person_id.
                raise KeyError(person_id)
            # Chỉ chạy khối bên dưới khi has_names cho kết quả đúng.
            if has_names:
                # Thực hiện câu lệnh SQL và truyền các tham số đi kèm qua connection.execute(...); dữ liệu/tham số: 'DELETE FROM names.EnrollmentNames WHERE person_id=?', (person_id,).
                connection.execute("DELETE FROM names.EnrollmentNames WHERE person_id=?", (person_id,))
            # Commit xác nhận toàn bộ thay đổi; nếu chưa commit thì chưa coi là lưu thành công.
            # Xác nhận toàn bộ thay đổi trong giao dịch; sau dòng này dữ liệu mới được coi là đã lưu thành công.
            connection.commit()
        # Bắt lỗi BaseException để xử lý trong khối bên dưới.
        except BaseException:
            # Rollback hoàn tác toàn bộ giao dịch khi có lỗi, kể cả lỗi ở file tên đi kèm.
            # Hoàn tác giao dịch do có lỗi; hủy cả thay đổi vector và tên thay vì giữ hồ sơ dở dang.
            connection.rollback()
            # Ném lại lỗi đang xử lý để phía gọi biết thao tác chưa thành công.
            raise
