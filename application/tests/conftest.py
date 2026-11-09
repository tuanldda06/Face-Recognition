# Thêm thư mục gốc dự án vào đường dẫn import để pytest tìm được các package.
# Lấy Path từ pathlib để biểu diễn đường dẫn và thao tác file bằng đối tượng Path.
from pathlib import Path
# Nạp sys để truy cập đường dẫn import, tham số và luồng báo lỗi của Python.
import sys

# Từ file trong application/tests đi lên ba cấp: parents[0] là tests, [1] là application, [2] là thư mục dự án.
ROOT = Path(__file__).resolve().parents[2]
# Chỉ chạy khối bên dưới khi str(ROOT) not in sys.path cho kết quả đúng.
if str(ROOT) not in sys.path:
    # Gọi sys.path.insert(...) với 0, str(ROOT); dữ liệu/tham số: 0, str(ROOT) (đổi sang chuỗi bằng str(...)).
    sys.path.insert(0, str(ROOT))
