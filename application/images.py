# Chuyển dữ liệu file thành ảnh RGB và chặn ảnh sai định dạng hoặc vượt giới hạn.
# Lấy BytesIO từ io để đọc/ghi byte ảnh trong bộ nhớ bằng BytesIO.
from io import BytesIO
# Nạp warnings để điều khiển cách xử lý cảnh báo của thư viện.
import warnings

# Nạp numpy dưới tên ngắn np để tính toán mảng ảnh, vector và ma trận.
import numpy as np
# Lấy Image, ImageOps, UnidentifiedImageError từ PIL để đọc/ghi ảnh, xử lý hướng EXIF, chọn font và vẽ lên ảnh bằng Pillow.
from PIL import Image, ImageOps, UnidentifiedImageError

# Lấy Settings từ .config; dấu . chỉ module cùng package.
from .config import Settings
# Lấy ApplicationError từ .errors; dấu . chỉ module cùng package.
from .errors import ApplicationError


# Ảnh hợp lệ phải là mảng uint8 có dạng (cao, rộng, 3 kênh đỏ-xanh lá-xanh dương).
# Định nghĩa hàm validate_rgb(image, settings).
def validate_rgb(image, settings: Settings):
    # Chỉ chạy khối bên dưới khi not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or (image.dtype != np.uint8) cho kết quả đúng.
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            # Bổ sung trường hợp được chấp nhận bằng or: or image.shape[2] != 3 or image.dtype != np.uint8).
            or image.shape[2] != 3 or image.dtype != np.uint8):
        # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'invalid_image', 'Cần ảnh RGB uint8 (H,W,3).'.
        raise ApplicationError("invalid_image", "Cần ảnh RGB uint8 (H,W,3).")
    # Tách kết quả theo thứ tự và gán cho (height, width): lấy phần tử/lát cắt image.shape[:2] từ image.shape.
    height, width = image.shape[:2]
    # Kiểm tra giới hạn chiều ảnh và số pixel bằng validate_size(...); dữ liệu/tham số: width, height, settings.
    validate_size(width, height, settings)
    # Kết thúc hàm và trả image (ảnh đang xử lý).
    return image


# Kiểm tra chiều ảnh và tổng số pixel để tránh ảnh quá nhỏ hoặc tốn quá nhiều bộ nhớ.
# Định nghĩa hàm validate_size(width, height, settings).
def validate_size(width, height, settings):
    # Chỉ chạy khối bên dưới khi min(width, height) < 20 cho kết quả đúng.
    if min(width, height) < 20:
        # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'image_too_small', 'Ảnh phải có chiều rộng và cao ít nhất 20 pixel.'.
        raise ApplicationError("image_too_small", "Ảnh phải có chiều rộng và cao ít nhất 20 pixel.")
    # Chỉ chạy khối bên dưới khi max(width, height) > settings.max_dimension or width * height > settings.max_pixels cho kết quả đúng.
    if max(width, height) > settings.max_dimension or width * height > settings.max_pixels:
        # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'image_too_large', 'Ảnh vượt giới hạn kích thước/pixel.', 413.
        raise ApplicationError("image_too_large", "Ảnh vượt giới hạn kích thước/pixel.", 413)


# Đọc byte ảnh, kiểm tra định dạng, sửa chiều theo EXIF rồi chuyển về RGB.
# Định nghĩa hàm decode_image(data, settings).
def decode_image(data: bytes, settings: Settings):
    # Chỉ chạy khối bên dưới khi len(data) > settings.max_file_bytes cho kết quả đúng.
    if len(data) > settings.max_file_bytes:
        # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'file_too_large', 'Mỗi ảnh tối đa 8 MiB.', 413.
        raise ApplicationError("file_too_large", "Mỗi ảnh tối đa 8 MiB.", 413)
    # Chỉ chạy khối bên dưới khi not data cho kết quả đúng.
    if not data:
        # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'invalid_image', 'File ảnh rỗng.'.
        raise ApplicationError("invalid_image", "File ảnh rỗng.")
    # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
    try:
        # Mở khối with để gọi warnings.catch_warnings(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with warnings.catch_warnings():
            # Ảnh nén nhỏ nhưng bung ra quá lớn cũng bị coi là lỗi để bảo vệ bộ nhớ.
            # Nâng cảnh báo ảnh giải nén quá lớn thành ngoại lệ để nhánh except có thể từ chối ảnh.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            # Mở khối with để mở dữ liệu ảnh bằng Pillow bằng Image.open(...); dữ liệu/tham số: BytesIO(data) (tạo luồng byte trong bộ nhớ thay vì file trên ổ đĩa bằng BytesIO(...)); gọi tài nguyên là source; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
            with Image.open(BytesIO(data)) as source:
                # Chỉ chạy khối bên dưới khi source.format not in {'JPEG', 'PNG', 'WEBP'} or getattr(source, 'n_frames', 1) != 1 cho kết quả đúng.
                if source.format not in {"JPEG", "PNG", "WEBP"} or getattr(source, "n_frames", 1) != 1:
                    # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'unsupported_image', 'Chỉ nhận JPEG, PNG hoặc WebP tĩnh.', 415.
                    raise ApplicationError("unsupported_image", "Chỉ nhận JPEG, PNG hoặc WebP tĩnh.", 415)
                # Kiểm tra giới hạn chiều ảnh và số pixel bằng validate_size(...); dữ liệu/tham số: *source.size, settings.
                validate_size(*source.size, settings)
                # Buộc Pillow giải nén/đọc hết pixel ngay khi file còn mở; nhờ đó lỗi file hỏng được bắt tại đây.
                source.load()
                # EXIF lưu hướng chụp của điện thoại; sửa chiều trước khi đưa ảnh vào model.
                # Xoay/lật theo EXIF, đổi về 3 kênh RGB, rồi tạo mảng uint8 với mỗi kênh trong khoảng 0–255.
                rgb = np.array(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8)
    # Bắt lỗi ApplicationError để xử lý trong khối bên dưới.
    except ApplicationError:
        # Ném lại lỗi đang xử lý để phía gọi biết thao tác chưa thành công.
        raise
    # Bắt lỗi (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.Decompr… và đặt tên error để xử lý trong khối bên dưới.
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            # Tiếp phần biểu thức nhiều dòng: Image.DecompressionBombWarning (đọc thuộc tính).
            Image.DecompressionBombWarning) as error:
        # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'invalid_image', 'Không giải mã được ảnh hợp lệ.'; from giữ lại lỗi gốc để truy nguyên.
        raise ApplicationError("invalid_image", "Không giải mã được ảnh hợp lệ.") from error
    # Kết thúc hàm và trả kiểm tra kiểu mảng, số kênh và kích thước RGB bằng validate_rgb(...); dữ liệu/tham số: rgb, settings.
    return validate_rgb(rgb, settings)
