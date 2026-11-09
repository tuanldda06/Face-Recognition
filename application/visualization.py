# Vẽ khung và nhãn nhận diện lên bản sao của ảnh, sau đó trả về mảng RGB.
# Lấy Path từ pathlib để biểu diễn đường dẫn và thao tác file bằng đối tượng Path.
from pathlib import Path

# Nạp numpy dưới tên ngắn np để tính toán mảng ảnh, vector và ma trận.
import numpy as np
# Lấy Image, ImageDraw, ImageFont từ PIL để đọc/ghi ảnh, xử lý hướng EXIF, chọn font và vẽ lên ảnh bằng Pillow.
from PIL import Image, ImageDraw, ImageFont


# Mặt đã nhận diện có khung xanh; mặt chưa xác định có khung cam và nhãn Unknown.
# Định nghĩa hàm annotate(image_rgb, faces).
def annotate(image_rgb, faces):
    # Sao chép mảng RGB rồi tạo ảnh Pillow; vẽ khung không làm thay đổi ảnh gốc của bên gọi.
    image = Image.fromarray(image_rgb.copy())
    # Gán cho draw: tạo công cụ vẽ lên ảnh bằng ImageDraw.Draw(...); dữ liệu/tham số: image.
    draw = ImageDraw.Draw(image)
    # Gán cho size: lấy giá trị lớn nhất bằng max(...); dữ liệu/tham số: 13, min(22, image.width // 40) (lấy giá trị nhỏ nhất bằng min(...)).
    size = max(13, min(22, image.width // 40))
    # Gán cho font: nạp font mặc định bằng ImageFont.load_default(...); dữ liệu/tham số: size=size.
    font = ImageFont.load_default(size=size)
    # Lặp qua (Path('C:/Windows/Fonts/segoeui.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.…; mỗi lượt gán phần tử cho path.
    for path in (Path("C:/Windows/Fonts/segoeui.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        # Chỉ chạy khối bên dưới khi path.is_file() cho kết quả đúng.
        if path.is_file():
            # Gán cho font: nạp font TrueType để vẽ nhãn bằng ImageFont.truetype(...); dữ liệu/tham số: str(path) (đổi sang chuỗi bằng str(...)), size.
            font = ImageFont.truetype(str(path), size)
            # Thoát ngay khỏi vòng lặp gần nhất để tiếp tục phần sau vòng lặp.
            break
    # Lặp qua faces; mỗi lượt gán phần tử cho face.
    for face in faces:
        # So trạng thái với recognized; known là True khi mặt được ghép với một hồ sơ hợp lệ.
        known = face["status"] == "recognized"
        # Chọn bộ ba RGB xanh cho mặt đã biết; chọn cam khi chưa nhận ra danh tính.
        color = (22, 163, 74) if known else (234, 110, 30)
        # Tách khung mặt thành tọa độ trái, trên, phải, dưới để vẽ hình chữ nhật.
        x1, y1, x2, y2 = face["bbox"]
        # Gán cho name: chọn face['person_id'] nếu known; ngược lại chọn 'Unknown'.
        name = face["person_id"] if known else "Unknown"
        # Gán cho score: chọn 'n/a' nếu face['similarity'] is None; ngược lại chọn f'{face['similarity']:.3f}'.
        score = "n/a" if face["similarity"] is None else f'{face["similarity"]:.3f}'
        # det là độ tin cậy phát hiện mặt; cos là độ tương đồng vector với hồ sơ, không phải cùng một chỉ số.
        # Gán cho label: chuỗi f-string: chèn giá trị biểu thức nằm trong {...} vào nội dung.
        label = f'{name} | det {face["confidence"]:.2f} | cos {score}'
        # Gọi draw.rectangle(...) với (x1, y1, x2 - 1, y2 - 1); dữ liệu/tham số: (x1, y1, x2 - 1, y2 - 1), outline=color, width=3.
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=3)
        # Gán cho y: lấy giá trị lớn nhất bằng max(...); dữ liệu/tham số: 0, y1 - size - 8.
        y = max(0, y1 - size - 8)
        # Gán cho bounds: gọi draw.textbbox(...) với (x1, y), label; dữ liệu/tham số: (x1, y), label, font=font.
        bounds = draw.textbbox((x1, y), label, font=font)
        # Gọi draw.rectangle(...) với bounds; dữ liệu/tham số: bounds, fill=color.
        draw.rectangle(bounds, fill=color)
        # Gọi draw.text(...) với (x1, y), label; dữ liệu/tham số: (x1, y), label, font=font, fill='white'.
        draw.text((x1, y), label, font=font, fill="white")
    # Đổi ảnh Pillow đã vẽ trở lại mảng NumPy RGB để Gradio hoặc bên gọi hiển thị.
    return np.asarray(image)
