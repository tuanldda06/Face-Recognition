# Giao diện Gradio: các hàm callback nhận thao tác và trả kết quả cho các thành phần tương ứng.
# Nạp os để làm việc với biến môi trường và thông tin hệ điều hành.
import os
# Lấy Path từ pathlib để biểu diễn đường dẫn và thao tác file bằng đối tượng Path.
from pathlib import Path

# Tắt gửi thống kê và lưu lịch sử chạy Gradio cho ảnh khuôn mặt tải lên.
# Đặt biến môi trường nếu nó chưa có giá trị bằng os.environ.setdefault(...); dữ liệu/tham số: 'GRADIO_ANALYTICS_ENABLED', 'False'.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# Nạp gradio dưới tên ngắn gr để tạo giao diện web và nối sự kiện với hàm Python.
import gradio as gr

# Lấy ApplicationError từ .errors; dấu . chỉ module cùng package.
from .errors import ApplicationError
# Lấy decode_image từ .images; dấu . chỉ module cùng package.
from .images import decode_image
# Lấy annotate từ .visualization; dấu . chỉ module cùng package.
from .visualization import annotate
# Lấy process_frame từ .webcam; dấu . chỉ module cùng package.
from .webcam import process_frame
# Lấy LABELS, POSE_COUNT, POSES từ .webcam_enrollment; dấu . chỉ module cùng package.
from .webcam_enrollment import LABELS, POSE_COUNT, POSES


# Đọc CSS UTF-8 cạnh file Python; theme bên dưới đặt màu cho cả chế độ sáng và tối.
# Gán cho UI_CSS: đọc nội dung file thành chuỗi qua Path(__file__).with_name('ui.css').read_text(...); dữ liệu/tham số: encoding='utf-8'.
UI_CSS = Path(__file__).with_name("ui.css").read_text(encoding="utf-8")
# Tạo theme Gradio Base rồi dùng .set(...) tùy chỉnh màu cho chế độ sáng/tối; UI_THEME được truyền khi gắn giao diện.
UI_THEME = gr.themes.Base(
    # Truyền ba bảng màu theme: màu chính xanh teal; màu phụ và trung tính theo slate.
    primary_hue="teal", secondary_hue="slate", neutral_hue="slate",
    # Truyền font = ('Segoe UI', 'system-ui', 'sans-serif').
    font=("Segoe UI", "system-ui", "sans-serif"),
# Kết thúc tạo theme Base, rồi nối lời gọi .set(...) để đặt các màu chi tiết.
).set(
    # Đặt nền toàn trang: màu sáng ở body_background_fill, màu tối ở thuộc tính có đuôi _dark.
    body_background_fill="#eef2f7", body_background_fill_dark="#0b1320",
    # Đặt nền chính của các vùng nội dung cho chế độ sáng và tối.
    background_fill_primary="#fff", background_fill_primary_dark="#111e30",
    # Đặt màu đường viền khối cho hai chế độ giao diện.
    block_border_color="#dbe3ee", block_border_color_dark="#263950",
    # Đặt nền nút chính khi bình thường và khi trỏ chuột lên nút (hover), ở chế độ sáng.
    button_primary_background_fill="#0f766e", button_primary_background_fill_hover="#115e59",
    # Truyền button_primary_text_color = '#fff'.
    button_primary_text_color="#fff",
    # Đặt nền nút chính và nền khi hover trong chế độ tối.
    button_primary_background_fill_dark="#5eead4", button_primary_background_fill_hover_dark="#99f6e4",
    # Truyền button_primary_text_color_dark = '#102e2c'.
    button_primary_text_color_dark="#102e2c",
# Đóng phần biểu thức/tham số nhiều dòng đang mở; các dấu ngoặc không tạo bước xử lý riêng.
)

# Mẫu HTML phần đầu trang; tiền tố f cho phép chèn POSE_COUNT vào nội dung.
# Chuỗi HERO, dòng 1: gán chuỗi f-string nhiều dòng cho HERO; ba dấu nháy kép mở nội dung HTML.
# Chuỗi HERO, dòng 2: mở thẻ <header class="studio-hero">; class dùng chọn kiểu CSS, aria-hidden ẩn phần trang trí với trình đọc màn hình nếu có.
# Chuỗi HERO, dòng 3: mở thẻ <div class="studio-brand">; class dùng chọn kiểu CSS, aria-hidden ẩn phần trang trí với trình đọc màn hình nếu có.
# Chuỗi HERO, dòng 4: mở thẻ <span class="studio-mark" aria-hidden="true">; class dùng chọn kiểu CSS, aria-hidden ẩn phần trang trí với trình đọc màn hình nếu có.
# Chuỗi HERO, dòng 5: mở ảnh vector SVG; width/height đặt kích thước, viewBox đặt hệ tọa độ.
# Chuỗi HERO, dòng 6: đặt độ dày nét, kiểu đầu nét và chỗ nối nét của SVG.
# Chuỗi HERO, dòng 7: vẽ một nét của biểu tượng SVG; d mô tả đường đi, các số là tọa độ.
# Chuỗi HERO, dòng 8: vẽ một nét của biểu tượng SVG; d mô tả đường đi, các số là tọa độ.
# Chuỗi HERO, dòng 9: đóng thẻ </svg> đã mở phía trên.
# Chuỗi HERO, dòng 10: đóng thẻ </span> đã mở phía trên.
# Chuỗi HERO, dòng 11: mở thẻ <div>; class dùng chọn kiểu CSS, aria-hidden ẩn phần trang trí với trình đọc màn hình nếu có.
# Chuỗi HERO, dòng 12: thẻ p hiển thị mô tả; class nối phần tử với kiểu trong ui.css.
# Chuỗi HERO, dòng 13: thẻ h1 hiển thị tiêu đề chính của trang.
# Chuỗi HERO, dòng 14: thẻ p hiển thị mô tả; class nối phần tử với kiểu trong ui.css.
# Chuỗi HERO, dòng 15: đóng thẻ </div> đã mở phía trên.
# Chuỗi HERO, dòng 16: đóng thẻ </div> đã mở phía trên.
# Chuỗi HERO, dòng 17: nhúng số góc POSE_COUNT bằng f-string vào nhãn ở đầu trang.
# Chuỗi HERO, dòng 18: đóng thẻ </header> đã mở phía trên.
# Gán cho HERO: chuỗi f-string: chèn giá trị biểu thức nằm trong {...} vào nội dung.
HERO = f"""
<header class="studio-hero">
  <div class="studio-brand">
    <span class="studio-mark" aria-hidden="true">
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none" stroke="currentColor"
           stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
        <path d="M10 4H5v5m17-5h5v5M5 23v5h5m17-5v5h-5"/>
        <path d="M11 13h.01M21 13h.01M16 14v5m-4 3c2 2 6 2 8 0"/>
      </svg>
    </span>
    <div>
      <p class="studio-kicker">Face Recognition Studio</p>
      <h1>Nhận diện khuôn mặt</h1>
      <p>Đăng ký qua webcam, đối chiếu bằng ảnh hoặc camera trực tiếp.</p>
    </div>
  </div>
  <span class="studio-badge">Webcam · {POSE_COUNT} góc</span>
</header>
"""

# Gán cho POSE_SHORT_LABELS: từ điển {'front': 'Nhìn thẳng', 'left': 'Quay trái', 'right': 'Quay phải', 'up': 'Ngẩng đầu'} (ánh xạ khóa → giá trị).
POSE_SHORT_LABELS = {"front": "Nhìn thẳng", "left": "Quay trái", "right": "Quay phải", "up": "Ngẩng đầu"}


# JavaScript này chạy trong trình duyệt để xin quyền camera, rồi dừng stream thử ngay.
# Chuỗi CAMERA_CHECK_JS, dòng 1: hàm JavaScript bất đồng bộ không nhận tham số; Gradio chạy hàm ở trình duyệt.
# Chuỗi CAMERA_CHECK_JS, dòng 2: bắt đầu đoạn thử mở camera; lỗi sẽ chuyển đến catch.
# Chuỗi CAMERA_CHECK_JS, dòng 3: kiểm tra trình duyệt có API camera; ?. truy cập an toàn khi mediaDevices vắng mặt.
# Chuỗi CAMERA_CHECK_JS, dòng 4: await xin quyền lấy video, không lấy âm thanh; const giữ luồng vừa mở.
# Chuỗi CAMERA_CHECK_JS, dòng 5: duyệt từng track và stop ngay để giải phóng camera sau bước thử quyền.
# Chuỗi CAMERA_CHECK_JS, dòng 6: trả thông báo cho ô trạng thái camera trong Gradio.
# Chuỗi CAMERA_CHECK_JS, dòng 7: bắt lỗi mở camera vào biến error.
# Chuỗi CAMERA_CHECK_JS, dòng 8: tạo bảng thông báo theo tên lỗi; NotAllowedError nghĩa là bị từ chối quyền.
# Chuỗi CAMERA_CHECK_JS, dòng 9: thông báo riêng cho trường hợp không tìm được thiết bị camera.
# Chuỗi CAMERA_CHECK_JS, dòng 10: thông báo camera bận hoặc không thể đọc được.
# Chuỗi CAMERA_CHECK_JS, dòng 11: tra thông báo theo tên lỗi; || dùng thông báo dự phòng khi không có mục phù hợp.
# Chuỗi CAMERA_CHECK_JS, dòng 12: dấu ngoặc đóng khối JavaScript đang mở.
# Chuỗi CAMERA_CHECK_JS, dòng 13: dấu ngoặc đóng khối JavaScript đang mở.
CAMERA_CHECK_JS = """async () => {
  try {
    if (!navigator.mediaDevices?.getUserMedia) return 'Camera cần localhost hoặc HTTPS và trình duyệt hỗ trợ getUserMedia.';
    const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
    stream.getTracks().forEach(t => t.stop());
    return 'Camera đã được cấp quyền. Bật camera bên dưới, bấm Bắt đầu nhận diện rồi nút ghi hình của camera.';
  } catch (error) {
    const messages = {NotAllowedError:'Camera bị từ chối. Mở quyền camera của trang trong trình duyệt và thử lại.',
      NotFoundError:'Không tìm thấy camera. Kiểm tra kết nối thiết bị.',
      NotReadableError:'Camera đang bận hoặc không đọc được. Đóng ứng dụng khác đang dùng camera.'};
    return messages[error.name] || ('Không mở được camera: ' + error.name);
  }
}"""


# Dựng bốn tab và nối sự kiện; thứ tự giá trị callback phải khớp danh sách outputs.
# Định nghĩa hàm build_ui(service).
def build_ui(service):
    # Gán cho enrollment: service.webcam_enrollment (đọc thuộc tính).
    enrollment = service.webcam_enrollment

    # Tạo thẻ ảnh cho 4 góc; ảnh thu nhỏ được nhúng Base64 vào HTML.
    # Định nghĩa hàm preview_html(token, next_pose). Mặc định: token=None, next_pose=None.
    def preview_html(token=None, next_pose=None):
        # Nạp base64 để đưa byte ảnh về chuỗi để nhúng trong HTML hoặc JSON.
        import base64
        # Lấy BytesIO từ io để đọc/ghi byte ảnh trong bộ nhớ bằng BytesIO.
        from io import BytesIO
        # Lấy Image từ PIL để đọc/ghi ảnh, xử lý hướng EXIF, chọn font và vẽ lên ảnh bằng Pillow.
        from PIL import Image
        # Gán cho captures: chọn dict(enrollment.review(token)) nếu token; ngược lại chọn {}.
        captures = dict(enrollment.review(token)) if token else {}
        # Gán cho cards: danh sách [] ([] giữ các phần tử theo thứ tự).
        cards = []
        # Lặp qua enumerate(POSES, 1); mỗi lượt gán phần tử cho (number, pose).
        for number, pose in enumerate(POSES, 1):
            # Gán cho label: lấy phần tử/lát cắt POSE_SHORT_LABELS[pose] từ POSE_SHORT_LABELS.
            label = POSE_SHORT_LABELS[pose]
            # Chỉ chạy khối bên dưới khi pose in captures cho kết quả đúng.
            if pose in captures:
                # Gán cho data: tạo luồng byte trong bộ nhớ thay vì file trên ổ đĩa bằng BytesIO(...).
                data = BytesIO()
                # Gán cho thumbnail: tạo ảnh Pillow từ mảng NumPy bằng Image.fromarray(...); dữ liệu/tham số: captures[pose].
                thumbnail = Image.fromarray(captures[pose])
                # Thu ảnh xem trước để vừa khung 240×240 pixel nhưng giữ tỷ lệ; Pillow sửa trực tiếp đối tượng thumbnail.
                thumbnail.thumbnail((240, 240))
                # Mã hóa và lưu ảnh qua thumbnail.save(...); dữ liệu/tham số: data, format='JPEG'.
                thumbnail.save(data, format="JPEG")
                # Lấy byte JPEG từ bộ đệm, mã hóa Base64 rồi đổi byte Base64 sang chữ ASCII để chèn vào thuộc tính src của ảnh.
                encoded = base64.b64encode(data.getvalue()).decode("ascii")
                # Gán cho visual: chuỗi f-string: chèn giá trị biểu thức nằm trong {...} vào nội dung.
                visual = f'<img class="studio-pose-image" alt="{label}" src="data:image/jpeg;base64,{encoded}">'
                # Tách kết quả theo thứ tự và gán cho (state, note): bộ giá trị ('is-done', 'Đã chụp') (tuple giữ thứ tự, không gán lại từng phần tử).
                state, note = "is-done", "Đã chụp"
            # Nhánh else: chạy khi nhánh if trước đó không được chọn (hoặc vòng lặp kết thúc bình thường nếu thuộc vòng lặp).
            else:
                # Gán cho visual: chuỗi f-string: chèn giá trị biểu thức nằm trong {...} vào nội dung.
                visual = f'<div class="studio-pose-image" aria-hidden="true"><span>{number:02}</span></div>'
                # Tách kết quả theo thứ tự và gán cho (state, note): chọn ('is-next', 'Góc hiện tại') nếu pose == next_pose; ngược lại chọn ('', 'Chờ chụp').
                state, note = ("is-next", "Góc hiện tại") if pose == next_pose else ("", "Chờ chụp")
            # Thêm một phần tử vào cuối danh sách qua cards.append(...); dữ liệu/tham số: f'<figure class="studio-pose {state}">{visual}<figcaption><strong>{label}</strong><small>{note}</sm….
            cards.append(f'<figure class="studio-pose {state}">{visual}'
                         # Tiếp phần biểu thức nhiều dòng: chuỗi '</small></figcaption></figure>'.
                         f'<figcaption><strong>{label}</strong><small>{note}</small></figcaption></figure>')
        # Kết thúc hàm và trả cộng hoặc nối trong '<div class="studio-poses">' + ''.join(cards) + '</div>'.
        return '<div class="studio-poses">' + ''.join(cards) + '</div>'

    # Tạo thanh trạng thái dễ đọc và giữ kết quả chi tiết cho ô JSON.
    # Định nghĩa hàm show_health().
    def show_health():
        # Gán cho result: đọc tình trạng hệ thống qua service.health(...).
        result = service.health()
        # Gán cho ready: kết quả đúng/sai của điều kiện result['status'] == 'ok'.
        ready = result["status"] == "ok"
        # Gán cho label: chọn 'Hệ thống sẵn sàng' nếu ready; ngược lại chọn 'Kiểm tra trạng thái hệ thống'.
        label = "Hệ thống sẵn sàng" if ready else "Kiểm tra trạng thái hệ thống"
        # Gán cho count: lấy phần tử/lát cắt result['person_count'] từ result.
        count = result["person_count"]
        # Gán cho people: chọn f'{count} hồ sơ đã đăng ký' nếu count is not None; ngược lại chọn 'Đang chờ dữ liệu hồ sơ'.
        people = f"{count} hồ sơ đã đăng ký" if count is not None else "Đang chờ dữ liệu hồ sơ"
        # Gán cho state: chọn '' nếu ready; ngược lại chọn 'is-alert'.
        state = "" if ready else "is-alert"
        # Kết thúc hàm và trả bộ giá trị (f'<div class="studio-health-bar" role="status"><span class="studio-health-pill {sta… (tuple giữ thứ tự, không gán lại từng phần tử).
        return (f'<div class="studio-health-bar" role="status"><span class="studio-health-pill {state}">'
                # Tiếp phần biểu thức nhiều dòng: chuỗi '</span><span>'.
                f'{label}</span><span>{people}</span></div>', result)

    # Tạo tiêu đề và mô tả dùng chung cho các vùng giao diện.
    # Định nghĩa hàm section_title(title, description). Sau dấu * phải truyền theo tên: kicker=None, level=2.
    def section_title(title, description, *, kicker=None, level=2):
        # Gán cho prefix: chọn f'<p class="studio-kicker">{kicker}</p>' nếu kicker; ngược lại chọn ''.
        prefix = f'<p class="studio-kicker">{kicker}</p>' if kicker else ""
        # Hiển thị đoạn HTML bằng gr.HTML(...); dữ liệu/tham số: f'<div class="studio-section-title">{prefix}<h{level}>{title}</h{level}><p>{description}</p></div>'.
        gr.HTML(f'<div class="studio-section-title">{prefix}<h{level}>{title}</h{level}>'
                # Tiếp phần biểu thức nhiều dòng: biểu thức {description}.
                f'<p>{description}</p></div>')

    # Đọc hồ sơ để tạo bảng và lựa chọn ID; thiếu tên cũ thì hiển thị dấu gạch.
    # Định nghĩa hàm load_registrations().
    def load_registrations():
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Gán cho people: đọc danh sách hồ sơ qua service.persons(...).
            people = service.persons()
        # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
        except ApplicationError as error:
            # Kết thúc hàm và trả bộ giá trị ([], gr.update(choices=[], value=None, interactive=False), error.message) (tuple giữ thứ tự, không gán lại từng phần tử).
            return [], gr.update(choices=[], value=None, interactive=False), error.message
        # Gán cho rows: tạo danh sách bằng cách duyệt person trong people; mỗi phần tử là [person['person_id'], person.get('display_na….
        rows = [[person["person_id"], person.get("display_name", "—"), person["num_images"]] for person in people]
        # Gán cho choices: tạo danh sách bằng cách duyệt person trong people; mỗi phần tử là (f'{person['display_name']} · {person['perso….
        choices = [
            # Tiếp phần biểu thức nhiều dòng: chọn f'{person['display_name']} · {person['person… nếu person.get('display_name'); ngược lại chọn person['person_id'].
            (f'{person["display_name"]} · {person["person_id"]}' if person.get("display_name") else person["person_id"],
             # Tiếp phần biểu thức nhiều dòng: lấy phần tử/lát cắt person['person_id'] từ person.
             person["person_id"])
            # Phần for của biểu thức rút gọn: duyệt từng phần tử theo thứ tự và tạo kết quả.
            for person in people
        # Đóng phần biểu thức/tham số nhiều dòng đang mở; các dấu ngoặc không tạo bước xử lý riêng.
        ]
        # Gán cho summary: chọn f'{len(people)} hồ sơ đã đăng ký.' nếu people; ngược lại chọn 'Chưa có hồ sơ nào. Bạn có thể tạo hồ sơ tro….
        summary = f"{len(people)} hồ sơ đã đăng ký." if people else "Chưa có hồ sơ nào. Bạn có thể tạo hồ sơ trong tab Đăng ký webcam."
        # Kết thúc hàm và trả bộ giá trị (rows, gr.update(choices=choices, value=None, interactive=bool(choices)), summary) (tuple giữ thứ tự, không gán lại từng phần tử).
        return rows, gr.update(choices=choices, value=None, interactive=bool(choices)), summary

    # Làm mới danh sách và xóa trạng thái chờ xác nhận xóa trước đó.
    # Định nghĩa hàm refresh_registrations().
    def refresh_registrations():
        # Kết thúc hàm và trả bộ giá trị (*load_registrations(), None, gr.update(visible=False), '') (tuple giữ thứ tự, không gán lại từng phần tử).
        return (*load_registrations(), None, gr.update(visible=False), "")

    # Lưu ID đang chờ và mở hộp xác nhận; chưa xóa dữ liệu ở bước này.
    # Định nghĩa hàm request_registration_delete(person_id).
    def request_registration_delete(person_id):
        # Chỉ chạy khối bên dưới khi not person_id cho kết quả đúng.
        if not person_id:
            # Kết thúc hàm và trả bộ giá trị (None, gr.update(visible=False), '', 'Hãy chọn hồ sơ cần xóa.') (tuple giữ thứ tự, không gán lại từng phần tử).
            return None, gr.update(visible=False), "", "Hãy chọn hồ sơ cần xóa."
        # Kết thúc hàm và trả bộ giá trị (person_id, gr.update(visible=True), f'Bạn có muốn xóa hồ sơ ID {person_id}?', '') (tuple giữ thứ tự, không gán lại từng phần tử).
        return person_id, gr.update(visible=True), f"Bạn có muốn xóa hồ sơ ID {person_id}?", ""

    # Xóa ID đã chọn rồi nạp lại bảng, lựa chọn và thông báo.
    # Định nghĩa hàm confirm_registration_delete(person_id).
    def confirm_registration_delete(person_id):
        # Chỉ chạy khối bên dưới khi not person_id cho kết quả đúng.
        if not person_id:
            # Gán cho message: chuỗi 'Hãy chọn hồ sơ và mở xác nhận xóa.'.
            message = "Hãy chọn hồ sơ và mở xác nhận xóa."
        # Nhánh else: chạy khi nhánh if trước đó không được chọn (hoặc vòng lặp kết thúc bình thường nếu thuộc vòng lặp).
        else:
            # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
            try:
                # Gửi lệnh xóa theo ID hoặc URL qua service.delete(...); dữ liệu/tham số: person_id.
                service.delete(person_id)
                # Gán cho message: chuỗi f-string: chèn giá trị biểu thức nằm trong {...} vào nội dung.
                message = f"Đã xóa hồ sơ ID {person_id}."
            # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
            except ApplicationError as error:
                # Gán cho message: error.message (đọc thuộc tính).
                message = error.message
        # Kết thúc hàm và trả bộ giá trị (*load_registrations(), None, gr.update(visible=False), message) (tuple giữ thứ tự, không gán lại từng phần tử).
        return (*load_registrations(), None, gr.update(visible=False), message)

    # Chuyển trạng thái phiên thành 7 giá trị theo đúng thứ tự enrollment_outputs.
    # Định nghĩa hàm enrollment_result(result, refresh_preview). Mặc định: refresh_preview=True.
    def enrollment_result(result, refresh_preview=True):
        # Gán cho state: tạo từ điển bằng vòng lặp rút gọn; khóa là key, giá trị là result[key].
        state = {key: result[key] for key in ("session_id", "revision")}
        # Gán cho message: chuỗi f-string: chèn giá trị biểu thức nằm trong {...} vào nội dung.
        message = f'{result["progress"]}/{result["total"]} · {result["instruction"]}\n{result["message"]}'
        # Gán cho public: tạo từ điển bằng vòng lặp rút gọn; khóa là k, giá trị là v.
        public = {k: v for k, v in result.items() if k != "session_id"}
        # Gán cho preview: chọn preview_html(state['session_id'], result['ne… nếu refresh_preview; ngược lại chọn gr.skip().
        preview = preview_html(state["session_id"], result["next_pose"]) if refresh_preview else gr.skip()
        # Thứ tự trả về: phiên, hướng dẫn, JSON, ảnh xem lại, ô đồng ý, nút lưu, nút bắt đầu.
        # Kết thúc hàm và trả bộ giá trị (state, message, public, preview, False, gr.update(interactive=result['status'] == '… (tuple giữ thứ tự, không gán lại từng phần tử).
        return (state, message, public, preview, False,
                # Truyền interactive (cho phép người dùng thao tác hay chỉ xem) = result['status'] == 'review'.
                gr.update(interactive=result["status"] == "review"), gr.update(interactive=False))

    # Hiển thị lỗi; nếu revision đã cũ thì đồng bộ lại trạng thái mới nhất từ dịch vụ.
    # Định nghĩa hàm enrollment_error(state, error).
    def enrollment_error(state, error):
        # Chỉ chạy khối bên dưới khi error.code == 'session_missing' cho kết quả đúng.
        if error.code == "session_missing":
            # Kết thúc hàm và trả bộ giá trị ({}, error.message, error.payload(), preview_html(), False, gr.update(interactive=Fa… (tuple giữ thứ tự, không gán lại từng phần tử).
            return ({}, error.message, error.payload(), preview_html(), False,
                    # Truyền interactive (cho phép người dùng thao tác hay chỉ xem) = False.
                    gr.update(interactive=False), gr.update(interactive=True))
        # Chỉ chạy khối bên dưới khi error.code in {'stale_frame', 'stale_review'} and state cho kết quả đúng.
        if error.code in {"stale_frame", "stale_review"} and state:
            # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
            try:
                # Kết thúc hàm và trả đổi trạng thái phiên thành các giá trị cập nhật giao diện bằng enrollment_result(...); dữ liệu/tham số: enrollment.status(state['session_id']) (lấy tiến độ phiên hiện tại qua enrollment.status(...)).
                return enrollment_result(enrollment.status(state["session_id"]))
            # Bắt lỗi ApplicationError và đặt tên current_error để xử lý trong khối bên dưới.
            except ApplicationError as current_error:
                # Kết thúc hàm và trả đổi lỗi phiên thành thông báo và trạng thái giao diện bằng enrollment_error(...); dữ liệu/tham số: state, current_error.
                return enrollment_error(state, current_error)
        # Kết thúc hàm và trả bộ giá trị (state, error.message, error.payload(), gr.skip(), False, gr.skip(), gr.skip()) (tuple giữ thứ tự, không gán lại từng phần tử).
        return (state, error.message, error.payload(), gr.skip(), False, gr.skip(), gr.skip())

    # Bắt đầu phiên chụp cho ID và tên được cung cấp.
    # Định nghĩa hàm start_enrollment(person_id, name, state).
    def start_enrollment(person_id, name, state):
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Chỉ chạy khối bên dưới khi state cho kết quả đúng.
            if state:
                # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'session_active', 'Hủy phiên hiện tại trước khi đăng ký người khác.'.
                raise ApplicationError("session_active", "Hủy phiên hiện tại trước khi đăng ký người khác.")
            # Kết thúc hàm và trả đổi trạng thái phiên thành các giá trị cập nhật giao diện bằng enrollment_result(...); dữ liệu/tham số: enrollment.start(person_id, name) (khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua enrollment.start(...)).
            return enrollment_result(enrollment.start(person_id, name))
        # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
        except ApplicationError as error:
            # Kết thúc hàm và trả đổi lỗi phiên thành thông báo và trạng thái giao diện bằng enrollment_error(...); dữ liệu/tham số: state, error.
            return enrollment_error(state, error)

    # Xử lý frame đang truyền; gr.skip() giữ nguyên thành phần không cần cập nhật.
    # Định nghĩa hàm enrollment_stream(frame, state).
    def enrollment_stream(frame, state):
        # Chỉ chạy khối bên dưới khi not state cho kết quả đúng.
        if not state:
            # Trả tuple gồm 7 giá trị gr.skip() để giữ nguyên cả 7 thành phần đầu ra; dấu phẩy tạo tuple một phần tử, * 7 lặp tuple đó.
            return (gr.skip(),) * 7
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Gán cho status: lấy tiến độ phiên hiện tại qua enrollment.status(...); dữ liệu/tham số: state['session_id'].
            status = enrollment.status(state["session_id"])
            # Chỉ chạy khối bên dưới khi status['status'] in {'review', 'paused'} cho kết quả đúng.
            if status["status"] in {"review", "paused"}:
                # Trả tuple gồm 7 giá trị gr.skip() để giữ nguyên cả 7 thành phần đầu ra; dấu phẩy tạo tuple một phần tử, * 7 lặp tuple đó.
                return (gr.skip(),) * 7
            # Gán cho result: kiểm tra và xử lý một frame đăng ký qua enrollment.frame(...); dữ liệu/tham số: state['session_id'], frame, state['revision'].
            result = enrollment.frame(state["session_id"], frame, state["revision"])
            # Kết thúc hàm và trả đổi trạng thái phiên thành các giá trị cập nhật giao diện bằng enrollment_result(...); dữ liệu/tham số: result, refresh_preview=result.get('captured', False).
            return enrollment_result(result, refresh_preview=result.get("captured", False))
        # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
        except ApplicationError as error:
            # Chỉ chạy khối bên dưới khi error.code == 'session_missing' cho kết quả đúng.
            if error.code == "session_missing":
                # Phản hồi stream đang truyền không được ghi đè thông báo thành công hoặc
                # hủy của nút điều khiển vừa kết thúc phiên.
                # Trả tuple gồm 7 giá trị gr.skip() để giữ nguyên cả 7 thành phần đầu ra; dấu phẩy tạo tuple một phần tử, * 7 lặp tuple đó.
                return (gr.skip(),) * 7
            # Kết thúc hàm và trả đổi lỗi phiên thành thông báo và trạng thái giao diện bằng enrollment_error(...); dữ liệu/tham số: state, error.
            return enrollment_error(state, error)

    # Yêu cầu chụp lại một góc trong phiên đăng ký hiện tại.
    # Định nghĩa hàm retake_enrollment(state, pose).
    def retake_enrollment(state, pose):
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Kết thúc hàm và trả đổi trạng thái phiên thành các giá trị cập nhật giao diện bằng enrollment_result(...); dữ liệu/tham số: enrollment.retake(state.get('session_id'), pose) (yêu cầu chụp lại góc đã chọn qua enrollment.retake(...)).
            return enrollment_result(enrollment.retake(state.get("session_id"), pose))
        # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
        except ApplicationError as error:
            # Kết thúc hàm và trả đổi lỗi phiên thành thông báo và trạng thái giao diện bằng enrollment_error(...); dữ liệu/tham số: state, error.
            return enrollment_error(state, error)

    # Hủy phiên và trả trạng thái để giao diện dọn dữ liệu đang hiển thị.
    # Định nghĩa hàm cancel_enrollment(state).
    def cancel_enrollment(state):
        # Gán cho result: hủy phiên và ảnh tạm qua enrollment.cancel(...); dữ liệu/tham số: state.get('session_id') (đọc giá trị theo khóa, dùng mặc định nếu khóa vắng mặt qua state.get(...)).
        result = enrollment.cancel(state.get("session_id"))
        # Gán cho message: chọn f'0/{result['total']} · Đã hủy phiên. Chưa l… nếu result['status'] == 'cancelled'; ngược lại chọn 'Phiên đã kết thúc hoặc hết hạn.'.
        message = (f'0/{result["total"]} · Đã hủy phiên. Chưa lưu danh tính.' if result["status"] == "cancelled"
                   # Nhánh còn lại của biểu thức điều kiện: chọn giá trị sau else khi điều kiện trước sai.
                   else "Phiên đã kết thúc hoặc hết hạn.")
        # Kết thúc hàm và trả bộ giá trị ({}, message, result, preview_html(), False, gr.update(interactive=False), gr.update… (tuple giữ thứ tự, không gán lại từng phần tử).
        return ({}, message, result, preview_html(), False,
                # Truyền interactive (cho phép người dùng thao tác hay chỉ xem) = False.
                gr.update(interactive=False), gr.update(interactive=True))

    # Tạm dừng phiên khi camera dừng hoặc người dùng ngắt camera.
    # Định nghĩa hàm pause_enrollment(state).
    def pause_enrollment(state):
        # Chỉ chạy khối bên dưới khi not state cho kết quả đúng.
        if not state:
            # Trả tuple gồm 7 giá trị gr.skip() để giữ nguyên cả 7 thành phần đầu ra; dấu phẩy tạo tuple một phần tử, * 7 lặp tuple đó.
            return (gr.skip(),) * 7
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Kết thúc hàm và trả đổi trạng thái phiên thành các giá trị cập nhật giao diện bằng enrollment_result(...); dữ liệu/tham số: enrollment.pause(state['session_id']) (tạm dừng phiên chụp qua enrollment.pause(...)).
            return enrollment_result(enrollment.pause(state["session_id"]))
        # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
        except ApplicationError as error:
            # Kết thúc hàm và trả đổi lỗi phiên thành thông báo và trạng thái giao diện bằng enrollment_error(...); dữ liệu/tham số: state, error.
            return enrollment_error(state, error)

    # Tiếp tục chụp sau khi bật lại camera.
    # Định nghĩa hàm resume_enrollment(state).
    def resume_enrollment(state):
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Kết thúc hàm và trả đổi trạng thái phiên thành các giá trị cập nhật giao diện bằng enrollment_result(...); dữ liệu/tham số: enrollment.resume(state.get('session_id')) (tiếp tục phiên chụp qua enrollment.resume(...)).
            return enrollment_result(enrollment.resume(state.get("session_id")))
        # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
        except ApplicationError as error:
            # Kết thúc hàm và trả đổi lỗi phiên thành thông báo và trạng thái giao diện bằng enrollment_error(...); dữ liệu/tham số: state, error.
            return enrollment_error(state, error)

    # Xác nhận bộ ảnh đang xem, lưu hồ sơ rồi đưa giao diện về trạng thái tạo phiên mới.
    # Định nghĩa hàm save_enrollment(state, accepted).
    def save_enrollment(state, accepted):
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Gán cho result: xác nhận lưu bộ ảnh đã xem qua enrollment.confirm(...); dữ liệu/tham số: state.get('session_id') (đọc giá trị theo khóa, dùng mặc định nếu khóa vắng mặt qua state.get(...)), accepted, state.get('revision') (đọc giá trị theo khóa, dùng mặc định nếu khóa vắng mặt qua state.get(...)).
            result = enrollment.confirm(state.get("session_id"), accepted, state.get("revision"))
            # Kết thúc hàm và trả bộ giá trị ({}, f'{POSE_COUNT}/{POSE_COUNT} · Đã lưu. Chuyển sang Nhận diện ảnh hoặc Webcam để … (tuple giữ thứ tự, không gán lại từng phần tử).
            return ({}, f"{POSE_COUNT}/{POSE_COUNT} · Đã lưu. Chuyển sang Nhận diện ảnh hoặc Webcam để kiểm tra danh tính.", result,
                    # Tiếp phần biểu thức nhiều dòng: chuỗi '<p class="studio-footer">Đã lưu hồ sơ. Bắt đầu phiên mới để đăn….
                    '<p class="studio-footer">Đã lưu hồ sơ. Bắt đầu phiên mới để đăng ký thêm.</p>',
                    # Truyền interactive (cho phép người dùng thao tác hay chỉ xem) = False.
                    False, gr.update(interactive=False), gr.update(interactive=True))
        # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
        except ApplicationError as error:
            # Kết thúc hàm và trả đổi lỗi phiên thành thông báo và trạng thái giao diện bằng enrollment_error(...); dữ liệu/tham số: state, error.
            return enrollment_error(state, error)

    # Kiểm tra định kỳ và nhắc khi không còn nhận được frame camera.
    # Định nghĩa hàm enrollment_watchdog(state).
    def enrollment_watchdog(state):
        # Chỉ chạy khối bên dưới khi not state cho kết quả đúng.
        if not state:
            # Kết thúc hàm và trả giữ nguyên giá trị giao diện hiện tại qua gr.skip(...).
            return gr.skip()
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Gán cho status: lấy tiến độ phiên hiện tại qua enrollment.status(...); dữ liệu/tham số: state['session_id'].
            status = enrollment.status(state["session_id"])
            # Chỉ chạy khối bên dưới khi status['status'] == 'capturing' and 'Không nhận được' in status['message'] cho kết quả đúng.
            if status["status"] == "capturing" and "Không nhận được" in status["message"]:
                # Kết thúc hàm và trả chuỗi f-string: chèn giá trị biểu thức nằm trong {...} vào nội dung.
                return f'{status["progress"]}/{status["total"]} · {status["instruction"]}\n{status["message"]}'
        # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
        except ApplicationError as error:
            # Kết thúc hàm và trả error.message (đọc thuộc tính).
            return error.message
        # Kết thúc hàm và trả giữ nguyên giá trị giao diện hiện tại qua gr.skip(...).
        return gr.skip()

    # Đọc file tạm do Gradio cung cấp với giới hạn dung lượng trước khi giải mã.
    # Định nghĩa hàm read_upload(path).
    def read_upload(path):
        # Chỉ chạy khối bên dưới khi path is None cho kết quả đúng.
        if path is None:
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'missing_image', 'Hãy chọn ảnh.'.
            raise ApplicationError("missing_image", "Hãy chọn ảnh.")
        # Mở khối with để mở file hoặc ảnh theo chế độ chỉ định qua Path(path).open(...); dữ liệu/tham số: 'rb'; gọi tài nguyên là stream; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with Path(path).open("rb") as stream:
            # Kết thúc hàm và trả giải mã byte thành ảnh RGB đã kiểm tra bằng decode_image(...); dữ liệu/tham số: stream.read(service.settings.max_file_bytes + 1) (đọc dữ liệu từ luồng qua stream.read(...)), service.settings.
            return decode_image(stream.read(service.settings.max_file_bytes + 1), service.settings)

    # Chạy nhận diện và trả kết quả cho từng khuôn mặt cùng thời gian xử lý.
    # Định nghĩa hàm recognize(path).
    def recognize(path):
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Gán cho image: đọc file ảnh tạm do giao diện cung cấp bằng read_upload(...); dữ liệu/tham số: path.
            image = read_upload(path)
            # Gán cho result: nhận diện ảnh bằng dịch vụ qua service.recognize(...); dữ liệu/tham số: image.
            result = service.recognize(image)
            # Gán cho recognized: cộng các giá trị bằng sum(...); dữ liệu/tham số: (face['status'] == 'recognized' for face in result['faces']).
            recognized = sum(face["status"] == "recognized" for face in result["faces"])
            # Gán cho message: chọn f'Nhận diện được {recognized}/{result['face_… nếu result['faces']; ngược lại chọn 'Không tìm thấy khuôn mặt.'.
            message = (f'Nhận diện được {recognized}/{result["face_count"]} khuôn mặt.'
                       # Phần if của biểu thức rút gọn: chỉ giữ phần tử thỏa điều kiện này.
                       if result["faces"] else "Không tìm thấy khuôn mặt.")
            # Kết thúc hàm và trả bộ giá trị (annotate(image, result['faces']), result, message) (tuple giữ thứ tự, không gán lại từng phần tử).
            return annotate(image, result["faces"]), result, message
        # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
        except ApplicationError as error:
            # Kết thúc hàm và trả bộ giá trị (None, error.payload(), error.message) (tuple giữ thứ tự, không gán lại từng phần tử).
            return None, error.payload(), error.message

    # Chỉ nhận diện khi cờ enabled bật; trả ảnh, chi tiết, bộ đếm và thông báo.
    # Định nghĩa hàm stream(frame, enabled, state).
    def stream(frame, enabled, state):
        # Chỉ chạy khối bên dưới khi not enabled cho kết quả đúng.
        if not enabled:
            # Kết thúc hàm và trả bộ giá trị (gr.skip(), gr.skip(), state, 'Đã dừng nhận diện. Dừng camera bằng nút trong khung c… (tuple giữ thứ tự, không gán lại từng phần tử).
            return gr.skip(), gr.skip(), state, "Đã dừng nhận diện. Dừng camera bằng nút trong khung camera."
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Tách kết quả theo thứ tự và gán cho (output, result, state): xử lý một frame và cập nhật thống kê phiên webcam bằng process_frame(...); dữ liệu/tham số: service, frame, state.
            output, result, state = process_frame(service, frame, state)
            # Gán cho text: chuỗi f-string: chèn giá trị biểu thức nằm trong {...} vào nội dung.
            text = (f'{result["face_count"]} khuôn mặt · {result["frames_processed"]} khung hình đã xử lý · '
                    # Tiếp phần biểu thức nhiều dòng: biểu thức {result['frame_latency_ms']:.1f}.
                    f'{result["stream_fps"]:.2f} FPS · {result["frame_latency_ms"]:.1f} ms mỗi khung')
            # Kết thúc hàm và trả bộ giá trị (output, result, state, text) (tuple giữ thứ tự, không gán lại từng phần tử).
            return output, result, state, text
        # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
        except ApplicationError as error:
            # Kết thúc hàm và trả bộ giá trị (None, error.payload(), state, error.message) (tuple giữ thứ tự, không gán lại từng phần tử).
            return None, error.payload(), state, error.message

    # Mở khối with để tạo khung giao diện Gradio chứa các thành phần và sự kiện bằng gr.Blocks(...); dữ liệu/tham số: title='Face Recognition Studio', analytics_enabled=False, delete_cache=(60, 300); gọi tài nguyên là demo; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with gr.Blocks(title="Face Recognition Studio", analytics_enabled=False,
                   # Truyền delete_cache = (60, 300).
                   delete_cache=(60, 300)) as demo:
        # Hiển thị đoạn HTML bằng gr.HTML(...); dữ liệu/tham số: HERO, elem_id='studio-header'.
        gr.HTML(HERO, elem_id="studio-header")
        # Gán cho health_summary: hiển thị đoạn HTML bằng gr.HTML(...); dữ liệu/tham số: '<div class="studio-health-bar">Đang kiểm tra hệ thống…</div>', elem_id='studio-health'.
        health_summary = gr.HTML('<div class="studio-health-bar">Đang kiểm tra hệ thống…</div>', elem_id="studio-health")
        # Gán cho camera_height: chuỗi 'clamp(260px, 42vw, 440px)'.
        camera_height = "clamp(260px, 42vw, 440px)"
        # Mở khối with để tạo nhóm tab bằng gr.Tabs(...); dữ liệu/tham số: elem_id='studio-tabs'; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with gr.Tabs(elem_id="studio-tabs"):
            # Mở khối with để tạo một tab giao diện bằng gr.Tab(...); dữ liệu/tham số: f'1 · Đăng ký webcam · {POSE_COUNT} góc'; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
            with gr.Tab(f"1 · Đăng ký webcam · {POSE_COUNT} góc"):
                # Thêm tiêu đề và hướng dẫn vào giao diện bằng section_title(...); dữ liệu/tham số: 'Tạo hồ sơ khuôn mặt', 'Giữ yên theo hướng dẫn trên camera để hệ thống tự chụp từng góc.'.
                section_title("Tạo hồ sơ khuôn mặt",
                              # Tiếp phần biểu thức nhiều dòng: chuỗi 'Giữ yên theo hướng dẫn trên camera để hệ thống tự chụp từng góc….
                              "Giữ yên theo hướng dẫn trên camera để hệ thống tự chụp từng góc.")
                # gr.State giữ token/revision riêng theo phiên giao diện; hết hạn sẽ gọi hàm hủy phiên.
                # Gán cho enrollment_state: tạo vùng trạng thái riêng của phiên giao diện bằng gr.State(...); dữ liệu/tham số: {}, time_to_live=1200, delete_callback=lambda state: enrollment.cancel(state.get('session_id')).
                enrollment_state = gr.State({}, time_to_live=1200,
                                            # Truyền delete_callback = lambda state: enrollment.cancel(state.get('session_id')).
                                            delete_callback=lambda state: enrollment.cancel(state.get("session_id")))
                # Mở khối with để xếp thành phần theo hàng ngang bằng gr.Row(...); dữ liệu/tham số: elem_id='enrollment-workspace'; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                with gr.Row(elem_id="enrollment-workspace"):
                    # Mở khối with để xếp thành phần theo cột bằng gr.Column(...); dữ liệu/tham số: elem_id='enrollment-camera-panel'; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                    with gr.Column(elem_id="enrollment-camera-panel"):
                        # Gán cho enrollment_camera: tạo vùng ảnh hoặc camera bằng gr.Image(...); dữ liệu/tham số: label='Camera đăng ký (không lật gương)', sources=['webcam'], type='numpy', streaming=True.
                        enrollment_camera = gr.Image(
                            # Truyền label (nhãn hiển thị cho người dùng) = 'Camera đăng ký (không lật gương)'.
                            label="Camera đăng ký (không lật gương)", sources=["webcam"],
                            # Truyền type (dạng dữ liệu đầu vào/đầu ra của thành phần) = 'numpy'.
                            type="numpy", streaming=True, height=camera_height,
                            # Truyền webcam_options = gr.WebcamOptions(mirror=False, constraints={'video': {'width': {'ideal': 640}, 'height': {'ideal': 480}}}).
                            webcam_options=gr.WebcamOptions(
                                # Truyền mirror (có lật gương hình camera hay không) = False.
                                mirror=False, constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}},
                            # Đóng phần biểu thức/tham số nhiều dòng đang mở; các dấu ngoặc không tạo bước xử lý riêng.
                            ),
                        # Đóng phần biểu thức/tham số nhiều dòng đang mở; các dấu ngoặc không tạo bước xử lý riêng.
                        )
                        # Gán cho enrollment_message: tạo ô nhập hoặc hiển thị văn bản bằng gr.Textbox(...); dữ liệu/tham số: value=f'0/{POSE_COUNT} · Bắt đầu phiên để chụp góc nhìn thẳng.', label='Tiến độ & hướng dẫn', interactive=False, show_label=False.
                        enrollment_message = gr.Textbox(
                            # Truyền value (giá trị ban đầu) = f'0/{POSE_COUNT} · Bắt đầu phiên để chụp góc nhìn thẳng.'.
                            value=f"0/{POSE_COUNT} · Bắt đầu phiên để chụp góc nhìn thẳng.", label="Tiến độ & hướng dẫn",
                            # Truyền interactive (cho phép người dùng thao tác hay chỉ xem) = False.
                            interactive=False, show_label=False, container=False,
                            # Truyền lines = 2.
                            lines=2, max_lines=5, elem_id="enrollment-camera-status",
                        # Đóng phần biểu thức/tham số nhiều dòng đang mở; các dấu ngoặc không tạo bước xử lý riêng.
                        )
                    # Mở khối with để xếp thành phần theo cột bằng gr.Column(...); dữ liệu/tham số: elem_id='enrollment-form', elem_classes='studio-card'; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                    with gr.Column(elem_id="enrollment-form", elem_classes="studio-card"):
                        # Thêm tiêu đề và hướng dẫn vào giao diện bằng section_title(...); dữ liệu/tham số: 'Thông tin đăng ký', 'Nhập thông tin, bắt đầu phiên rồi bật camera và bấm ghi hình.', kicker='Bước 01', level=3.
                        section_title("Thông tin đăng ký",
                                      # Tiếp phần biểu thức nhiều dòng: chuỗi 'Nhập thông tin, bắt đầu phiên rồi bật camera và bấm ghi hình.'.
                                      "Nhập thông tin, bắt đầu phiên rồi bật camera và bấm ghi hình.",
                                      # Truyền kicker = 'Bước 01'.
                                      kicker="Bước 01", level=3)
                        # Gán cho enrollment_id: tạo ô nhập hoặc hiển thị văn bản bằng gr.Textbox(...); dữ liệu/tham số: label='ID đăng ký', placeholder='Ví dụ: NV001', max_length=100.
                        enrollment_id = gr.Textbox(label="ID đăng ký", placeholder="Ví dụ: NV001", max_length=100)
                        # Gán cho enrollment_name: tạo ô nhập hoặc hiển thị văn bản bằng gr.Textbox(...); dữ liệu/tham số: label='Họ và tên', placeholder='Nhập tên hiển thị', max_length=150.
                        enrollment_name = gr.Textbox(label="Họ và tên", placeholder="Nhập tên hiển thị", max_length=150)
                        # Gán cho enrollment_start: tạo nút bấm bằng gr.Button(...); dữ liệu/tham số: 'Bắt đầu phiên đăng ký', variant='primary'.
                        enrollment_start = gr.Button("Bắt đầu phiên đăng ký", variant="primary")
                        # Gán cho enrollment_cancel: tạo nút bấm bằng gr.Button(...); dữ liệu/tham số: 'Hủy phiên đăng ký'.
                        enrollment_cancel = gr.Button("Hủy phiên đăng ký")
                        # Mở khối with để tạo vùng nội dung có thể mở/thu gọn bằng gr.Accordion(...); dữ liệu/tham số: 'Tùy chọn camera', open=False; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                        with gr.Accordion("Tùy chọn camera", open=False):
                            # Gán cho enrollment_check: tạo nút bấm bằng gr.Button(...); dữ liệu/tham số: 'Kiểm tra quyền camera đăng ký'.
                            enrollment_check = gr.Button("Kiểm tra quyền camera đăng ký")
                            # Gán cho enrollment_camera_message: tạo ô nhập hoặc hiển thị văn bản bằng gr.Textbox(...); dữ liệu/tham số: label='Quyền camera', interactive=False.
                            enrollment_camera_message = gr.Textbox(label="Quyền camera", interactive=False)
                            # Gán cho enrollment_resume: tạo nút bấm bằng gr.Button(...); dữ liệu/tham số: 'Tiếp tục sau khi bật lại camera'.
                            enrollment_resume = gr.Button("Tiếp tục sau khi bật lại camera")
                        # Hiển thị đoạn HTML bằng gr.HTML(...); dữ liệu/tham số: '<p class="studio-footer">Quay trái và phải theo phía của bạn. Phiên có hiệu lực 20 phút.</p>'.
                        gr.HTML('<p class="studio-footer">Quay trái và phải theo phía của bạn. Phiên có hiệu lực 20 phút.</p>')
                    # Mở khối with để xếp thành phần theo cột bằng gr.Column(...); dữ liệu/tham số: elem_id='enrollment-review', elem_classes='studio-card'; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                    with gr.Column(elem_id="enrollment-review", elem_classes="studio-card"):
                        # Thêm tiêu đề và hướng dẫn vào giao diện bằng section_title(...); dữ liệu/tham số: f'Ảnh của {POSE_COUNT} góc', 'Xem lại ảnh, chụp lại nếu cần rồi xác nhận lưu.', kicker='Bước 02', level=3.
                        section_title(f"Ảnh của {POSE_COUNT} góc", "Xem lại ảnh, chụp lại nếu cần rồi xác nhận lưu.",
                                      # Truyền kicker = 'Bước 02'.
                                      kicker="Bước 02", level=3)
                        # Gán cho enrollment_preview: hiển thị đoạn HTML bằng gr.HTML(...); dữ liệu/tham số: preview_html() (tạo HTML cho các góc đã chụp và góc đang chờ bằng preview_html(...)), label='Xem lại ảnh đã chụp', elem_id='enrollment-preview'.
                        enrollment_preview = gr.HTML(preview_html(), label="Xem lại ảnh đã chụp", elem_id="enrollment-preview")
                        # Mở khối with để xếp thành phần theo hàng ngang bằng gr.Row(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                        with gr.Row():
                            # Gán cho retake_pose: tạo danh sách chọn thả xuống bằng gr.Dropdown(...); dữ liệu/tham số: choices=[(LABELS[p], p) for p in POSES], value='front', label='Góc cần chụp lại'.
                            retake_pose = gr.Dropdown(choices=[(LABELS[p], p) for p in POSES],
                                                     # Truyền value (giá trị ban đầu) = 'front'.
                                                     value="front", label="Góc cần chụp lại")
                            # Gán cho enrollment_retake: tạo nút bấm bằng gr.Button(...); dữ liệu/tham số: 'Chụp lại góc đã chọn'.
                            enrollment_retake = gr.Button("Chụp lại góc đã chọn")
                        # Gán cho enrollment_accept: tạo ô tích xác nhận bằng gr.Checkbox(...); dữ liệu/tham số: label=f'Tôi đã xem lại {POSE_COUNT} ảnh và đồng ý lưu hồ sơ', value=False.
                        enrollment_accept = gr.Checkbox(label=f"Tôi đã xem lại {POSE_COUNT} ảnh và đồng ý lưu hồ sơ", value=False)
                        # Gán cho enrollment_save: tạo nút bấm bằng gr.Button(...); dữ liệu/tham số: 'Xác nhận lưu đăng ký', variant='primary', interactive=False.
                        enrollment_save = gr.Button("Xác nhận lưu đăng ký", variant="primary", interactive=False)
                # Mở khối with để tạo vùng nội dung có thể mở/thu gọn bằng gr.Accordion(...); dữ liệu/tham số: 'Chi tiết tư thế và chất lượng', open=False; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                with gr.Accordion("Chi tiết tư thế và chất lượng", open=False):
                    # Gán cho enrollment_details: hiển thị dữ liệu JSON bằng gr.JSON(...); dữ liệu/tham số: label='Tư thế, chất lượng & trạng thái phiên'.
                    enrollment_details = gr.JSON(label="Tư thế, chất lượng & trạng thái phiên")
                # Gắn callback vào sự kiện bấm nút qua enrollment_check.click(...); dữ liệu/tham số: fn=None, outputs=enrollment_camera_message, js=CAMERA_CHECK_JS.replace('Bắt đầu nhận diện', 'Bắt đầu phiên đăng ký'), queue=False.
                enrollment_check.click(fn=None, outputs=enrollment_camera_message,
                                       # Truyền js (JavaScript chạy ở trình duyệt) = CAMERA_CHECK_JS.replace('Bắt đầu nhận diện', 'Bắt đầu phiên đăng ký').
                                       js=CAMERA_CHECK_JS.replace("Bắt đầu nhận diện", "Bắt đầu phiên đăng ký"), queue=False)
                # Các callback đăng ký phải trả đúng 7 giá trị theo danh sách thành phần này.
                # Gán cho enrollment_outputs: danh sách [enrollment_state, enrollment_message, enrollment_details, enrollment_preview, enrol… ([] giữ các phần tử theo thứ tự).
                enrollment_outputs = [enrollment_state, enrollment_message, enrollment_details, enrollment_preview,
                                      # Tiếp phần biểu thức nhiều dòng: enrollment_accept.
                                      enrollment_accept, enrollment_save, enrollment_start]
                # Các nút điều khiển chạy tuần tự trong một nhóm để tránh thay đổi phiên cùng lúc.
                # Gán cho event_options: tạo từ điển khóa–giá trị bằng dict(...); dữ liệu/tham số: api_visibility='private', concurrency_limit=1, concurrency_id='webcam-enrollment-controls'.
                event_options = dict(api_visibility="private", concurrency_limit=1, concurrency_id="webcam-enrollment-controls")
                # Gắn callback vào sự kiện bấm nút qua enrollment_start.click(...); dữ liệu/tham số: start_enrollment, [enrollment_id, enrollment_name, enrollment_state], enrollment_outputs, **=event_options.
                enrollment_start.click(start_enrollment, [enrollment_id, enrollment_name, enrollment_state], enrollment_outputs, **event_options)
                # Stream gửi frame khoảng 0,25 giây/lần; dùng nhóm riêng để không chiếm hàng đợi của nút.
                # Gán cho enrollment_stream_event: gắn hàm xử lý chuỗi frame camera qua enrollment_camera.stream(...); dữ liệu/tham số: enrollment_stream, [enrollment_camera, enrollment_state], enrollment_outputs, stream_every=0.25, time_limit=1200, api_visibility='private', concurrency_limit=enrollment.policy.max_sessions.
                enrollment_stream_event = enrollment_camera.stream(
                    # Tiếp phần biểu thức nhiều dòng: danh sách [enrollment_camera, enrollment_state] ([] giữ các phần tử theo thứ tự).
                    enrollment_stream, [enrollment_camera, enrollment_state], enrollment_outputs,
                    # Truyền stream_every (khoảng thời gian gửi frame, tính bằng giây) = 0.25.
                    stream_every=0.25, time_limit=1200, api_visibility="private",
                    # Truyền concurrency_limit (số tác vụ được xử lý đồng thời trong nhóm) = enrollment.policy.max_sessions.
                    concurrency_limit=enrollment.policy.max_sessions, concurrency_id="webcam-enrollment-stream")
                # Gắn callback vào sự kiện bấm nút qua enrollment_retake.click(...); dữ liệu/tham số: retake_enrollment, [enrollment_state, retake_pose], enrollment_outputs, **=event_options.
                enrollment_retake.click(retake_enrollment, [enrollment_state, retake_pose], enrollment_outputs, **event_options)
                # Gắn callback vào sự kiện bấm nút qua enrollment_cancel.click(...); dữ liệu/tham số: cancel_enrollment, enrollment_state, enrollment_outputs, cancels=[enrollment_stream_event], **=event_options.
                enrollment_cancel.click(cancel_enrollment, enrollment_state, enrollment_outputs,
                                        # Truyền cancels (những sự kiện đang chạy cần hủy) = [enrollment_stream_event].
                                        cancels=[enrollment_stream_event], **event_options)
                # Xóa các phần tử hoặc đăng ký sự kiện xóa tùy đối tượng qua enrollment_camera.clear(...); dữ liệu/tham số: pause_enrollment, enrollment_state, enrollment_outputs, cancels=[enrollment_stream_event], **=event_options.
                enrollment_camera.clear(pause_enrollment, enrollment_state, enrollment_outputs,
                                        # Truyền cancels (những sự kiện đang chạy cần hủy) = [enrollment_stream_event].
                                        cancels=[enrollment_stream_event], **event_options)
                # Gắn callback vào sự kiện bấm nút qua enrollment_resume.click(...); dữ liệu/tham số: resume_enrollment, enrollment_state, enrollment_outputs, **=event_options.
                enrollment_resume.click(resume_enrollment, enrollment_state, enrollment_outputs, **event_options)
                # Gán cho saved: gắn callback vào sự kiện bấm nút qua enrollment_save.click(...); dữ liệu/tham số: save_enrollment, [enrollment_state, enrollment_accept], enrollment_outputs, **=event_options.
                saved = enrollment_save.click(save_enrollment, [enrollment_state, enrollment_accept], enrollment_outputs, **event_options)
                # Gán cho enrollment_timer: tạo đồng hồ kích hoạt định kỳ bằng gr.Timer(...); dữ liệu/tham số: 1.
                enrollment_timer = gr.Timer(1)
                # Gắn callback chạy định kỳ theo bộ hẹn giờ qua enrollment_timer.tick(...); dữ liệu/tham số: enrollment_watchdog, enrollment_state, enrollment_message, queue=False, api_visibility='private'.
                enrollment_timer.tick(enrollment_watchdog, enrollment_state, enrollment_message,
                                      # Truyền queue (có đưa sự kiện vào hàng đợi hay không) = False.
                                      queue=False, api_visibility="private")
            # Mở khối with để tạo một tab giao diện bằng gr.Tab(...); dữ liệu/tham số: '2 · Nhận diện ảnh'; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
            with gr.Tab("2 · Nhận diện ảnh"):
                # Thêm tiêu đề và hướng dẫn vào giao diện bằng section_title(...); dữ liệu/tham số: 'Đối chiếu từ ảnh', 'Chọn một ảnh JPG, PNG hoặc WebP. Kết quả sẽ xuất hiện ở khung bên cạnh.'.
                section_title("Đối chiếu từ ảnh", "Chọn một ảnh JPG, PNG hoặc WebP. Kết quả sẽ xuất hiện ở khung bên cạnh.")
                # Mở khối with để xếp thành phần theo hàng ngang bằng gr.Row(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                with gr.Row():
                    # Mở khối with để xếp thành phần theo cột bằng gr.Column(...); dữ liệu/tham số: elem_classes='studio-card'; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                    with gr.Column(elem_classes="studio-card"):
                        # Gán cho upload: tạo vùng tải file bằng gr.File(...); dữ liệu/tham số: label='Chọn ảnh để tự động nhận diện', file_count='single', file_types=['.jpg', '.jpeg', '.png', '.webp'], type='filepath'.
                        upload = gr.File(label="Chọn ảnh để tự động nhận diện", file_count="single",
                                         # Truyền file_types = ['.jpg', '.jpeg', '.png', '.webp'].
                                         file_types=[".jpg", ".jpeg", ".png", ".webp"], type="filepath")
                        # Gán cho message: tạo ô nhập hoặc hiển thị văn bản bằng gr.Textbox(...); dữ liệu/tham số: value='Chọn ảnh để bắt đầu.', label='Trạng thái', interactive=False.
                        message = gr.Textbox(value="Chọn ảnh để bắt đầu.", label="Trạng thái", interactive=False)
                    # Gán cho output: tạo vùng ảnh hoặc camera bằng gr.Image(...); dữ liệu/tham số: label='Kết quả nhận diện', format='png', height=camera_height.
                    output = gr.Image(label="Kết quả nhận diện", format="png", height=camera_height)
                # Mở khối with để tạo vùng nội dung có thể mở/thu gọn bằng gr.Accordion(...); dữ liệu/tham số: 'Chi tiết kết quả', open=False; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                with gr.Accordion("Chi tiết kết quả", open=False):
                    # Gán cho details: hiển thị dữ liệu JSON bằng gr.JSON(...); dữ liệu/tham số: label='Vị trí khuôn mặt và độ tương đồng'.
                    details = gr.JSON(label="Vị trí khuôn mặt và độ tương đồng")
                # Gắn callback vào sự kiện dữ liệu thay đổi qua upload.change(...); dữ liệu/tham số: recognize, upload, [output, details, message], api_visibility='private'.
                upload.change(recognize, upload, [output, details, message], api_visibility="private")
            # Mở khối with để tạo một tab giao diện bằng gr.Tab(...); dữ liệu/tham số: '3 · Webcam trực tiếp'; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
            with gr.Tab("3 · Webcam trực tiếp"):
                # Thêm tiêu đề và hướng dẫn vào giao diện bằng section_title(...); dữ liệu/tham số: 'Nhận diện trực tiếp', 'Bật camera, bấm Bắt đầu nhận diện rồi bấm ghi hình trong khung camera.'.
                section_title("Nhận diện trực tiếp",
                              # Tiếp phần biểu thức nhiều dòng: chuỗi 'Bật camera, bấm Bắt đầu nhận diện rồi bấm ghi hình trong khung ….
                              "Bật camera, bấm Bắt đầu nhận diện rồi bấm ghi hình trong khung camera.")
                # Gán cho enabled: tạo vùng trạng thái riêng của phiên giao diện bằng gr.State(...); dữ liệu/tham số: False.
                enabled = gr.State(False)
                # Gán cho timing: tạo vùng trạng thái riêng của phiên giao diện bằng gr.State(...); dữ liệu/tham số: {}.
                timing = gr.State({})
                # Mở khối with để xếp thành phần theo hàng ngang bằng gr.Row(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                with gr.Row():
                    # Gán cho start: tạo nút bấm bằng gr.Button(...); dữ liệu/tham số: 'Bắt đầu nhận diện', variant='primary'.
                    start = gr.Button("Bắt đầu nhận diện", variant="primary")
                    # Gán cho stop: tạo nút bấm bằng gr.Button(...); dữ liệu/tham số: 'Dừng nhận diện'.
                    stop = gr.Button("Dừng nhận diện")
                # Mở khối with để xếp thành phần theo hàng ngang bằng gr.Row(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                with gr.Row():
                    # Gán cho camera: tạo vùng ảnh hoặc camera bằng gr.Image(...); dữ liệu/tham số: label='Camera trình duyệt', sources=['webcam'], type='numpy', streaming=True.
                    camera = gr.Image(label="Camera trình duyệt", sources=["webcam"], type="numpy",
                                      # Truyền streaming (bật truyền/xử lý ảnh liên tục) = True.
                                      streaming=True, height=camera_height)
                    # Gán cho live: tạo vùng ảnh hoặc camera bằng gr.Image(...); dữ liệu/tham số: label='Nhận diện trực tiếp', format='jpeg', streaming=True, height=camera_height.
                    live = gr.Image(label="Nhận diện trực tiếp", format="jpeg", streaming=True, height=camera_height)
                # Gán cho live_message: tạo ô nhập hoặc hiển thị văn bản bằng gr.Textbox(...); dữ liệu/tham số: value='Đang chờ camera.', label='Trạng thái nhận diện', interactive=False.
                live_message = gr.Textbox(value="Đang chờ camera.", label="Trạng thái nhận diện", interactive=False)
                # Mở khối with để tạo vùng nội dung có thể mở/thu gọn bằng gr.Accordion(...); dữ liệu/tham số: 'Chi tiết camera và kết quả', open=False; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                with gr.Accordion("Chi tiết camera và kết quả", open=False):
                    # Gán cho camera_check: tạo nút bấm bằng gr.Button(...); dữ liệu/tham số: 'Kiểm tra quyền camera'.
                    camera_check = gr.Button("Kiểm tra quyền camera")
                    # Gán cho camera_message: tạo ô nhập hoặc hiển thị văn bản bằng gr.Textbox(...); dữ liệu/tham số: label='Trạng thái camera', interactive=False.
                    camera_message = gr.Textbox(label="Trạng thái camera", interactive=False)
                    # Gán cho live_details: hiển thị dữ liệu JSON bằng gr.JSON(...); dữ liệu/tham số: label='Kết quả frame gần nhất'.
                    live_details = gr.JSON(label="Kết quả frame gần nhất")
                # Gắn callback vào sự kiện bấm nút qua camera_check.click(...); dữ liệu/tham số: fn=None, outputs=camera_message, js=CAMERA_CHECK_JS, queue=False.
                camera_check.click(fn=None, outputs=camera_message, js=CAMERA_CHECK_JS, queue=False)
                # Gán cho stream_event: gắn hàm xử lý chuỗi frame camera qua camera.stream(...); dữ liệu/tham số: stream, [camera, enabled, timing], [live, live_details, timing, live_message], stream_every=0.25, time_limit=30, concurrency_limit=1, concurrency_id='face-model'.
                stream_event = camera.stream(stream, [camera, enabled, timing], [live, live_details, timing, live_message],
                                            # Truyền stream_every (khoảng thời gian gửi frame, tính bằng giây) = 0.25.
                                            stream_every=0.25, time_limit=30, concurrency_limit=1,
                                            # Truyền concurrency_id (tên nhóm dùng chung giới hạn đồng thời) = 'face-model'.
                                            concurrency_id="face-model", api_visibility="private")
                # Gắn callback vào sự kiện bấm nút qua start.click(...); dữ liệu/tham số: lambda: (True, {}, 'Đang nhận diện. Bật ghi hình trong khung camera nếu chưa có frame.'), outputs=[enabled, timing, live_message], queue=False, api_visibility='private'.
                start.click(lambda: (True, {}, "Đang nhận diện. Bật ghi hình trong khung camera nếu chưa có frame."),
                            # Truyền outputs (các thành phần nhận kết quả callback theo thứ tự) = [enabled, timing, live_message].
                            outputs=[enabled, timing, live_message], queue=False, api_visibility="private")
                # Gắn callback vào sự kiện bấm nút qua stop.click(...); dữ liệu/tham số: lambda: (False, {}, 'Đã dừng nhận diện.'), outputs=[enabled, timing, live_message], cancels=[stream_event], queue=False, api_visibility='private'.
                stop.click(lambda: (False, {}, "Đã dừng nhận diện."), outputs=[enabled, timing, live_message],
                           # Truyền cancels (những sự kiện đang chạy cần hủy) = [stream_event].
                           cancels=[stream_event], queue=False, api_visibility="private")
                # Xóa các phần tử hoặc đăng ký sự kiện xóa tùy đối tượng qua camera.clear(...); dữ liệu/tham số: lambda: (False, {}, 'Camera đã dừng.'), outputs=[enabled, timing, live_message], cancels=[stream_event], queue=False, api_visibility='private'.
                camera.clear(lambda: (False, {}, "Camera đã dừng."), outputs=[enabled, timing, live_message],
                             # Truyền cancels (những sự kiện đang chạy cần hủy) = [stream_event].
                             cancels=[stream_event], queue=False, api_visibility="private")
            # Mở khối with để tạo một tab giao diện bằng gr.Tab(...); dữ liệu/tham số: '4 · Danh sách đăng ký'; gọi tài nguyên là registrations_tab; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
            with gr.Tab("4 · Danh sách đăng ký") as registrations_tab:
                # Thêm tiêu đề và hướng dẫn vào giao diện bằng section_title(...); dữ liệu/tham số: 'Hồ sơ đã đăng ký', 'Xem các hồ sơ khuôn mặt và chọn hồ sơ bạn muốn xóa.'.
                section_title("Hồ sơ đã đăng ký", "Xem các hồ sơ khuôn mặt và chọn hồ sơ bạn muốn xóa.")
                # Mở khối with để xếp thành phần theo hàng ngang bằng gr.Row(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                with gr.Row():
                    # Mở khối with để xếp thành phần theo cột bằng gr.Column(...); dữ liệu/tham số: scale=3, elem_classes='studio-card'; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                    with gr.Column(scale=3, elem_classes="studio-card"):
                        # Gán cho registration_summary: hiển thị nội dung Markdown bằng gr.Markdown(...); dữ liệu/tham số: 'Đang tải danh sách…'.
                        registration_summary = gr.Markdown("Đang tải danh sách…")
                        # Gán cho registration_table: hiển thị bảng hồ sơ bằng gr.Dataframe(...); dữ liệu/tham số: value=[], headers=['ID đăng ký', 'Họ và tên', 'Số ảnh'], datatype=['str', 'str', 'number'], interactive=False.
                        registration_table = gr.Dataframe(
                            # Truyền value (giá trị ban đầu) = [].
                            value=[], headers=["ID đăng ký", "Họ và tên", "Số ảnh"],
                            # Truyền datatype = ['str', 'str', 'number'].
                            datatype=["str", "str", "number"], interactive=False, show_label=False,
                        # Đóng phần biểu thức/tham số nhiều dòng đang mở; các dấu ngoặc không tạo bước xử lý riêng.
                        )
                        # Gán cho registration_refresh: tạo nút bấm bằng gr.Button(...); dữ liệu/tham số: 'Làm mới danh sách', size='sm'.
                        registration_refresh = gr.Button("Làm mới danh sách", size="sm")
                    # Mở khối with để xếp thành phần theo cột bằng gr.Column(...); dữ liệu/tham số: scale=2, elem_classes='studio-card'; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                    with gr.Column(scale=2, elem_classes="studio-card"):
                        # Thêm tiêu đề và hướng dẫn vào giao diện bằng section_title(...); dữ liệu/tham số: 'Xóa đăng ký', 'Chọn hồ sơ theo tên hoặc ID, sau đó xác nhận xóa.', level=3.
                        section_title("Xóa đăng ký", "Chọn hồ sơ theo tên hoặc ID, sau đó xác nhận xóa.", level=3)
                        # Gán cho registration_choice: tạo danh sách chọn thả xuống bằng gr.Dropdown(...); dữ liệu/tham số: label='Hồ sơ cần xóa', choices=[], interactive=False, filterable=True.
                        registration_choice = gr.Dropdown(
                            # Truyền label (nhãn hiển thị cho người dùng) = 'Hồ sơ cần xóa'.
                            label="Hồ sơ cần xóa", choices=[], interactive=False, filterable=True,
                        # Đóng phần biểu thức/tham số nhiều dòng đang mở; các dấu ngoặc không tạo bước xử lý riêng.
                        )
                        # Gán cho registration_delete: tạo nút bấm bằng gr.Button(...); dữ liệu/tham số: 'Xóa đăng ký', variant='stop'.
                        registration_delete = gr.Button("Xóa đăng ký", variant="stop")
                        # Gán cho pending_registration: tạo vùng trạng thái riêng của phiên giao diện bằng gr.State(...); dữ liệu/tham số: None.
                        pending_registration = gr.State(None)
                        # Mở khối with để gom thành phần để điều khiển cùng lúc bằng gr.Group(...); dữ liệu/tham số: visible=False; gọi tài nguyên là registration_confirmation; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                        with gr.Group(visible=False) as registration_confirmation:
                            # Gán cho registration_question: tạo ô nhập hoặc hiển thị văn bản bằng gr.Textbox(...); dữ liệu/tham số: label='Xác nhận hồ sơ cần xóa', interactive=False.
                            registration_question = gr.Textbox(label="Xác nhận hồ sơ cần xóa", interactive=False)
                            # Mở khối with để xếp thành phần theo hàng ngang bằng gr.Row(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
                            with gr.Row():
                                # Gán cho registration_confirm: tạo nút bấm bằng gr.Button(...); dữ liệu/tham số: 'Xác nhận xóa', variant='stop'.
                                registration_confirm = gr.Button("Xác nhận xóa", variant="stop")
                                # Gán cho registration_cancel: tạo nút bấm bằng gr.Button(...); dữ liệu/tham số: 'Hủy xóa'.
                                registration_cancel = gr.Button("Hủy xóa")
                        # Gán cho registration_notice: tạo ô nhập hoặc hiển thị văn bản bằng gr.Textbox(...); dữ liệu/tham số: label='Trạng thái', interactive=False.
                        registration_notice = gr.Textbox(label="Trạng thái", interactive=False)
        # Mở khối with để tạo vùng nội dung có thể mở/thu gọn bằng gr.Accordion(...); dữ liệu/tham số: 'Thông tin hệ thống', open=False; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with gr.Accordion("Thông tin hệ thống", open=False):
            # Gán cho health: hiển thị dữ liệu JSON bằng gr.JSON(...); dữ liệu/tham số: label='Trạng thái hệ thống'.
            health = gr.JSON(label="Trạng thái hệ thống")
            # Gán cho refresh: tạo nút bấm bằng gr.Button(...); dữ liệu/tham số: 'Kiểm tra hệ thống'.
            refresh = gr.Button("Kiểm tra hệ thống")
        # Gán cho health_outputs: danh sách [health_summary, health] ([] giữ các phần tử theo thứ tự).
        health_outputs = [health_summary, health]
        # Gắn callback vào sự kiện bấm nút qua refresh.click(...); dữ liệu/tham số: show_health, outputs=health_outputs, queue=False, api_visibility='private'.
        refresh.click(show_health, outputs=health_outputs, queue=False, api_visibility="private")
        # Nạp dữ liệu hoặc gắn sự kiện mở trang qua demo.load(...); dữ liệu/tham số: show_health, outputs=health_outputs, api_visibility='private'.
        demo.load(show_health, outputs=health_outputs, api_visibility="private")
        # Gán cho registration_outputs: danh sách [registration_table, registration_choice, registration_summary, pending_registration… ([] giữ các phần tử theo thứ tự).
        registration_outputs = [registration_table, registration_choice, registration_summary,
                                # Tiếp phần biểu thức nhiều dòng: registration_confirmation.
                                pending_registration, registration_confirmation, registration_notice]
        # Gán cho registration_options: tạo từ điển khóa–giá trị bằng dict(...); dữ liệu/tham số: concurrency_id='registration-management', concurrency_limit=1, api_visibility='private'.
        registration_options = dict(concurrency_id="registration-management", concurrency_limit=1,
                                    # Truyền api_visibility (mức công khai của sự kiện trong API Gradio) = 'private'.
                                    api_visibility="private")
        # Gắn callback khi chọn tab qua registrations_tab.select(...); dữ liệu/tham số: refresh_registrations, outputs=registration_outputs, **=registration_options.
        registrations_tab.select(refresh_registrations, outputs=registration_outputs, **registration_options)
        # Gắn callback vào sự kiện bấm nút qua registration_refresh.click(...); dữ liệu/tham số: refresh_registrations, outputs=registration_outputs, **=registration_options.
        registration_refresh.click(refresh_registrations, outputs=registration_outputs, **registration_options)
        # then nối thao tác sau khi callback trước kết thúc: làm mới danh sách rồi cập nhật health.
        # Gán cho refreshed_after_save: nối tác vụ chạy sau sự kiện trước đó qua saved.then(...); dữ liệu/tham số: refresh_registrations, outputs=registration_outputs, **=registration_options.
        refreshed_after_save = saved.then(refresh_registrations, outputs=registration_outputs, **registration_options)
        # Nối tác vụ chạy sau sự kiện trước đó qua refreshed_after_save.then(...); dữ liệu/tham số: show_health, outputs=health_outputs, queue=False, api_visibility='private'.
        refreshed_after_save.then(show_health, outputs=health_outputs, queue=False, api_visibility="private")
        # Gán cho delete_prompt_outputs: danh sách [pending_registration, registration_confirmation, registration_question, registratio… ([] giữ các phần tử theo thứ tự).
        delete_prompt_outputs = [pending_registration, registration_confirmation, registration_question, registration_notice]
        # Gắn callback vào sự kiện bấm nút qua registration_delete.click(...); dữ liệu/tham số: request_registration_delete, inputs=registration_choice, outputs=delete_prompt_outputs, **=registration_options.
        registration_delete.click(request_registration_delete, inputs=registration_choice,
                                  # Truyền outputs (các thành phần nhận kết quả callback theo thứ tự) = delete_prompt_outputs.
                                  outputs=delete_prompt_outputs, **registration_options)
        # Gắn callback vào sự kiện bấm nút qua registration_cancel.click(...); dữ liệu/tham số: lambda: (None, gr.update(visible=False), '', ''), outputs=delete_prompt_outputs, **=registration_options.
        registration_cancel.click(lambda: (None, gr.update(visible=False), "", ""),
                                  # Truyền outputs (các thành phần nhận kết quả callback theo thứ tự) = delete_prompt_outputs.
                                  outputs=delete_prompt_outputs, **registration_options)
        # Gắn callback khi người dùng đổi lựa chọn qua registration_choice.input(...); dữ liệu/tham số: lambda: (None, gr.update(visible=False), '', ''), outputs=delete_prompt_outputs, **=registration_options.
        registration_choice.input(lambda: (None, gr.update(visible=False), "", ""),
                                  # Truyền outputs (các thành phần nhận kết quả callback theo thứ tự) = delete_prompt_outputs.
                                  outputs=delete_prompt_outputs, **registration_options)
        # Gán cho deleted: gắn callback vào sự kiện bấm nút qua registration_confirm.click(...); dữ liệu/tham số: confirm_registration_delete, inputs=pending_registration, outputs=registration_outputs, **=registration_options.
        deleted = registration_confirm.click(confirm_registration_delete, inputs=pending_registration,
                                             # Truyền outputs (các thành phần nhận kết quả callback theo thứ tự) = registration_outputs.
                                             outputs=registration_outputs, **registration_options)
        # Nối tác vụ chạy sau sự kiện trước đó qua deleted.then(...); dữ liệu/tham số: show_health, outputs=health_outputs, queue=False, api_visibility='private'.
        deleted.then(show_health, outputs=health_outputs, queue=False, api_visibility="private")
        # Hiển thị đoạn HTML bằng gr.HTML(...); dữ liệu/tham số: '<p class="studio-footer">Unknown hiển thị khi hệ thống chưa tìm được hồ sơ khớp. <a href="/docs">T….
        gr.HTML('<p class="studio-footer">Unknown hiển thị khi hệ thống chưa tìm được hồ sơ khớp. '
                # Tiếp chuỗi/dữ liệu của câu lệnh trên; Python tự nối các chuỗi liền kề trong ngoặc.
                '<a href="/docs">Tài liệu API</a></p>')
    # Hàng đợi giới hạn số yêu cầu chờ để tránh dồn quá nhiều tác vụ xử lý model.
    # Kết thúc hàm và trả bật hàng đợi và giới hạn yêu cầu chờ qua demo.queue(...); dữ liệu/tham số: max_size=8, default_concurrency_limit=1.
    return demo.queue(max_size=8, default_concurrency_limit=1)
