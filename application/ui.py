import os
from pathlib import Path

# No analytics requests or persistent Gradio run history for face uploads.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from .errors import ApplicationError
from .images import decode_image
from .visualization import annotate
from .webcam import process_frame
from .webcam_enrollment import LABELS, POSE_COUNT, POSES


UI_CSS = Path(__file__).with_name("ui.css").read_text(encoding="utf-8")
UI_THEME = gr.themes.Base(
    primary_hue="teal", secondary_hue="slate", neutral_hue="slate",
    font=("Segoe UI", "system-ui", "sans-serif"),
).set(
    body_background_fill="#eef2f7", body_background_fill_dark="#0b1320",
    background_fill_primary="#fff", background_fill_primary_dark="#111e30",
    block_border_color="#dbe3ee", block_border_color_dark="#263950",
    button_primary_background_fill="#0f766e", button_primary_background_fill_hover="#115e59",
    button_primary_text_color="#fff",
    button_primary_background_fill_dark="#5eead4", button_primary_background_fill_hover_dark="#99f6e4",
    button_primary_text_color_dark="#102e2c",
)

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

POSE_SHORT_LABELS = {"front": "Nhìn thẳng", "left": "Quay trái", "right": "Quay phải", "up": "Ngẩng đầu"}


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


def build_ui(service):
    enrollment = service.webcam_enrollment

    def preview_html(token=None, next_pose=None):
        import base64
        from io import BytesIO
        from PIL import Image
        captures = dict(enrollment.review(token)) if token else {}
        cards = []
        for number, pose in enumerate(POSES, 1):
            label = POSE_SHORT_LABELS[pose]
            if pose in captures:
                data = BytesIO()
                thumbnail = Image.fromarray(captures[pose])
                thumbnail.thumbnail((240, 240))
                thumbnail.save(data, format="JPEG")
                encoded = base64.b64encode(data.getvalue()).decode("ascii")
                visual = f'<img class="studio-pose-image" alt="{label}" src="data:image/jpeg;base64,{encoded}">'
                state, note = "is-done", "Đã chụp"
            else:
                visual = f'<div class="studio-pose-image" aria-hidden="true"><span>{number:02}</span></div>'
                state, note = ("is-next", "Góc hiện tại") if pose == next_pose else ("", "Chờ chụp")
            cards.append(f'<figure class="studio-pose {state}">{visual}'
                         f'<figcaption><strong>{label}</strong><small>{note}</small></figcaption></figure>')
        return '<div class="studio-poses">' + ''.join(cards) + '</div>'

    def show_health():
        result = service.health()
        ready = result["status"] == "ok"
        label = "Hệ thống sẵn sàng" if ready else "Kiểm tra trạng thái hệ thống"
        count = result["person_count"]
        people = f"{count} hồ sơ đã đăng ký" if count is not None else "Đang chờ dữ liệu hồ sơ"
        state = "" if ready else "is-alert"
        return (f'<div class="studio-health-bar" role="status"><span class="studio-health-pill {state}">'
                f'{label}</span><span>{people}</span></div>', result)

    def section_title(title, description, *, kicker=None, level=2):
        prefix = f'<p class="studio-kicker">{kicker}</p>' if kicker else ""
        gr.HTML(f'<div class="studio-section-title">{prefix}<h{level}>{title}</h{level}>'
                f'<p>{description}</p></div>')

    def load_registrations():
        try:
            people = service.persons()
        except ApplicationError as error:
            return [], gr.update(choices=[], value=None, interactive=False), error.message
        rows = [[person["person_id"], person.get("display_name", "—"), person["num_images"]] for person in people]
        choices = [
            (f'{person["display_name"]} · {person["person_id"]}' if person.get("display_name") else person["person_id"],
             person["person_id"])
            for person in people
        ]
        summary = f"{len(people)} hồ sơ đã đăng ký." if people else "Chưa có hồ sơ nào. Bạn có thể tạo hồ sơ trong tab Đăng ký webcam."
        return rows, gr.update(choices=choices, value=None, interactive=bool(choices)), summary

    def refresh_registrations():
        return (*load_registrations(), None, gr.update(visible=False), "")

    def request_registration_delete(person_id):
        if not person_id:
            return None, gr.update(visible=False), "", "Hãy chọn hồ sơ cần xóa."
        return person_id, gr.update(visible=True), f"Bạn có muốn xóa hồ sơ ID {person_id}?", ""

    def confirm_registration_delete(person_id):
        if not person_id:
            message = "Hãy chọn hồ sơ và mở xác nhận xóa."
        else:
            try:
                service.delete(person_id)
                message = f"Đã xóa hồ sơ ID {person_id}."
            except ApplicationError as error:
                message = error.message
        return (*load_registrations(), None, gr.update(visible=False), message)

    def enrollment_result(result, refresh_preview=True):
        state = {key: result[key] for key in ("session_id", "revision")}
        message = f'{result["progress"]}/{result["total"]} · {result["instruction"]}\n{result["message"]}'
        public = {k: v for k, v in result.items() if k != "session_id"}
        preview = preview_html(state["session_id"], result["next_pose"]) if refresh_preview else gr.skip()
        return (state, message, public, preview, False,
                gr.update(interactive=result["status"] == "review"), gr.update(interactive=False))

    def enrollment_error(state, error):
        if error.code == "session_missing":
            return ({}, error.message, error.payload(), preview_html(), False,
                    gr.update(interactive=False), gr.update(interactive=True))
        if error.code in {"stale_frame", "stale_review"} and state:
            try:
                return enrollment_result(enrollment.status(state["session_id"]))
            except ApplicationError as current_error:
                return enrollment_error(state, current_error)
        return (state, error.message, error.payload(), gr.skip(), False, gr.skip(), gr.skip())

    def start_enrollment(person_id, name, state):
        try:
            if state:
                raise ApplicationError("session_active", "Hủy phiên hiện tại trước khi đăng ký người khác.")
            return enrollment_result(enrollment.start(person_id, name))
        except ApplicationError as error:
            return enrollment_error(state, error)

    def enrollment_stream(frame, state):
        if not state:
            return (gr.skip(),) * 7
        try:
            status = enrollment.status(state["session_id"])
            if status["status"] in {"review", "paused"}:
                return (gr.skip(),) * 7
            result = enrollment.frame(state["session_id"], frame, state["revision"])
            return enrollment_result(result, refresh_preview=result.get("captured", False))
        except ApplicationError as error:
            if error.code == "session_missing":
                # An in-flight stream response must not overwrite the success or
                # cancellation message from a control that just ended the session.
                return (gr.skip(),) * 7
            return enrollment_error(state, error)

    def retake_enrollment(state, pose):
        try:
            return enrollment_result(enrollment.retake(state.get("session_id"), pose))
        except ApplicationError as error:
            return enrollment_error(state, error)

    def cancel_enrollment(state):
        result = enrollment.cancel(state.get("session_id"))
        message = (f'0/{result["total"]} · Đã hủy phiên. Chưa lưu danh tính.' if result["status"] == "cancelled"
                   else "Phiên đã kết thúc hoặc hết hạn.")
        return ({}, message, result, preview_html(), False,
                gr.update(interactive=False), gr.update(interactive=True))

    def pause_enrollment(state):
        if not state:
            return (gr.skip(),) * 7
        try:
            return enrollment_result(enrollment.pause(state["session_id"]))
        except ApplicationError as error:
            return enrollment_error(state, error)

    def resume_enrollment(state):
        try:
            return enrollment_result(enrollment.resume(state.get("session_id")))
        except ApplicationError as error:
            return enrollment_error(state, error)

    def save_enrollment(state, accepted):
        try:
            result = enrollment.confirm(state.get("session_id"), accepted, state.get("revision"))
            return ({}, f"{POSE_COUNT}/{POSE_COUNT} · Đã lưu. Chuyển sang Nhận diện ảnh hoặc Webcam để kiểm tra danh tính.", result,
                    '<p class="studio-footer">Đã lưu hồ sơ. Bắt đầu phiên mới để đăng ký thêm.</p>',
                    False, gr.update(interactive=False), gr.update(interactive=True))
        except ApplicationError as error:
            return enrollment_error(state, error)

    def enrollment_watchdog(state):
        if not state:
            return gr.skip()
        try:
            status = enrollment.status(state["session_id"])
            if status["status"] == "capturing" and "Không nhận được" in status["message"]:
                return f'{status["progress"]}/{status["total"]} · {status["instruction"]}\n{status["message"]}'
        except ApplicationError as error:
            return error.message
        return gr.skip()

    def read_upload(path):
        if path is None:
            raise ApplicationError("missing_image", "Hãy chọn ảnh.")
        with Path(path).open("rb") as stream:
            return decode_image(stream.read(service.settings.max_file_bytes + 1), service.settings)

    def recognize(path):
        try:
            image = read_upload(path)
            result = service.recognize(image)
            recognized = sum(face["status"] == "recognized" for face in result["faces"])
            message = (f'Nhận diện được {recognized}/{result["face_count"]} khuôn mặt.'
                       if result["faces"] else "Không tìm thấy khuôn mặt.")
            return annotate(image, result["faces"]), result, message
        except ApplicationError as error:
            return None, error.payload(), error.message

    def stream(frame, enabled, state):
        if not enabled:
            return gr.skip(), gr.skip(), state, "Đã dừng nhận diện. Dừng camera bằng nút trong khung camera."
        try:
            output, result, state = process_frame(service, frame, state)
            text = (f'{result["face_count"]} khuôn mặt · {result["frames_processed"]} khung hình đã xử lý · '
                    f'{result["stream_fps"]:.2f} FPS · {result["frame_latency_ms"]:.1f} ms mỗi khung')
            return output, result, state, text
        except ApplicationError as error:
            return None, error.payload(), state, error.message

    with gr.Blocks(title="Face Recognition Studio", analytics_enabled=False,
                   delete_cache=(60, 300)) as demo:
        gr.HTML(HERO, elem_id="studio-header")
        health_summary = gr.HTML('<div class="studio-health-bar">Đang kiểm tra hệ thống…</div>', elem_id="studio-health")
        camera_height = "clamp(260px, 42vw, 440px)"
        with gr.Tabs(elem_id="studio-tabs"):
            with gr.Tab(f"1 · Đăng ký webcam · {POSE_COUNT} góc"):
                section_title("Tạo hồ sơ khuôn mặt",
                              "Giữ yên theo hướng dẫn trên camera để hệ thống tự chụp từng góc.")
                enrollment_state = gr.State({}, time_to_live=1200,
                                            delete_callback=lambda state: enrollment.cancel(state.get("session_id")))
                with gr.Row(elem_id="enrollment-workspace"):
                    with gr.Column(elem_id="enrollment-camera-panel"):
                        enrollment_camera = gr.Image(
                            label="Camera đăng ký (không lật gương)", sources=["webcam"],
                            type="numpy", streaming=True, height=camera_height,
                            webcam_options=gr.WebcamOptions(
                                mirror=False, constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}},
                            ),
                        )
                        enrollment_message = gr.Textbox(
                            value=f"0/{POSE_COUNT} · Bắt đầu phiên để chụp góc nhìn thẳng.", label="Tiến độ & hướng dẫn",
                            interactive=False, show_label=False, container=False,
                            lines=2, max_lines=5, elem_id="enrollment-camera-status",
                        )
                    with gr.Column(elem_id="enrollment-form", elem_classes="studio-card"):
                        section_title("Thông tin đăng ký",
                                      "Nhập thông tin, bắt đầu phiên rồi bật camera và bấm ghi hình.",
                                      kicker="Bước 01", level=3)
                        enrollment_id = gr.Textbox(label="ID đăng ký", placeholder="Ví dụ: NV001", max_length=100)
                        enrollment_name = gr.Textbox(label="Họ và tên", placeholder="Nhập tên hiển thị", max_length=150)
                        enrollment_start = gr.Button("Bắt đầu phiên đăng ký", variant="primary")
                        enrollment_cancel = gr.Button("Hủy phiên đăng ký")
                        with gr.Accordion("Tùy chọn camera", open=False):
                            enrollment_check = gr.Button("Kiểm tra quyền camera đăng ký")
                            enrollment_camera_message = gr.Textbox(label="Quyền camera", interactive=False)
                            enrollment_resume = gr.Button("Tiếp tục sau khi bật lại camera")
                        gr.HTML('<p class="studio-footer">Quay trái và phải theo phía của bạn. Phiên có hiệu lực 20 phút.</p>')
                    with gr.Column(elem_id="enrollment-review", elem_classes="studio-card"):
                        section_title(f"Ảnh của {POSE_COUNT} góc", "Xem lại ảnh, chụp lại nếu cần rồi xác nhận lưu.",
                                      kicker="Bước 02", level=3)
                        enrollment_preview = gr.HTML(preview_html(), label="Xem lại ảnh đã chụp", elem_id="enrollment-preview")
                        with gr.Row():
                            retake_pose = gr.Dropdown(choices=[(LABELS[p], p) for p in POSES],
                                                     value="front", label="Góc cần chụp lại")
                            enrollment_retake = gr.Button("Chụp lại góc đã chọn")
                        enrollment_accept = gr.Checkbox(label=f"Tôi đã xem lại {POSE_COUNT} ảnh và đồng ý lưu hồ sơ", value=False)
                        enrollment_save = gr.Button("Xác nhận lưu đăng ký", variant="primary", interactive=False)
                with gr.Accordion("Chi tiết tư thế và chất lượng", open=False):
                    enrollment_details = gr.JSON(label="Tư thế, chất lượng & trạng thái phiên")
                enrollment_check.click(fn=None, outputs=enrollment_camera_message,
                                       js=CAMERA_CHECK_JS.replace("Bắt đầu nhận diện", "Bắt đầu phiên đăng ký"), queue=False)
                enrollment_outputs = [enrollment_state, enrollment_message, enrollment_details, enrollment_preview,
                                      enrollment_accept, enrollment_save, enrollment_start]
                event_options = dict(api_visibility="private", concurrency_limit=1, concurrency_id="webcam-enrollment-controls")
                enrollment_start.click(start_enrollment, [enrollment_id, enrollment_name, enrollment_state], enrollment_outputs, **event_options)
                enrollment_stream_event = enrollment_camera.stream(
                    enrollment_stream, [enrollment_camera, enrollment_state], enrollment_outputs,
                    stream_every=0.25, time_limit=1200, api_visibility="private",
                    concurrency_limit=enrollment.policy.max_sessions, concurrency_id="webcam-enrollment-stream")
                enrollment_retake.click(retake_enrollment, [enrollment_state, retake_pose], enrollment_outputs, **event_options)
                enrollment_cancel.click(cancel_enrollment, enrollment_state, enrollment_outputs,
                                        cancels=[enrollment_stream_event], **event_options)
                enrollment_camera.clear(pause_enrollment, enrollment_state, enrollment_outputs,
                                        cancels=[enrollment_stream_event], **event_options)
                enrollment_resume.click(resume_enrollment, enrollment_state, enrollment_outputs, **event_options)
                saved = enrollment_save.click(save_enrollment, [enrollment_state, enrollment_accept], enrollment_outputs, **event_options)
                enrollment_timer = gr.Timer(1)
                enrollment_timer.tick(enrollment_watchdog, enrollment_state, enrollment_message,
                                      queue=False, api_visibility="private")
            with gr.Tab("2 · Nhận diện ảnh"):
                section_title("Đối chiếu từ ảnh", "Chọn một ảnh JPG, PNG hoặc WebP. Kết quả sẽ xuất hiện ở khung bên cạnh.")
                with gr.Row():
                    with gr.Column(elem_classes="studio-card"):
                        upload = gr.File(label="Chọn ảnh để tự động nhận diện", file_count="single",
                                         file_types=[".jpg", ".jpeg", ".png", ".webp"], type="filepath")
                        message = gr.Textbox(value="Chọn ảnh để bắt đầu.", label="Trạng thái", interactive=False)
                    output = gr.Image(label="Kết quả nhận diện", format="png", height=camera_height)
                with gr.Accordion("Chi tiết kết quả", open=False):
                    details = gr.JSON(label="Vị trí khuôn mặt và độ tương đồng")
                upload.change(recognize, upload, [output, details, message], api_visibility="private")
            with gr.Tab("3 · Webcam trực tiếp"):
                section_title("Nhận diện trực tiếp",
                              "Bật camera, bấm Bắt đầu nhận diện rồi bấm ghi hình trong khung camera.")
                enabled = gr.State(False)
                timing = gr.State({})
                with gr.Row():
                    start = gr.Button("Bắt đầu nhận diện", variant="primary")
                    stop = gr.Button("Dừng nhận diện")
                with gr.Row():
                    camera = gr.Image(label="Camera trình duyệt", sources=["webcam"], type="numpy",
                                      streaming=True, height=camera_height)
                    live = gr.Image(label="Nhận diện trực tiếp", format="jpeg", streaming=True, height=camera_height)
                live_message = gr.Textbox(value="Đang chờ camera.", label="Trạng thái nhận diện", interactive=False)
                with gr.Accordion("Chi tiết camera và kết quả", open=False):
                    camera_check = gr.Button("Kiểm tra quyền camera")
                    camera_message = gr.Textbox(label="Trạng thái camera", interactive=False)
                    live_details = gr.JSON(label="Kết quả frame gần nhất")
                camera_check.click(fn=None, outputs=camera_message, js=CAMERA_CHECK_JS, queue=False)
                stream_event = camera.stream(stream, [camera, enabled, timing], [live, live_details, timing, live_message],
                                            stream_every=0.25, time_limit=30, concurrency_limit=1,
                                            concurrency_id="face-model", api_visibility="private")
                start.click(lambda: (True, {}, "Đang nhận diện. Bật ghi hình trong khung camera nếu chưa có frame."),
                            outputs=[enabled, timing, live_message], queue=False, api_visibility="private")
                stop.click(lambda: (False, {}, "Đã dừng nhận diện."), outputs=[enabled, timing, live_message],
                           cancels=[stream_event], queue=False, api_visibility="private")
                camera.clear(lambda: (False, {}, "Camera đã dừng."), outputs=[enabled, timing, live_message],
                             cancels=[stream_event], queue=False, api_visibility="private")
            with gr.Tab("4 · Danh sách đăng ký") as registrations_tab:
                section_title("Hồ sơ đã đăng ký", "Xem các hồ sơ khuôn mặt và chọn hồ sơ bạn muốn xóa.")
                with gr.Row():
                    with gr.Column(scale=3, elem_classes="studio-card"):
                        registration_summary = gr.Markdown("Đang tải danh sách…")
                        registration_table = gr.Dataframe(
                            value=[], headers=["ID đăng ký", "Họ và tên", "Số ảnh"],
                            datatype=["str", "str", "number"], interactive=False, show_label=False,
                        )
                        registration_refresh = gr.Button("Làm mới danh sách", size="sm")
                    with gr.Column(scale=2, elem_classes="studio-card"):
                        section_title("Xóa đăng ký", "Chọn hồ sơ theo tên hoặc ID, sau đó xác nhận xóa.", level=3)
                        registration_choice = gr.Dropdown(
                            label="Hồ sơ cần xóa", choices=[], interactive=False, filterable=True,
                        )
                        registration_delete = gr.Button("Xóa đăng ký", variant="stop")
                        pending_registration = gr.State(None)
                        with gr.Group(visible=False) as registration_confirmation:
                            registration_question = gr.Textbox(label="Xác nhận hồ sơ cần xóa", interactive=False)
                            with gr.Row():
                                registration_confirm = gr.Button("Xác nhận xóa", variant="stop")
                                registration_cancel = gr.Button("Hủy xóa")
                        registration_notice = gr.Textbox(label="Trạng thái", interactive=False)
        with gr.Accordion("Thông tin hệ thống", open=False):
            health = gr.JSON(label="Trạng thái hệ thống")
            refresh = gr.Button("Kiểm tra hệ thống")
        health_outputs = [health_summary, health]
        refresh.click(show_health, outputs=health_outputs, queue=False, api_visibility="private")
        demo.load(show_health, outputs=health_outputs, api_visibility="private")
        registration_outputs = [registration_table, registration_choice, registration_summary,
                                pending_registration, registration_confirmation, registration_notice]
        registration_options = dict(concurrency_id="registration-management", concurrency_limit=1,
                                    api_visibility="private")
        registrations_tab.select(refresh_registrations, outputs=registration_outputs, **registration_options)
        registration_refresh.click(refresh_registrations, outputs=registration_outputs, **registration_options)
        refreshed_after_save = saved.then(refresh_registrations, outputs=registration_outputs, **registration_options)
        refreshed_after_save.then(show_health, outputs=health_outputs, queue=False, api_visibility="private")
        delete_prompt_outputs = [pending_registration, registration_confirmation, registration_question, registration_notice]
        registration_delete.click(request_registration_delete, inputs=registration_choice,
                                  outputs=delete_prompt_outputs, **registration_options)
        registration_cancel.click(lambda: (None, gr.update(visible=False), "", ""),
                                  outputs=delete_prompt_outputs, **registration_options)
        registration_choice.input(lambda: (None, gr.update(visible=False), "", ""),
                                  outputs=delete_prompt_outputs, **registration_options)
        deleted = registration_confirm.click(confirm_registration_delete, inputs=pending_registration,
                                             outputs=registration_outputs, **registration_options)
        deleted.then(show_health, outputs=health_outputs, queue=False, api_visibility="private")
        gr.HTML('<p class="studio-footer">Unknown hiển thị khi hệ thống chưa tìm được hồ sơ khớp. '
                '<a href="/docs">Tài liệu API</a></p>')
    return demo.queue(max_size=8, default_concurrency_limit=1)
