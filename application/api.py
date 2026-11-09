# Tầng HTTP: nhận yêu cầu bằng FastAPI, kiểm tra dữ liệu rồi gọi lớp dịch vụ xử lý.
# Lấy asynccontextmanager từ contextlib để quản lý tài nguyên bằng with, yield và thao tác dọn tự động.
from contextlib import asynccontextmanager
# Lấy Annotated từ typing để ghi gợi ý kiểu và mô tả bổ sung cho tham số.
from typing import Annotated

# Lấy FastAPI, File, Form, UploadFile từ fastapi để tạo API và khai báo tham số lấy từ form/file upload.
from fastapi import FastAPI, File, Form, UploadFile
# Lấy BaseModel, Field từ pydantic để khai báo cấu trúc và kiểm tra dữ liệu đầu vào.
from pydantic import BaseModel, Field
# Lấy JSONResponse, RedirectResponse từ fastapi.responses để tạo phản hồi JSON và chuyển hướng trang.
from fastapi.responses import JSONResponse, RedirectResponse
# Lấy run_in_threadpool từ starlette.concurrency để chuyển tác vụ đồng bộ nặng sang nhóm luồng phụ.
from starlette.concurrency import run_in_threadpool

# Lấy ApplicationError từ .errors; dấu . chỉ module cùng package.
from .errors import ApplicationError
# Lấy decode_image từ .images; dấu . chỉ module cùng package.
from .images import decode_image
# Lấy ApplicationService từ .service; dấu . chỉ module cùng package.
from .service import ApplicationService


# Lớp chặn trước FastAPI: giới hạn tổng dữ liệu upload, kể cả khi gửi nhiều phần.
# Định nghĩa lớp BodyLimitMiddleware; lớp gom dữ liệu và các hàm liên quan.
class BodyLimitMiddleware:
    # Chuỗi tài liệu (docstring) mô tả cách dùng; Python lưu nội dung này trong __doc__.
    """Bound multipart body before parsing, including chunked requests."""

    # Khởi tạo đối tượng và giữ các phụ thuộc/trạng thái cần dùng trong những lần gọi sau.
    # Định nghĩa hàm __init__(self, app, limit). self là đối tượng đang gọi phương thức.
    def __init__(self, app, limit):
        # Tách kết quả theo thứ tự và gán cho (self.app, self.limit): bộ giá trị (app, limit) (tuple giữ thứ tự, không gán lại từng phần tử).
        self.app, self.limit = app, limit

    # Cho phép gọi đối tượng như hàm; middleware nhận và chuyển tiếp các sự kiện ASGI.
    # Định nghĩa hàm __call__(self, scope, receive, send). self là đối tượng đang gọi phương thức. async cho phép chờ tác vụ bằng await.
    async def __call__(self, scope, receive, send):
        # Chỉ chạy khối bên dưới khi scope['type'] != 'http' or not (scope['path'] == '/recognize' or scope['path'].endswith('/upload') or scope['path'].startswi… cho kết quả đúng.
        if scope["type"] != "http" or not (
            # Tiếp phần biểu thức nhiều dòng: kết hợp bằng or: lấy giá trị đúng đầu tiên nếu có trong scope['path'] == '/recognize' or scope['path'].endswith('/upload') or scope['path'].….
            scope["path"] == "/recognize" or scope["path"].endswith("/upload")
            # Bổ sung trường hợp được chấp nhận bằng or: or scope["path"].startswith("/enrollment-sessions").
            or scope["path"].startswith("/enrollment-sessions")
        # Đóng phần biểu thức/tham số nhiều dòng đang mở; các dấu ngoặc không tạo bước xử lý riêng.
        ):
            # Kết thúc hàm và trả chờ tác vụ bất đồng bộ hoàn tất: gọi self.app(...) với scope, receive, send.
            return await self.app(scope, receive, send)
        # Lấy danh sách cặp header từ ASGI, mặc định []; chuyển thành dict để tra content-length nhanh.
        headers = dict(scope.get("headers", []))
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Đọc Content-Length dạng byte và đổi thành số nguyên; b"0" là mặc định khi không có header.
            length = int(headers.get(b"content-length", b"0"))
        # Bắt lỗi ValueError để xử lý trong khối bên dưới.
        except ValueError:
            # Khi Content-Length không đọc được, cố ý đặt lớn hơn giới hạn để đi vào nhánh từ chối upload.
            length = self.limit + 1
        # Đọc và đếm byte thực tế; không chỉ tin Content-Length do phía gửi cung cấp.
        # Gán cho body: tạo vùng byte có thể bổ sung dữ liệu bằng bytearray(...).
        body = bytearray()
        # Chỉ chạy khối bên dưới khi length <= self.limit cho kết quả đúng.
        if length <= self.limit:
            # Lặp lại khối bên dưới chừng nào True còn đúng.
            while True:
                # Gán cho event: chờ tác vụ bất đồng bộ hoàn tất: gọi receive(...).
                event = await receive()
                # Chỉ chạy khối bên dưới khi event['type'] == 'http.disconnect' cho kết quả đúng.
                if event["type"] == "http.disconnect":
                    # Kết thúc hàm, trả None vì không kèm giá trị.
                    return
                # Ghép các byte vừa nhận vào phần thân; nếu sự kiện không có body thì thêm chuỗi byte rỗng.
                body.extend(event.get("body", b""))
                # Chỉ chạy khối bên dưới khi len(body) > self.limit or not event.get('more_body', False) cho kết quả đúng.
                if len(body) > self.limit or not event.get("more_body", False):
                    # Thoát ngay khỏi vòng lặp gần nhất để tiếp tục phần sau vòng lặp.
                    break
        # Chỉ chạy khối bên dưới khi length > self.limit or len(body) > self.limit cho kết quả đúng.
        if length > self.limit or len(body) > self.limit:
            # Gán cho response: tạo phản hồi JSON bằng JSONResponse(...); dữ liệu/tham số: ApplicationError('request_too_large', 'Tổng upload tối đa 32 MiB.', 413).payload() (đưa lỗi về từ điển để trả JSON qua ApplicationError('request_too_large', 'Tổng upload tối đa 32 MiB.', 413).payload(...)), 413.
            response = JSONResponse(ApplicationError("request_too_large", "Tổng upload tối đa 32 MiB.", 413).payload(), 413)
            # Kết thúc hàm và trả chờ tác vụ bất đồng bộ hoàn tất: gọi response(...) với scope, receive, send.
            return await response(scope, receive, send)
        # Phần thân đã đọc phải được phát lại đúng một lần để FastAPI còn phân tích được upload.
        # Ban đầu phần thân đã đệm chưa được chuyển cho FastAPI; cờ này giúp chỉ phát lại một lần.
        delivered = False

        # Phát lại phần thân yêu cầu đã kiểm tra cho bộ phân tích HTTP phía sau.
        # Định nghĩa hàm bounded_receive(). async cho phép chờ tác vụ bằng await.
        async def bounded_receive():
            # Dùng biến delivered của hàm bao ngoài; phép gán sẽ cập nhật biến đó.
            nonlocal delivered
            # Chỉ chạy khối bên dưới khi not delivered cho kết quả đúng.
            if not delivered:
                # Đánh dấu đã phát lại phần thân; lần receive tiếp theo sẽ đọc từ nguồn ASGI gốc.
                delivered = True
                # Kết thúc hàm và trả từ điển {'type': 'http.request', 'body': bytes(body), 'more_body': False} (ánh xạ khóa → giá trị).
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            # Kết thúc hàm và trả chờ tác vụ bất đồng bộ hoàn tất: gọi receive(...).
            return await receive()

        # Chờ tác vụ bất đồng bộ hoàn tất: gọi self.app(...) với scope, bounded_receive, send.
        await self.app(scope, bounded_receive, send)


# Pydantic kiểm tra kiểu và độ dài ID/tên trong JSON bắt đầu đăng ký.
# Định nghĩa lớp EnrollmentStart kế thừa BaseModel; lớp gom dữ liệu và các hàm liên quan.
class EnrollmentStart(BaseModel):
    # Khai báo person_id có kiểu gợi ý str; gán khai báo điều kiện kiểm tra trường dữ liệu bằng Field(...); dữ liệu/tham số: max_length=100.
    person_id: str = Field(max_length=100)
    # Khai báo display_name có kiểu gợi ý str; gán khai báo điều kiện kiểm tra trường dữ liệu bằng Field(...); dữ liệu/tham số: max_length=150.
    display_name: str = Field(max_length=150)


# JSON xác nhận gồm sự đồng ý và revision của bộ ảnh đang được xem.
# Định nghĩa lớp EnrollmentConfirmation kế thừa BaseModel; lớp gom dữ liệu và các hàm liên quan.
class EnrollmentConfirmation(BaseModel):
    # Khai báo confirmed có kiểu gợi ý bool; gán False (sai/tắt).
    confirmed: bool = False
    # Khai báo revision có kiểu gợi ý int; giá trị được cung cấp khi tạo đối tượng.
    revision: int


# Tạo FastAPI, đăng ký xử lý lỗi/các đường dẫn và tùy chọn gắn giao diện Gradio.
# Định nghĩa hàm create_app(service). Mặc định: service=None. Sau dấu * phải truyền theo tên: with_ui=False.
def create_app(service=None, *, with_ui=False):
    # Dùng service được truyền vào nếu có giá trị; nếu không thì tạo dịch vụ mới. Đây là cách hỗ trợ truyền model giả khi test.
    service = service or ApplicationService()

    # Nạp model khi máy chủ khởi động; threadpool giúp tác vụ đồng bộ không chặn vòng lặp async.
    # Decorator asynccontextmanager: biến hàm có yield thành ngữ cảnh dùng với with.
    @asynccontextmanager
    # Định nghĩa hàm lifespan(app). async cho phép chờ tác vụ bằng await.
    async def lifespan(app):
        # Chờ tác vụ bất đồng bộ hoàn tất: chạy hàm đồng bộ trong luồng phụ để không chặn vòng lặp async bằng run_in_threadpool(...).
        await run_in_threadpool(service.start)
        # Tạm nhường quyền chạy cho thân khối with; phần sau yield tiếp tục khi khối with kết thúc.
        yield

    # Gán cho app: tạo ứng dụng API bằng FastAPI(...); dữ liệu/tham số: title='MTCNN + FaceNet Application', version='1.0.0', lifespan=lifespan.
    app = FastAPI(title="MTCNN + FaceNet Application", version="1.0.0", lifespan=lifespan)
    # Gắn dịch vụ vào vùng state của FastAPI để mã khác có thể truy cập cùng một đối tượng.
    app.state.service = service
    # Gọi app.add_middleware(...) với BodyLimitMiddleware; dữ liệu/tham số: BodyLimitMiddleware, limit=service.settings.max_request_bytes.
    app.add_middleware(BodyLimitMiddleware, limit=service.settings.max_request_bytes)

    # Chuyển ApplicationError thành JSON với mã trạng thái HTTP tương ứng.
    # Decorator @app.exception_handler(ApplicationError): gắn hàm ngay bên dưới với đường dẫn HTTP hoặc bộ xử lý lỗi này.
    @app.exception_handler(ApplicationError)
    # Định nghĩa hàm application_error(request, error). async cho phép chờ tác vụ bằng await.
    async def application_error(request, error):
        # Kết thúc hàm và trả tạo phản hồi JSON bằng JSONResponse(...); dữ liệu/tham số: error.payload() (đưa lỗi về từ điển để trả JSON qua error.payload(...)), status_code=error.status.
        return JSONResponse(error.payload(), status_code=error.status)

    # Tổng hợp tình trạng model và kho hồ sơ để phía gọi biết hệ thống có sẵn sàng hay không.
    # Decorator @app.get("/health"): gắn hàm ngay bên dưới với đường dẫn HTTP hoặc bộ xử lý lỗi này.
    @app.get("/health")
    # Định nghĩa hàm health().
    def health():
        # Gán cho result: đọc tình trạng hệ thống qua service.health(...).
        result = service.health()
        # Kết thúc hàm và trả tạo phản hồi JSON bằng JSONResponse(...); dữ liệu/tham số: result, status_code=200 if result['status'] == 'ok' else 503.
        return JSONResponse(result, status_code=200 if result["status"] == "ok" else 503)

    # Đọc danh sách hồ sơ; bổ sung tên hiển thị nếu có trong kho tên đi kèm.
    # Decorator @app.get("/persons"): gắn hàm ngay bên dưới với đường dẫn HTTP hoặc bộ xử lý lỗi này.
    @app.get("/persons")
    # Định nghĩa hàm persons().
    def persons():
        # Kết thúc hàm và trả từ điển {'persons': service.persons()} (ánh xạ khóa → giá trị).
        return {"persons": service.persons()}

    # Xóa hồ sơ theo ID; báo lỗi nếu không tìm thấy người cần xóa.
    # Decorator @app.delete("/persons/{person_id}"): gắn hàm ngay bên dưới với đường dẫn HTTP hoặc bộ xử lý lỗi này.
    @app.delete("/persons/{person_id}")
    # Định nghĩa hàm delete(person_id).
    def delete(person_id: str):
        # Kết thúc hàm và trả gửi lệnh xóa theo ID hoặc URL qua service.delete(...); dữ liệu/tham số: person_id.
        return service.delete(person_id)

    # Đọc tối đa giới hạn cộng 1 byte để phát hiện quá dung lượng, rồi giải mã ngoài vòng lặp async.
    # Định nghĩa hàm read_image(upload). async cho phép chờ tác vụ bằng await.
    async def read_image(upload):
        # Gán cho data: chờ tác vụ bất đồng bộ hoàn tất: đọc dữ liệu từ luồng qua upload.read(...).
        data = await upload.read(service.settings.max_file_bytes + 1)
        # Kết thúc hàm và trả chờ tác vụ bất đồng bộ hoàn tất: chạy hàm đồng bộ trong luồng phụ để không chặn vòng lặp async bằng run_in_threadpool(...).
        return await run_in_threadpool(decode_image, data, service.settings)

    # Gán cho enrollment: service.webcam_enrollment (đọc thuộc tính).
    enrollment = service.webcam_enrollment

    # Bắt đầu phiên chụp cho ID và tên được cung cấp.
    # Decorator @app.post("/enrollment-sessions", status_code=201): gắn hàm ngay bên dưới với đường dẫn HTTP hoặc bộ xử lý lỗi này.
    @app.post("/enrollment-sessions", status_code=201)
    # Định nghĩa hàm start_enrollment(body).
    def start_enrollment(body: EnrollmentStart):
        # Kết thúc hàm và trả khởi động model hoặc bắt đầu phiên theo đối tượng nhận lệnh qua enrollment.start(...); dữ liệu/tham số: body.person_id, body.display_name.
        return enrollment.start(body.person_id, body.display_name)

    # Trả tiến độ của phiên được xác định bằng token.
    # Decorator @app.get("/enrollment-sessions/{token}"): gắn hàm ngay bên dưới với đường dẫn HTTP hoặc bộ xử lý lỗi này.
    @app.get("/enrollment-sessions/{token}")
    # Định nghĩa hàm enrollment_status(token).
    def enrollment_status(token: str):
        # Kết thúc hàm và trả lấy tiến độ phiên hiện tại qua enrollment.status(...); dữ liệu/tham số: token.
        return enrollment.status(token)

    # Nhận frame upload cùng revision; luôn đóng file upload sau khi xử lý.
    # Decorator @app.post("/enrollment-sessions/{token}/frames"): gắn hàm ngay bên dưới với đường dẫn HTTP hoặc bộ xử lý lỗi này.
    @app.post("/enrollment-sessions/{token}/frames")
    # Định nghĩa hàm enrollment_frame(token, revision, image). async cho phép chờ tác vụ bằng await.
    async def enrollment_frame(token: str, revision: Annotated[int, Form()], image: Annotated[UploadFile, File()]):
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
            try:
                # Gán cho rgb: chờ tác vụ bất đồng bộ hoàn tất: đọc upload có giới hạn rồi giải mã ảnh bằng read_image(...).
                rgb = await read_image(image)
            # Bắt lỗi ApplicationError để xử lý trong khối bên dưới.
            except ApplicationError:
                # Frame không hợp lệ cũng phải xóa chuỗi ổn định trước đó.
                # Chờ tác vụ bất đồng bộ hoàn tất: chạy hàm đồng bộ trong luồng phụ để không chặn vòng lặp async bằng run_in_threadpool(...).
                await run_in_threadpool(enrollment.frame, token, None, revision)
                # Ném lại lỗi đang xử lý để phía gọi biết thao tác chưa thành công.
                raise
            # Kết thúc hàm và trả chờ tác vụ bất đồng bộ hoàn tất: chạy hàm đồng bộ trong luồng phụ để không chặn vòng lặp async bằng run_in_threadpool(...).
            return await run_in_threadpool(enrollment.frame, token, rgb, revision)
        # Khối finally luôn chạy sau try, kể cả khi có lỗi hoặc return; dùng để dọn tài nguyên.
        finally:
            # finally bảo đảm đóng upload cả khi giải mã hoặc xử lý model thất bại.
            # Chờ đóng file upload trong finally; tài nguyên được giải phóng cả khi nhận diện hoặc giải mã bị lỗi.
            await image.close()

    # Chuyển ảnh xem lại thành JPEG Base64 để trình duyệt hiển thị trực tiếp.
    # Decorator @app.get("/enrollment-sessions/{token}/review"): gắn hàm ngay bên dưới với đường dẫn HTTP hoặc bộ xử lý lỗi này.
    @app.get("/enrollment-sessions/{token}/review")
    # Định nghĩa hàm enrollment_review(token).
    def enrollment_review(token: str):
        # Nạp base64 để đưa byte ảnh về chuỗi để nhúng trong HTML hoặc JSON.
        import base64
        # Lấy BytesIO từ io để đọc/ghi byte ảnh trong bộ nhớ bằng BytesIO.
        from io import BytesIO
        # Lấy Image từ PIL để đọc/ghi ảnh, xử lý hướng EXIF, chọn font và vẽ lên ảnh bằng Pillow.
        from PIL import Image
        # Gán cho items: danh sách [] ([] giữ các phần tử theo thứ tự).
        items = []
        # Gán cho snapshot: lấy ảnh xem lại kèm revision qua enrollment.review_snapshot(...); dữ liệu/tham số: token.
        snapshot = enrollment.review_snapshot(token)
        # Lặp qua snapshot['images']; mỗi lượt gán phần tử cho (pose, image).
        for pose, image in snapshot["images"]:
            # Gán cho data: tạo luồng byte trong bộ nhớ thay vì file trên ổ đĩa bằng BytesIO(...).
            data = BytesIO()
            # Mã hóa và lưu ảnh qua Image.fromarray(image).save(...); dữ liệu/tham số: data, format='JPEG'.
            Image.fromarray(image).save(data, format="JPEG")
            # Thêm một phần tử vào cuối danh sách qua items.append(...); dữ liệu/tham số: {'pose': pose, 'image': 'data:image/jpeg;base64,' + base64.b64encode(data.getvalue()).decode('ascii….
            items.append({"pose": pose, "image": "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode("ascii")})
        # Kết thúc hàm và trả từ điển {'revision': snapshot['revision'], 'images': items} (ánh xạ khóa → giá trị).
        return {"revision": snapshot["revision"], "images": items}

    # Yêu cầu chụp lại một góc trong phiên đăng ký hiện tại.
    # Decorator @app.post("/enrollment-sessions/{token}/retake/{pose}"): gắn hàm ngay bên dưới với đường dẫn HTTP hoặc bộ xử lý lỗi này.
    @app.post("/enrollment-sessions/{token}/retake/{pose}")
    # Định nghĩa hàm retake_enrollment(token, pose).
    def retake_enrollment(token: str, pose: str):
        # Kết thúc hàm và trả yêu cầu chụp lại góc đã chọn qua enrollment.retake(...); dữ liệu/tham số: token, pose.
        return enrollment.retake(token, pose)

    # Tạm dừng phiên khi camera dừng hoặc người dùng ngắt camera.
    # Decorator @app.post("/enrollment-sessions/{token}/pause"): gắn hàm ngay bên dưới với đường dẫn HTTP hoặc bộ xử lý lỗi này.
    @app.post("/enrollment-sessions/{token}/pause")
    # Định nghĩa hàm pause_enrollment(token).
    def pause_enrollment(token: str):
        # Kết thúc hàm và trả tạm dừng phiên chụp qua enrollment.pause(...); dữ liệu/tham số: token.
        return enrollment.pause(token)

    # Tiếp tục chụp sau khi bật lại camera.
    # Decorator @app.post("/enrollment-sessions/{token}/resume"): gắn hàm ngay bên dưới với đường dẫn HTTP hoặc bộ xử lý lỗi này.
    @app.post("/enrollment-sessions/{token}/resume")
    # Định nghĩa hàm resume_enrollment(token).
    def resume_enrollment(token: str):
        # Kết thúc hàm và trả tiếp tục phiên chụp qua enrollment.resume(...); dữ liệu/tham số: token.
        return enrollment.resume(token)

    # Hủy phiên và trả trạng thái để giao diện dọn dữ liệu đang hiển thị.
    # Decorator @app.delete("/enrollment-sessions/{token}"): gắn hàm ngay bên dưới với đường dẫn HTTP hoặc bộ xử lý lỗi này.
    @app.delete("/enrollment-sessions/{token}")
    # Định nghĩa hàm cancel_enrollment(token).
    def cancel_enrollment(token: str):
        # Kết thúc hàm và trả hủy phiên và ảnh tạm qua enrollment.cancel(...); dữ liệu/tham số: token.
        return enrollment.cancel(token)

    # Gửi sự đồng ý và revision xuống tầng đăng ký để kiểm tra trước khi lưu.
    # Decorator @app.post("/enrollment-sessions/{token}/confirm", status_code=201): gắn hàm ngay bên dưới với đường dẫn HTTP hoặc bộ xử lý lỗi này.
    @app.post("/enrollment-sessions/{token}/confirm", status_code=201)
    # Định nghĩa hàm confirm_enrollment(token, body).
    def confirm_enrollment(token: str, body: EnrollmentConfirmation):
        # Kết thúc hàm và trả xác nhận lưu bộ ảnh đã xem qua enrollment.confirm(...); dữ liệu/tham số: token, body.confirmed, body.revision.
        return enrollment.confirm(token, body.confirmed, body.revision)

    # Chạy nhận diện và trả kết quả cho từng khuôn mặt cùng thời gian xử lý.
    # Decorator @app.post("/recognize"): gắn hàm ngay bên dưới với đường dẫn HTTP hoặc bộ xử lý lỗi này.
    @app.post("/recognize")
    # Định nghĩa hàm recognize(image). async cho phép chờ tác vụ bằng await.
    async def recognize(image: Annotated[UploadFile, File()]):
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Gán cho rgb: chờ tác vụ bất đồng bộ hoàn tất: đọc upload có giới hạn rồi giải mã ảnh bằng read_image(...).
            rgb = await read_image(image)
            # Kết thúc hàm và trả chờ tác vụ bất đồng bộ hoàn tất: chạy hàm đồng bộ trong luồng phụ để không chặn vòng lặp async bằng run_in_threadpool(...).
            return await run_in_threadpool(service.recognize, rgb)
        # Khối finally luôn chạy sau try, kể cả khi có lỗi hoặc return; dùng để dọn tài nguyên.
        finally:
            # finally bảo đảm đóng upload cả khi giải mã hoặc xử lý model thất bại.
            # Chờ đóng file upload trong finally; tài nguyên được giải phóng cả khi nhận diện hoặc giải mã bị lỗi.
            await image.close()

    # Chuyển trang gốc đến giao diện nếu có, hoặc tài liệu API nếu chạy chế độ API.
    # Decorator @app.get("/", include_in_schema=False): gắn hàm ngay bên dưới với đường dẫn HTTP hoặc bộ xử lý lỗi này.
    @app.get("/", include_in_schema=False)
    # Định nghĩa hàm index().
    def index():
        # Kết thúc hàm và trả tạo phản hồi chuyển trang bằng RedirectResponse(...); dữ liệu/tham số: '/ui/' if with_ui else '/docs'.
        return RedirectResponse("/ui/" if with_ui else "/docs")

    # Chỉ chạy khối bên dưới khi with_ui cho kết quả đúng.
    if with_ui:
        # Nạp gradio dưới tên ngắn gr để tạo giao diện web và nối sự kiện với hàm Python.
        import gradio as gr
        # Lấy build_ui, UI_CSS, UI_THEME từ .ui; dấu . chỉ module cùng package.
        from .ui import build_ui, UI_CSS, UI_THEME
        # Gắn giao diện vào /ui; giới hạn file cũng được áp dụng ở phía Gradio.
        # Gán cho app: gắn giao diện Gradio vào một đường dẫn của FastAPI bằng gr.mount_gradio_app(...); dữ liệu/tham số: app, build_ui(service) (dựng giao diện Gradio bằng build_ui(...)), path='/ui', css=UI_CSS, theme=UI_THEME, max_file_size=service.settings.max_file_bytes.
        app = gr.mount_gradio_app(app, build_ui(service), path="/ui",
                                 # Truyền css = UI_CSS.
                                 css=UI_CSS, theme=UI_THEME,
                                 # Truyền max_file_size = service.settings.max_file_bytes.
                                 max_file_size=service.settings.max_file_bytes,
                                 # Truyền show_error = False.
                                 show_error=False, run_history=False,
                                 # Truyền footer_links = [].
                                 footer_links=[], enable_monitoring=False)
    # Kết thúc hàm và trả app (ứng dụng web).
    return app


# Hàm tiện ích tạo ứng dụng có giao diện để máy chủ có thể import và chạy.
# Định nghĩa hàm create_demo_app().
def create_demo_app():
    # Kết thúc hàm và trả tạo API và tùy chọn gắn giao diện bằng create_app(...); dữ liệu/tham số: with_ui=True.
    return create_app(with_ui=True)
