# Luồng đăng ký: tạo phiên → chụp 4 góc đạt chất lượng → xem lại → xác nhận lưu.
# Lấy dataclass, field, asdict từ dataclasses để tạo lớp dữ liệu và sao chép/chuyển đổi dữ liệu của lớp.
from dataclasses import dataclass, field, asdict
# Nạp hashlib để tính mã băm để kiểm tra nội dung file/ảnh/vector.
import hashlib
# Lấy Path từ pathlib để biểu diễn đường dẫn và thao tác file bằng đối tượng Path.
from pathlib import Path
# Nạp secrets để sinh mã phiên ngẫu nhiên khó đoán.
import secrets
# Nạp tempfile để tạo file/thư mục tạm có kiểm soát vòng đời.
import tempfile
# Lấy monotonic từ time để đo thời gian xử lý hoặc mô phỏng độ trễ.
from time import monotonic

# Nạp cv2 để xử lý ảnh, chạy ONNX và tính toán hình học camera bằng OpenCV.
import cv2
# Nạp numpy dưới tên ngắn np để tính toán mảng ảnh, vector và ma trận.
import numpy as np

# Lấy initialize_enrollment_registry, enroll_person_from_images từ FaceNet.
from FaceNet import initialize_enrollment_registry, enroll_person_from_images
# Lấy validate_embedding_consistency từ integration.webcam_identity.
from integration.webcam_identity import validate_embedding_consistency
# Lấy validate_detection_result từ MTCNN.mtcnn.
from MTCNN.mtcnn import validate_detection_result
# Lấy ApplicationError từ .errors; dấu . chỉ module cùng package.
from .errors import ApplicationError
# Lấy estimate_head_pose, classify_pose từ .head_pose; dấu . chỉ module cùng package.
from .head_pose import estimate_head_pose, classify_pose
# Lấy validate_rgb từ .images; dấu . chỉ module cùng package.
from .images import validate_rgb
# Lấy commit_identity từ .webcam_registry; dấu . chỉ module cùng package.
from .webcam_registry import commit_identity


# Thứ tự bắt buộc của 4 ảnh; down có trong bộ phân loại nhưng không thuộc luồng đăng ký.
# Gán cho POSES: bộ giá trị ('front', 'left', 'right', 'up') (tuple giữ thứ tự, không gán lại từng phần tử).
POSES = ("front", "left", "right", "up")
# Gán cho POSE_COUNT: đếm số phần tử bằng len(...); dữ liệu/tham số: POSES.
POSE_COUNT = len(POSES)
# Gán cho LABELS: tạo từ điển khóa–giá trị bằng dict(...); dữ liệu/tham số: zip(POSES, ('Nhìn thẳng', 'Quay sang trái của bạn', 'Quay sang phải của bạn', 'Ngẩng đầu')) (ghép các phần tử cùng vị trí bằng zip(...)).
LABELS = dict(zip(POSES, ("Nhìn thẳng", "Quay sang trái của bạn", "Quay sang phải của bạn", "Ngẩng đầu")))


# Các ngưỡng chụp: số frame ổn định, thời gian giữ, chất lượng ảnh và tuổi thọ phiên.
# Decorator dataclass: tự sinh hàm khởi tạo từ các trường; frozen=True chặn gán lại trường nếu được bật.
@dataclass(frozen=True)
# Định nghĩa lớp EnrollmentPolicy; lớp gom dữ liệu và các hàm liên quan.
class EnrollmentPolicy:
    # Cần đủ số frame mới liên tiếp đạt tư thế; vẫn phải thỏa điều kiện thời gian hold_seconds. Gợi ý kiểu: int.
    stable_frames: int = 4
    # Thời gian giữ đúng tư thế tối thiểu tính bằng giây trước khi tự chụp. Gợi ý kiểu: float.
    hold_seconds: float = 0.8
    # Khoảng cách thời gian tối đa giữa hai frame; quá số giây này thì phải giữ ổn định lại. Gợi ý kiểu: float.
    max_gap: float = 2.0
    # Chênh lệch lớn nhất cho phép giữa các góc trong chuỗi ổn định, tính bằng độ. Gợi ý kiểu: float.
    max_angle_spread: float = 6.0
    # Cạnh ngắn của vùng mặt phải đạt tối thiểu số pixel này. Gợi ý kiểu: int.
    min_face_pixels: int = 80
    # Độ sáng trung bình vùng mặt không được thấp hơn mức này trên thang 0–255. Gợi ý kiểu: float.
    min_brightness: float = 45
    # Độ sáng trung bình vùng mặt không được vượt mức này để tránh ảnh quá sáng. Gợi ý kiểu: float.
    max_brightness: float = 220
    # Phương sai Laplacian tối thiểu; thấp hơn coi là ảnh mặt mờ. Gợi ý kiểu: float.
    min_sharpness: float = 15
    # Độ tương đồng tối thiểu giữa các vector trong bộ ảnh để kiểm tra cùng danh tính; khác ngưỡng nhận diện hồ sơ. Gợi ý kiểu: float.
    min_similarity: float = 0.65
    # Tuổi thọ phiên tính từ lúc tạo, theo giây; 1200 giây tương đương 20 phút. Gợi ý kiểu: float.
    ttl_seconds: float = 1200
    # Giới hạn số phiên đăng ký cùng được giữ trong RAM. Gợi ý kiểu: int.
    max_sessions: int = 8


# Trạng thái tạm trong RAM của một người; chưa ghi vào kho chính khi đang chụp.
# Decorator dataclass: tự sinh hàm khởi tạo từ các trường; frozen=True chặn gán lại trường nếu được bật.
@dataclass
# Định nghĩa lớp EnrollmentSession; lớp gom dữ liệu và các hàm liên quan.
class EnrollmentSession:
    # Khai báo person_id có kiểu gợi ý str; giá trị được cung cấp khi tạo đối tượng.
    person_id: str
    # Khai báo display_name có kiểu gợi ý str; giá trị được cung cấp khi tạo đối tượng.
    display_name: str
    # Khai báo created có kiểu gợi ý float; giá trị được cung cấp khi tạo đối tượng.
    created: float
    # Khai báo version có kiểu gợi ý str; giá trị được cung cấp khi tạo đối tượng.
    version: str
    # captures giữ ảnh theo góc; stable giữ các góc đo được của chuỗi frame đang xét.
    # Khai báo captures có kiểu gợi ý dict; gán gọi field(...); dữ liệu/tham số: default_factory=dict.
    captures: dict = field(default_factory=dict)
    # Khai báo stable có kiểu gợi ý list; gán gọi field(...); dữ liệu/tham số: default_factory=list.
    stable: list = field(default_factory=list)
    # Mốc bắt đầu chuỗi giữ tư thế hiện tại; None khi chưa có chuỗi đạt yêu cầu. Gợi ý kiểu: float | None.
    stable_since: float | None = None
    # Mốc nhận frame mới nhất, dùng để phát hiện camera ngắt quãng. Gợi ý kiểu: float | None.
    last_frame: float | None = None
    # Mã băm frame trước, dùng phát hiện ảnh bị gửi lặp nguyên vẹn. Gợi ý kiểu: str | None.
    last_digest: str | None = None
    # revision tăng khi bước chụp thay đổi; reviewed_revision ghi bộ ảnh đã xem để chặn lưu nhầm ảnh cũ.
    # Khai báo revision có kiểu gợi ý int; gán giá trị 0.
    revision: int = 0
    # Phiên bản bộ ảnh đã được xem lại; None nghĩa là chưa xem bộ ảnh hiện tại. Gợi ý kiểu: int | None.
    reviewed_revision: int | None = None
    # Cờ tạm dừng nhận ảnh; False cho phép tiếp tục xử lý frame. Gợi ý kiểu: bool.
    paused: bool = False
    # Khai báo message có kiểu gợi ý str; gán chuỗi 'Bật camera và giữ tư thế nhìn thẳng.'.
    message: str = "Bật camera và giữ tư thế nhìn thẳng."


# Bọc mạng để ghi lại đúng các vector đầu ra đã tạo, phục vụ kiểm tra cùng danh tính.
# Định nghĩa lớp RecordingNet; lớp gom dữ liệu và các hàm liên quan.
class RecordingNet:
    # Chuỗi tài liệu (docstring) mô tả cách dùng; Python lưu nội dung này trong __doc__.
    """Observe the exact pipeline forwards; leave all core math untouched."""
    # Khởi tạo đối tượng và giữ các phụ thuộc/trạng thái cần dùng trong những lần gọi sau.
    # Định nghĩa hàm __init__(self, net). self là đối tượng đang gọi phương thức.
    def __init__(self, net):
        # Tách kết quả theo thứ tự và gán cho (self.net, self.outputs): bộ giá trị (net, []) (tuple giữ thứ tự, không gán lại từng phần tử).
        self.net, self.outputs = net, []

    # Chuyển dữ liệu đầu vào cho mạng; giữ nguyên giao diện gọi của OpenCV DNN.
    # Định nghĩa hàm setInput(self). self là đối tượng đang gọi phương thức. *args gom các đối số vị trí còn lại thành tuple.
    def setInput(self, *args):
        # Kết thúc hàm và trả đưa dữ liệu đầu vào vào mạng qua self.net.setInput(...); dữ liệu/tham số: *args.
        return self.net.setInput(*args)

    # Chạy mạng và thu đầu ra; vector đặc trưng là dãy số mô tả khuôn mặt.
    # Định nghĩa hàm forward(self). self là đối tượng đang gọi phương thức. *args gom các đối số vị trí còn lại thành tuple.
    def forward(self, *args):
        # Gán cho result: chạy mạng để lấy vector đầu ra qua self.net.forward(...); dữ liệu/tham số: *args.
        result = self.net.forward(*args)
        # Thêm một phần tử vào cuối danh sách qua self.outputs.append(...); dữ liệu/tham số: result.copy() if isinstance(result, np.ndarray) else result.
        self.outputs.append(result.copy() if isinstance(result, np.ndarray) else result)
        # Kết thúc hàm và trả result (kết quả xử lý).
        return result


# Quản lý nhiều phiên độc lập, kiểm tra từng frame và chỉ lưu khi đủ điều kiện.
# Định nghĩa lớp WebcamEnrollment; lớp gom dữ liệu và các hàm liên quan.
class WebcamEnrollment:
    # Khởi tạo đối tượng và giữ các phụ thuộc/trạng thái cần dùng trong những lần gọi sau.
    # Định nghĩa hàm __init__(self, service, policy, clock). self là đối tượng đang gọi phương thức. Mặc định: policy=None, clock=monotonic.
    def __init__(self, service, policy=None, clock=monotonic):
        # Gán cho self.service: service (dịch vụ điều phối nhận diện).
        self.service = service
        # Gán cho self.policy: kết hợp bằng or: lấy giá trị đúng đầu tiên nếu có trong policy or EnrollmentPolicy().
        self.policy = policy or EnrollmentPolicy()
        # Gán cho self.clock: clock.
        self.clock = clock
        # Tạo dict quản lý phiên trong RAM: mỗi khóa là token, mỗi giá trị là một EnrollmentSession.
        self.sessions = {}

    # Dọn phiên quá hạn để giải phóng ảnh trong bộ nhớ và chỗ cho phiên mới.
    # Định nghĩa hàm _expire(self). self là đối tượng đang gọi phương thức.
    def _expire(self):
        # Gán cho now: gọi self.clock(...).
        now = self.clock()
        # Lặp qua list(self.sessions); mỗi lượt gán phần tử cho token.
        for token in list(self.sessions):
            # Chỉ chạy khối bên dưới khi now - self.sessions[token].created > self.policy.ttl_seconds cho kết quả đúng.
            if now - self.sessions[token].created > self.policy.ttl_seconds:
                # Xóa self.sessions[token] khỏi vùng dữ liệu hiện tại.
                del self.sessions[token]

    # Dọn phiên hết hạn trước khi tra cứu; token không hợp lệ sẽ bị từ chối.
    # Định nghĩa hàm _get(self, token). self là đối tượng đang gọi phương thức.
    def _get(self, token):
        # Dọn phiên hết hạn qua self._expire(...).
        self._expire()
        # Chỉ chạy khối bên dưới khi not isinstance(token, str) or token not in self.sessions cho kết quả đúng.
        if not isinstance(token, str) or token not in self.sessions:
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'session_missing', 'Phiên không tồn tại hoặc đã hết hạn; hãy bắt đầu lại.', 404.
            raise ApplicationError("session_missing", "Phiên không tồn tại hoặc đã hết hạn; hãy bắt đầu lại.", 404)
        # Kết thúc hàm và trả lấy phần tử/lát cắt self.sessions[token] từ self.sessions.
        return self.sessions[token]

    # Xóa chuỗi tư thế ổn định; người dùng phải giữ yên lại từ đầu.
    # Decorator staticmethod: hàm trong lớp không tự nhận tham số self.
    @staticmethod
    # Định nghĩa hàm _reset(session).
    def _reset(session):
        # Xóa các phần tử hoặc đăng ký sự kiện xóa tùy đối tượng qua session.stable.clear(...).
        session.stable.clear()
        # Gán cho session.stable_since: None (chưa có/không có giá trị).
        session.stable_since = None

    # Lấy góc còn thiếu đầu tiên theo POSES; None nghĩa là đã chụp đủ.
    # Decorator staticmethod: hàm trong lớp không tự nhận tham số self.
    @staticmethod
    # Định nghĩa hàm _target(session).
    def _target(session):
        # Kết thúc hàm và trả lấy phần tử tiếp theo, hoặc giá trị mặc định nếu đã hết bằng next(...); dữ liệu/tham số: (pose for pose in POSES if pose not in session.captures), None.
        return next((pose for pose in POSES if pose not in session.captures), None)

    # Kiểm tra ID/tên, giới hạn phiên và tạo token mới; lúc này chỉ lưu trạng thái trong RAM.
    # Định nghĩa hàm start(self, person_id, display_name). self là đối tượng đang gọi phương thức.
    def start(self, person_id, display_name):
        # Kiểm tra ID hợp lệ qua self.service.check_id(...); dữ liệu/tham số: person_id.
        self.service.check_id(person_id)
        # Chỉ chạy khối bên dưới khi not isinstance(display_name, str) or not display_name.strip() or len(display_name.strip()) > 150 or any((ord(c) < 32 for c i… cho kết quả đúng.
        if (not isinstance(display_name, str) or not display_name.strip()
                # Bổ sung trường hợp được chấp nhận bằng or: or len(display_name.strip()) > 150 or any(ord(c) < 32 for c in display_name)).
                or len(display_name.strip()) > 150 or any(ord(c) < 32 for c in display_name)):
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'invalid_name', 'Nhập tên từ 1–150 ký tự, không có ký tự điều khiển.'.
            raise ApplicationError("invalid_name", "Nhập tên từ 1–150 ký tự, không có ký tự điều khiển.")
        # Mở khối with để lấy khóa truy cập model và kho hồ sơ qua self.service.guard(...); bao thao tác để chuyển lỗi bất ngờ thành ApplicationError qua self.service._operation(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self.service.guard(), self.service._operation():
            # Bảo đảm model đã nạp trước khi xử lý qua self.service._require_model(...).
            self.service._require_model()
            # Gán cho people: đọc và xác minh kho hồ sơ qua self.service._registry(...).
            people = self.service._registry()
            # Chỉ chạy khối bên dưới khi any((p['person_id'] == person_id for p in people)) cho kết quả đúng.
            if any(p["person_id"] == person_id for p in people):
                # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'duplicate_person', 'person_id đã tồn tại; không ghi đè.', 409.
                raise ApplicationError("duplicate_person", "person_id đã tồn tại; không ghi đè.", 409)
            # Dọn phiên hết hạn qua self._expire(...).
            self._expire()
            # Chỉ chạy khối bên dưới khi len(self.sessions) >= self.policy.max_sessions cho kết quả đúng.
            if len(self.sessions) >= self.policy.max_sessions:
                # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'sessions_full', 'Đã đạt giới hạn phiên camera; hủy phiên cũ hoặc chờ hết hạn.', 503.
                raise ApplicationError("sessions_full", "Đã đạt giới hạn phiên camera; hủy phiên cũ hoặc chờ hết hạn.", 503)
            # Token ngẫu nhiên dùng tra cứu phiên; ID người dùng không được dùng trực tiếp làm token.
            # Sinh token từ 32 byte ngẫu nhiên, mã hóa thành ký tự an toàn trong URL để tra cứu phiên.
            token = secrets.token_urlsafe(32)
            # Gán cho self.sessions[token]: tạo trạng thái phiên chụp mới bằng EnrollmentSession(...); dữ liệu/tham số: person_id, display_name.strip() (bỏ khoảng trắng hai đầu chuỗi qua display_name.strip(...)), self.clock() (gọi self.clock(...)).
            self.sessions[token] = EnrollmentSession(person_id, display_name.strip(), self.clock(), self.service.runtime.version)
            # Kết thúc hàm và trả tạo dữ liệu tiến độ hiện tại của phiên qua self._snapshot(...); dữ liệu/tham số: token, self.sessions[token].
            return self._snapshot(token, self.sessions[token])

    # Tạo bản tóm tắt tiến độ phiên cho API/giao diện, không trả ảnh gốc tại đây.
    # Định nghĩa hàm _snapshot(self, token, session). self là đối tượng đang gọi phương thức.
    def _snapshot(self, token, session):
        # Gán cho target: tìm góc còn thiếu kế tiếp qua self._target(...); dữ liệu/tham số: session.
        target = self._target(session)
        # Kết thúc hàm và trả từ điển {'session_id': token, 'person_id': session.person_id, 'display_name': session.displa… (ánh xạ khóa → giá trị).
        return {"session_id": token, "person_id": session.person_id, "display_name": session.display_name,
                # Tiếp phần biểu thức nhiều dòng: đếm số phần tử bằng len(...).
                "revision": session.revision, "progress": len(session.captures), "total": POSE_COUNT,
                # Tiếp phần biểu thức nhiều dòng: tạo danh sách bằng cách duyệt p trong POSES và chỉ giữ phần tử đạt điều kiện if; mỗi phần tử là p.
                "captured_poses": [p for p in POSES if p in session.captures],
                # Tiếp phần biểu thức nhiều dòng: đọc giá trị theo khóa, dùng mặc định nếu khóa vắng mặt qua LABELS.get(...).
                "next_pose": target, "instruction": LABELS.get(target, f"Xem lại {POSE_COUNT} ảnh rồi xác nhận lưu."),
                # Tiếp phần biểu thức nhiều dòng: chọn 'review' nếu target is None; ngược lại chọn 'paused' if session.paused else 'capturing'.
                "status": "review" if target is None else ("paused" if session.paused else "capturing"),
                # Tiếp phần biểu thức nhiều dòng: self.policy.stable_frames (đọc thuộc tính).
                "stable_frames": len(session.stable), "required_stable_frames": self.policy.stable_frames,
                # Tiếp phần biểu thức nhiều dòng: lấy giá trị lớn nhất bằng max(...).
                "message": session.message, "expires_in_seconds": max(0, self.policy.ttl_seconds - (self.clock() - session.created))}

    # Lấy trạng thái hiện tại; camera ngắt quá lâu sẽ làm mất chuỗi ổn định.
    # Định nghĩa hàm status(self, token). self là đối tượng đang gọi phương thức.
    def status(self, token):
        # Mở khối with để self.service._mutex (lấy khóa trước khi vào khối, tự nhả khóa khi ra khỏi khối); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self.service._mutex:
            # Gán cho session: tra cứu phiên sau khi kiểm tra hạn qua self._get(...); dữ liệu/tham số: token.
            session = self._get(token)
            # Chỉ chạy khối bên dưới khi self._target(session) and session.last_frame is not None and (self.clock() - session.last_frame > self.policy.max_gap) cho kết quả đúng.
            if (self._target(session) and session.last_frame is not None
                    # Bổ sung điều kiện bắt buộc bằng and: and self.clock() - session.last_frame > self.policy.max_gap).
                    and self.clock() - session.last_frame > self.policy.max_gap):
                # Xóa chuỗi frame ổn định qua self._reset(...); dữ liệu/tham số: session.
                self._reset(session)
                # Gán cho session.message: chuỗi 'Không nhận được frame camera. Kiểm tra kết nối và bật lại ghi h….
                session.message = "Không nhận được frame camera. Kiểm tra kết nối và bật lại ghi hình."
            # Kết thúc hàm và trả tạo dữ liệu tiến độ hiện tại của phiên qua self._snapshot(...); dữ liệu/tham số: token, session.
            return self._snapshot(token, session)

    # Lấy danh sách ảnh đã chụp qua cơ chế xem lại có theo dõi phiên bản.
    # Định nghĩa hàm review(self, token). self là đối tượng đang gọi phương thức.
    def review(self, token):
        # Kết thúc hàm và trả lấy phần tử/lát cắt self.review_snapshot(token)['images'] từ self.review_snapshot(token).
        return self.review_snapshot(token)["images"]

    # Trả bản sao ảnh và đánh dấu bộ ảnh đầy đủ hiện tại đã được xem lại.
    # Định nghĩa hàm review_snapshot(self, token). self là đối tượng đang gọi phương thức.
    def review_snapshot(self, token):
        # Mở khối with để self.service._mutex (lấy khóa trước khi vào khối, tự nhả khóa khi ra khỏi khối); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self.service._mutex:
            # Gán cho session: tra cứu phiên sau khi kiểm tra hạn qua self._get(...); dữ liệu/tham số: token.
            session = self._get(token)
            # Chỉ chạy khối bên dưới khi set(session.captures) == set(POSES) cho kết quả đúng.
            if set(session.captures) == set(POSES):
                # Đánh dấu người dùng vừa xem bộ ảnh ở đúng phiên bản hiện tại.
                session.reviewed_revision = session.revision
            # Kết thúc hàm và trả từ điển {'revision': session.revision, 'images': [(pose, session.captures[pose].copy()) for … (ánh xạ khóa → giá trị).
            return {"revision": session.revision,
                    # Tiếp phần biểu thức nhiều dòng: tạo danh sách bằng cách duyệt pose trong POSES và chỉ giữ phần tử đạt điều kiện if; mỗi phần tử là (pose, session.captures[pose].copy()).
                    "images": [(pose, session.captures[pose].copy()) for pose in POSES if pose in session.captures]}

    # Bỏ phiên khỏi RAM; gọi lại với phiên đã kết thúc cũng trả trạng thái rõ ràng.
    # Định nghĩa hàm cancel(self, token). self là đối tượng đang gọi phương thức.
    def cancel(self, token):
        # Mở khối with để self.service._mutex (lấy khóa trước khi vào khối, tự nhả khóa khi ra khỏi khối); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self.service._mutex:
            # Gán cho removed: lấy và xóa phần tử theo khóa qua self.sessions.pop(...); dữ liệu/tham số: token, None.
            removed = self.sessions.pop(token, None)
            # Kết thúc hàm và trả từ điển {'status': 'cancelled' if removed is not None else 'session_ended', 'progress': 0, '… (ánh xạ khóa → giá trị).
            return {"status": "cancelled" if removed is not None else "session_ended", "progress": 0, "total": POSE_COUNT}

    # Tạm ngừng nhận ảnh, tăng phiên bản và yêu cầu giữ ổn định lại khi tiếp tục.
    # Định nghĩa hàm pause(self, token). self là đối tượng đang gọi phương thức.
    def pause(self, token):
        # Mở khối with để self.service._mutex (lấy khóa trước khi vào khối, tự nhả khóa khi ra khỏi khối); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self.service._mutex:
            # Gán cho session: tra cứu phiên sau khi kiểm tra hạn qua self._get(...); dữ liệu/tham số: token.
            session = self._get(token)
            # Gán cho session.paused: True (đúng/bật).
            session.paused = True
            # Tăng phiên bản bước chụp thêm 1; frame hoặc xác nhận gửi theo phiên bản cũ sẽ bị chặn.
            session.revision += 1
            # Xóa chuỗi frame ổn định qua self._reset(...); dữ liệu/tham số: session.
            self._reset(session)
            # Gán cho session.message: chuỗi 'Camera đã dừng/ngắt. Bật lại camera rồi bấm Tiếp tục.'.
            session.message = "Camera đã dừng/ngắt. Bật lại camera rồi bấm Tiếp tục."
            # Kết thúc hàm và trả tạo dữ liệu tiến độ hiện tại của phiên qua self._snapshot(...); dữ liệu/tham số: token, session.
            return self._snapshot(token, session)

    # Mở lại phiên chụp và xóa dấu vết frame cũ để bắt đầu chuỗi mới.
    # Định nghĩa hàm resume(self, token). self là đối tượng đang gọi phương thức.
    def resume(self, token):
        # Mở khối with để self.service._mutex (lấy khóa trước khi vào khối, tự nhả khóa khi ra khỏi khối); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self.service._mutex:
            # Gán cho session: tra cứu phiên sau khi kiểm tra hạn qua self._get(...); dữ liệu/tham số: token.
            session = self._get(token)
            # Gán cho session.paused: False (sai/tắt).
            session.paused = False
            # Tăng phiên bản bước chụp thêm 1; frame hoặc xác nhận gửi theo phiên bản cũ sẽ bị chặn.
            session.revision += 1
            # Xóa chuỗi frame ổn định qua self._reset(...); dữ liệu/tham số: session.
            self._reset(session)
            # Gán cho session.last_frame: None (chưa có/không có giá trị).
            session.last_frame = None
            # Gán cho session.last_digest: None (chưa có/không có giá trị).
            session.last_digest = None
            # Gán cho session.message: chuỗi 'Đã tiếp tục; giữ tư thế theo hướng dẫn.'.
            session.message = "Đã tiếp tục; giữ tư thế theo hướng dẫn."
            # Kết thúc hàm và trả tạo dữ liệu tiến độ hiện tại của phiên qua self._snapshot(...); dữ liệu/tham số: token, session.
            return self._snapshot(token, session)

    # Bỏ riêng ảnh góc được chọn; các góc khác còn nguyên, lần xem lại cũ mất hiệu lực.
    # Định nghĩa hàm retake(self, token, pose). self là đối tượng đang gọi phương thức.
    def retake(self, token, pose):
        # Mở khối with để self.service._mutex (lấy khóa trước khi vào khối, tự nhả khóa khi ra khỏi khối); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self.service._mutex:
            # Gán cho session: tra cứu phiên sau khi kiểm tra hạn qua self._get(...); dữ liệu/tham số: token.
            session = self._get(token)
            # Chỉ chạy khối bên dưới khi pose not in POSES or pose not in session.captures cho kết quả đúng.
            if pose not in POSES or pose not in session.captures:
                # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'invalid_retake', 'Chọn tư thế đã chụp để chụp lại.'.
                raise ApplicationError("invalid_retake", "Chọn tư thế đã chụp để chụp lại.")
            # Xóa session.captures[pose] khỏi vùng dữ liệu hiện tại.
            del session.captures[pose]
            # Tăng phiên bản bước chụp thêm 1; frame hoặc xác nhận gửi theo phiên bản cũ sẽ bị chặn.
            session.revision += 1
            # Hủy dấu đã xem vì bộ ảnh vừa thay đổi; người dùng phải xem lại bộ mới.
            session.reviewed_revision = None
            # Xóa chuỗi frame ổn định qua self._reset(...); dữ liệu/tham số: session.
            self._reset(session)
            # Gán cho session.message: chuỗi f-string: chèn giá trị biểu thức nằm trong {...} vào nội dung.
            session.message = f"Chụp lại: {LABELS[pose]}. Các góc khác được giữ trong phiên."
            # Kết thúc hàm và trả tạo dữ liệu tiến độ hiện tại của phiên qua self._snapshot(...); dữ liệu/tham số: token, session.
            return self._snapshot(token, session)

    # Kiểm tra RGB và thu nhỏ giữ tỷ lệ; tạo bản sao liên tục trong bộ nhớ để xử lý.
    # Định nghĩa hàm _prepare(self, frame). self là đối tượng đang gọi phương thức.
    def _prepare(self, frame):
        # Kiểm tra kiểu mảng, số kênh và kích thước RGB bằng validate_rgb(...); dữ liệu/tham số: frame, self.service.settings.
        validate_rgb(frame, self.service.settings)
        # Tách kết quả theo thứ tự và gán cho (h, w): lấy phần tử/lát cắt frame.shape[:2] từ frame.shape.
        h, w = frame.shape[:2]
        # Gán cho bound: self.service.settings.webcam_width (đọc thuộc tính).
        bound = self.service.settings.webcam_width
        # Chỉ chạy khối bên dưới khi max(h, w) > bound cho kết quả đúng.
        if max(h, w) > bound:
            # Gán cho scale: chia trong bound / max(h, w).
            scale = bound / max(h, w)
            # Gán cho frame: đổi kích thước ảnh theo chiều rộng/cao được chỉ định bằng cv2.resize(...); dữ liệu/tham số: frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA.
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        # Kết thúc hàm và trả tạo bản sao để thay đổi không tác động dữ liệu gốc qua np.ascontiguousarray(frame).copy(...).
        return np.ascontiguousarray(frame).copy()

    # Yêu cầu đúng một mặt đủ lớn, rõ và đủ sáng; sau đó đo hướng đầu.
    # Định nghĩa hàm _inspect(self, frame). self là đối tượng đang gọi phương thức.
    def _inspect(self, frame):
        # Gán cho result: phát hiện các mặt cùng khung và điểm mốc qua self.service.runtime.detector.detect(...); dữ liệu/tham số: frame.
        result = self.service.runtime.detector.detect(frame)
        # Kiểm tra cấu trúc đầu ra của detector bằng validate_detection_result(...); dữ liệu/tham số: result.
        validate_detection_result(result)
        # Gán cho count: đếm số phần tử bằng len(...); dữ liệu/tham số: result.boxes.
        count = len(result.boxes)
        # Chỉ chạy khối bên dưới khi count != 1 cho kết quả đúng.
        if count != 1:
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'enrollment_face_count', f'Cần đúng 1 khuôn mặt; phát hiện {count}.'.
            raise ApplicationError("enrollment_face_count", f"Cần đúng 1 khuôn mặt; phát hiện {count}.")
        # Tách kết quả theo thứ tự và gán cho (h, w): lấy phần tử/lát cắt frame.shape[:2] từ frame.shape.
        h, w = frame.shape[:2]
        # Tách kết quả theo thứ tự và gán cho (x1, y1, x2, y2): lấy phần tử/lát cắt result.boxes[0] từ result.boxes.
        x1, y1, x2, y2 = result.boxes[0]
        # Chỉ chạy khối bên dưới khi x1 < 0 or y1 < 0 or x2 > w or (y2 > h) cho kết quả đúng.
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'face_clipped', 'Đưa toàn bộ khuôn mặt vào khung hình.'.
            raise ApplicationError("face_clipped", "Đưa toàn bộ khuôn mặt vào khung hình.")
        # Chỉ chạy khối bên dưới khi min(x2 - x1, y2 - y1) < self.policy.min_face_pixels cho kết quả đúng.
        if min(x2 - x1, y2 - y1) < self.policy.min_face_pixels:
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'face_too_small', 'Đưa khuôn mặt gần camera hơn.'.
            raise ApplicationError("face_too_small", "Đưa khuôn mặt gần camera hơn.")
        # Gán cho crop: lấy phần tử/lát cắt frame[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] từ frame.
        crop = frame[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))]
        # Đổi riêng vùng mặt từ 3 kênh RGB về một kênh độ sáng; dùng đo sáng và nét.
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        # Đưa vùng mặt về cùng kích thước để ngưỡng độ nét ít phụ thuộc khoảng cách camera.
        # Đưa ảnh xám vùng mặt về 160×160 pixel; INTER_AREA phù hợp thu nhỏ để đo độ nét trên cùng một thang kích thước.
        gray = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
        # Lấy trung bình mọi pixel ảnh xám: gần 0 là tối, gần 255 là rất sáng.
        brightness = float(gray.mean())
        # Phương sai Laplacian đo mức biến thiên cạnh: ảnh nhòe thường có giá trị nhỏ.
        # Laplacian tìm biến thiên cạnh bằng số thực 64 bit; phương sai cao thường tương ứng ảnh có cạnh rõ hơn.
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Chỉ chạy khối bên dưới khi not self.policy.min_brightness <= brightness <= self.policy.max_brightness cho kết quả đúng.
        if not self.policy.min_brightness <= brightness <= self.policy.max_brightness:
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'poor_lighting', 'Ánh sáng chưa đạt; chiếu sáng đều vào mặt, tránh ngược sáng.'.
            raise ApplicationError("poor_lighting", "Ánh sáng chưa đạt; chiếu sáng đều vào mặt, tránh ngược sáng.")
        # Chỉ chạy khối bên dưới khi sharpness < self.policy.min_sharpness cho kết quả đúng.
        if sharpness < self.policy.min_sharpness:
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'blurry_image', 'Ảnh mờ; giữ yên và kiểm tra lấy nét camera.'.
            raise ApplicationError("blurry_image", "Ảnh mờ; giữ yên và kiểm tra lấy nét camera.")
        # Gán cho pose: ước lượng ba góc đầu từ điểm mốc và kích thước ảnh bằng estimate_head_pose(...); dữ liệu/tham số: result.landmarks[0], frame.shape.
        pose = estimate_head_pose(result.landmarks[0], frame.shape)
        # Chỉ chạy khối bên dưới khi pose is None cho kết quả đúng.
        if pose is None:
            # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'pose_unavailable', 'Landmarks chưa đủ tin cậy để đo hướng đầu; nhìn rõ vào camera.'.
            raise ApplicationError("pose_unavailable", "Landmarks chưa đủ tin cậy để đo hướng đầu; nhìn rõ vào camera.")
        # Kết thúc hàm và trả bộ giá trị (pose, {'brightness': brightness, 'sharpness': sharpness}) (tuple giữ thứ tự, không gán lại từng phần tử).
        return pose, {"brightness": brightness, "sharpness": sharpness}

    # Kiểm tra frame thuộc đúng bước, đúng góc và ổn định đủ lâu rồi mới tự chụp.
    # Định nghĩa hàm frame(self, token, frame, revision). self là đối tượng đang gọi phương thức.
    def frame(self, token, frame, revision):
        # Mở khối with để lấy khóa truy cập model và kho hồ sơ qua self.service.guard(...); bao thao tác để chuyển lỗi bất ngờ thành ApplicationError qua self.service._operation(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self.service.guard(), self.service._operation():
            # Gán cho session: tra cứu phiên sau khi kiểm tra hạn qua self._get(...); dữ liệu/tham số: token.
            session = self._get(token)
            # Từ chối frame còn đang truyền của bước cũ sau khi chụp/chụp lại/tạm dừng.
            # Chỉ chạy khối bên dưới khi revision != session.revision cho kết quả đúng.
            if revision != session.revision:
                # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'stale_frame', 'Frame thuộc bước cũ; gửi revision hiện tại.', 409.
                raise ApplicationError("stale_frame", "Frame thuộc bước cũ; gửi revision hiện tại.", 409)
            # Chỉ chạy khối bên dưới khi session.paused or self._target(session) is None cho kết quả đúng.
            if session.paused or self._target(session) is None:
                # Kết thúc hàm và trả tạo dữ liệu tiến độ hiện tại của phiên qua self._snapshot(...); dữ liệu/tham số: token, session.
                return self._snapshot(token, session)
            # Bảo đảm model đã nạp trước khi xử lý qua self.service._require_model(...).
            self.service._require_model()
            # Gán cho now: gọi self.clock(...).
            now = self.clock()
            # Chỉ chạy khối bên dưới khi session.last_frame is not None and now - session.last_frame > self.policy.max_gap cho kết quả đúng.
            if session.last_frame is not None and now - session.last_frame > self.policy.max_gap:
                # Xóa chuỗi frame ổn định qua self._reset(...); dữ liệu/tham số: session.
                self._reset(session)
            # Ghi mốc đã nhận frame gần nhất để phát hiện camera ngắt quãng ở lần kiểm tra sau.
            session.last_frame = now
            # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
            try:
                # Chỉ chạy khối bên dưới khi frame is None cho kết quả đúng.
                if frame is None:
                    # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'camera_unavailable', 'Không nhận được ảnh camera; kiểm tra quyền và kết nối.'.
                    raise ApplicationError("camera_unavailable", "Không nhận được ảnh camera; kiểm tra quyền và kết nối.")
                # Gán cho image: kiểm tra và thu nhỏ frame qua self._prepare(...); dữ liệu/tham số: frame.
                image = self._prepare(frame)
                # So mã băm toàn ảnh với frame trước để không dùng lặp một ảnh tĩnh làm chuỗi mới.
                # Chuyển toàn bộ pixel ảnh thành byte và tính mã SHA-256 dạng chữ; hai frame giống hệt sẽ có cùng mã.
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                # Chỉ chạy khối bên dưới khi digest == session.last_digest cho kết quả đúng.
                if digest == session.last_digest:
                    # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'repeated_frame', 'Đang chờ frame camera mới; không dùng lại một ảnh tĩnh.'.
                    raise ApplicationError("repeated_frame", "Đang chờ frame camera mới; không dùng lại một ảnh tĩnh.")
                # Nhớ mã frame này để phát hiện việc gửi lại đúng ảnh đó ở lượt kế tiếp.
                session.last_digest = digest
                # Tách kết quả theo thứ tự và gán cho (pose, quality): đo chất lượng và tư thế mặt qua self._inspect(...); dữ liệu/tham số: image.
                pose, quality = self._inspect(image)
                # Gán cho expected: tìm góc còn thiếu kế tiếp qua self._target(...); dữ liệu/tham số: session.
                expected = self._target(session)
                # Chỉ chạy khối bên dưới khi classify_pose(pose) != expected cho kết quả đúng.
                if classify_pose(pose) != expected:
                    # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'wrong_pose', f'Tư thế chưa đạt. Hãy {LABELS[expected].lower()}.'.
                    raise ApplicationError("wrong_pose", f"Tư thế chưa đạt. Hãy {LABELS[expected].lower()}.")
                # Gom ba góc hiện tại thành vector NumPy để so độ dao động giữa các frame.
                angles = np.array([pose.yaw, pose.pitch, pose.roll])
                # np.ptp tính lớn nhất trừ nhỏ nhất; góc dao động quá nhiều thì phải giữ yên lại.
                # Chỉ chạy khối bên dưới khi session.stable and np.ptp(np.vstack([*session.stable, angles]), axis=0).max() > self.policy.max_angle_spread cho kết quả đúng.
                if session.stable and np.ptp(np.vstack([*session.stable, angles]), axis=0).max() > self.policy.max_angle_spread:
                    # Xóa chuỗi frame ổn định qua self._reset(...); dữ liệu/tham số: session.
                    self._reset(session)
                # Chỉ chạy khối bên dưới khi not session.stable cho kết quả đúng.
                if not session.stable:
                    # Ghi thời điểm bắt đầu chuỗi giữ đúng tư thế; dùng kiểm tra đã giữ đủ lâu chưa.
                    session.stable_since = now
                # Thêm bộ ba góc của frame đạt yêu cầu vào chuỗi đang theo dõi ổn định.
                session.stable.append(angles)
                # Lát cắt âm giữ tối đa stable_frames phần tử gần nhất; bỏ các bộ góc quá cũ.
                session.stable = session.stable[-self.policy.stable_frames:]
                # Gán cho captured: False (sai/tắt).
                captured = False
                # Cần đồng thời đủ số frame và đủ thời gian giữ; chỉ đạt một điều kiện là chưa đủ.
                # Chỉ chạy khối bên dưới khi len(session.stable) >= self.policy.stable_frames and now - session.stable_since >= self.policy.hold_seconds cho kết quả đúng.
                if (len(session.stable) >= self.policy.stable_frames
                        # Bổ sung điều kiện bắt buộc bằng and: and now - session.stable_since >= self.policy.hold_seconds).
                        and now - session.stable_since >= self.policy.hold_seconds):
                    # Chỉ giữ ảnh đã kiểm tra trên máy chủ; khóa ghi để bên gọi không sửa ảnh đã chụp.
                    # Khóa khả năng ghi của mảng ảnh đã chụp để tránh ảnh bị sửa sau khi kiểm tra.
                    image.setflags(write=False)
                    # Lưu ảnh vào từ điển theo góc đang yêu cầu; ví dụ captures["left"] là ảnh quay trái.
                    session.captures[expected] = image
                    # Tăng phiên bản bước chụp thêm 1; frame hoặc xác nhận gửi theo phiên bản cũ sẽ bị chặn.
                    session.revision += 1
                    # Xóa chuỗi frame ổn định qua self._reset(...); dữ liệu/tham số: session.
                    self._reset(session)
                    # Gán cho captured: True (đúng/bật).
                    captured = True
                    # Gán cho session.message: chuỗi f-string: chèn giá trị biểu thức nằm trong {...} vào nội dung.
                    session.message = f"Đã chụp: {LABELS[expected]}."
                # Nhánh else: chạy khi nhánh if trước đó không được chọn (hoặc vòng lặp kết thúc bình thường nếu thuộc vòng lặp).
                else:
                    # Gán cho session.message: chuỗi 'Đúng tư thế; giữ yên cho đến khi tự chụp.'.
                    session.message = "Đúng tư thế; giữ yên cho đến khi tự chụp."
                # Kết thúc hàm và trả từ điển {**self._snapshot(token, session), 'pose': asdict(pose), 'quality': quality, 'captur… (ánh xạ khóa → giá trị).
                return {**self._snapshot(token, session), "pose": asdict(pose), "quality": quality, "captured": captured}
            # Bắt lỗi ApplicationError và đặt tên error để xử lý trong khối bên dưới.
            except ApplicationError as error:
                # Xóa chuỗi frame ổn định qua self._reset(...); dữ liệu/tham số: session.
                self._reset(session)
                # Gán cho session.message: error.message (đọc thuộc tính).
                session.message = error.message
                # Kết thúc hàm và trả từ điển {**self._snapshot(token, session), **error.payload(), 'captured': False} (ánh xạ khóa → giá trị).
                return {**self._snapshot(token, session), **error.payload(), "captured": False}
            # Bắt lỗi Exception để xử lý trong khối bên dưới.
            except Exception:
                # Xóa chuỗi frame ổn định qua self._reset(...); dữ liệu/tham số: session.
                self._reset(session)
                # Gán cho session.message: chuỗi 'Lỗi model khi xử lý camera; chưa lưu danh tính.'.
                session.message = "Lỗi model khi xử lý camera; chưa lưu danh tính."
                # Ném lại lỗi đang xử lý để phía gọi biết thao tác chưa thành công.
                raise

    # Xác minh bộ ảnh đã xem, kiểm tra cùng danh tính trong kho tạm rồi lưu kho chính.
    # Định nghĩa hàm confirm(self, token, confirmed, revision). self là đối tượng đang gọi phương thức. Mặc định: confirmed=False, revision=None.
    def confirm(self, token, confirmed=False, revision=None):
        # Mở khối with để lấy khóa truy cập model và kho hồ sơ qua self.service.guard(...); bao thao tác để chuyển lỗi bất ngờ thành ApplicationError qua self.service._operation(...); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
        with self.service.guard(), self.service._operation():
            # Gán cho session: tra cứu phiên sau khi kiểm tra hạn qua self._get(...); dữ liệu/tham số: token.
            session = self._get(token)
            # Chỉ chạy khối bên dưới khi confirmed is not True cho kết quả đúng.
            if confirmed is not True:
                # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'confirmation_required', f'Xem lại đủ {POSE_COUNT} ảnh và xác nhận trước khi lưu.'.
                raise ApplicationError("confirmation_required", f"Xem lại đủ {POSE_COUNT} ảnh và xác nhận trước khi lưu.")
            # Từ chối frame còn đang truyền của bước cũ sau khi chụp/chụp lại/tạm dừng.
            # Chỉ chạy khối bên dưới khi revision != session.revision cho kết quả đúng.
            if revision != session.revision:
                # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'stale_review', 'Ảnh đã thay đổi; xem lại và xác nhận lần nữa.', 409.
                raise ApplicationError("stale_review", "Ảnh đã thay đổi; xem lại và xác nhận lần nữa.", 409)
            # Chỉ chạy khối bên dưới khi set(session.captures) != set(POSES) cho kết quả đúng.
            if set(session.captures) != set(POSES):
                # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'incomplete_poses', f'Cần đủ {POSE_COUNT} tư thế khác nhau trước khi lưu.'.
                raise ApplicationError("incomplete_poses", f"Cần đủ {POSE_COUNT} tư thế khác nhau trước khi lưu.")
            # Chỉ chạy khối bên dưới khi session.reviewed_revision != session.revision cho kết quả đúng.
            if session.reviewed_revision != session.revision:
                # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'review_required', f'Hãy xem lại bộ {POSE_COUNT} ảnh mới nhất trước khi xác nhận lưu.'.
                raise ApplicationError("review_required", f"Hãy xem lại bộ {POSE_COUNT} ảnh mới nhất trước khi xác nhận lưu.")
            # Bảo đảm model đã nạp trước khi xử lý qua self.service._require_model(...).
            self.service._require_model()
            # Chỉ chạy khối bên dưới khi session.version != self.service.runtime.version cho kết quả đúng.
            if session.version != self.service.runtime.version:
                # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'model_changed', 'Model đã thay đổi; bắt đầu phiên đăng ký mới.', 409.
                raise ApplicationError("model_changed", "Model đã thay đổi; bắt đầu phiên đăng ký mới.", 409)
            # Gán cho people: đọc và xác minh kho hồ sơ qua self.service._registry(...).
            people = self.service._registry()
            # Chỉ chạy khối bên dưới khi any((p['person_id'] == session.person_id for p in people)) cho kết quả đúng.
            if any(p["person_id"] == session.person_id for p in people):
                # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'duplicate_person', 'person_id đã tồn tại; không ghi đè.', 409.
                raise ApplicationError("duplicate_person", "person_id đã tồn tại; không ghi đè.", 409)
            # Kiểm tra lại từng ảnh và tư thế ở thời điểm lưu, không chỉ dựa vào trạng thái trước đó.
            # Gán cho images: tạo danh sách bằng cách duyệt p trong POSES; mỗi phần tử là session.captures[p].copy().
            images = [session.captures[p].copy() for p in POSES]
            # Lặp qua zip(POSES, images); mỗi lượt gán phần tử cho (expected, image).
            for expected, image in zip(POSES, images):
                # Kiểm tra kiểu mảng, số kênh và kích thước RGB bằng validate_rgb(...); dữ liệu/tham số: image, self.service.settings.
                validate_rgb(image, self.service.settings)
                # Tách kết quả theo thứ tự và gán cho (pose, _): đo chất lượng và tư thế mặt qua self._inspect(...); dữ liệu/tham số: image.
                pose, _ = self._inspect(image)
                # Chỉ chạy khối bên dưới khi classify_pose(pose) != expected cho kết quả đúng.
                if classify_pose(pose) != expected:
                    # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'pose_changed', 'Ảnh không đạt tư thế khi kiểm tra lại; hãy chụp lại.'.
                    raise ApplicationError("pose_changed", "Ảnh không đạt tư thế khi kiểm tra lại; hãy chụp lại.")
            # Gán cho rt: self.service.runtime (đọc thuộc tính).
            rt = self.service.runtime
            # Gán cho observed: bọc mạng để ghi lại các đầu ra đã suy luận bằng RecordingNet(...); dữ liệu/tham số: rt.net.
            observed = RecordingNet(rt.net)
            # Chạy đăng ký trong kho tạm để tạo vector mà chưa chạm kho hồ sơ chính.
            # Mở khối with để tạo thư mục tạm và tự xóa khi ra khỏi khối with bằng tempfile.TemporaryDirectory(...); dữ liệu/tham số: prefix='face_enrollment_'; gọi tài nguyên là folder; tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
            with tempfile.TemporaryDirectory(prefix="face_enrollment_") as folder:
                # Gán cho staging: ghép đường dẫn bằng toán tử / của pathlib trong Path(folder) / 'staging.sqlite'. Ở đây / nối thư mục với tên file, không phải phép chia số.
                staging = Path(folder) / "staging.sqlite"
                # Khởi tạo kho SQLite theo cấu trúc FaceNet bằng initialize_enrollment_registry(...); dữ liệu/tham số: staging.
                initialize_enrollment_registry(staging)
                # Gán cho result: tạo vector đại diện từ các ảnh và ghi hồ sơ bằng enroll_person_from_images(...); dữ liệu/tham số: session.person_id, images, rt.detector.
                result = enroll_person_from_images(session.person_id, images, rt.detector, observed, staging, rt.version)
                # Pipeline đã kiểm tra đầu ra thô; dùng đúng những vector đã ghi lại
                # để tránh chạy lại và vô tình kiểm tra một bộ vector khác.
                # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
                try:
                    # So vector của các ảnh để phát hiện bộ ảnh không nhất quán về danh tính.
                    # Kiểm tra các vector ảnh có nhất quán danh tính bằng validate_embedding_consistency(...); dữ liệu/tham số: [output[0] for output in observed.outputs], self.policy.min_similarity, expected_count=POSE_COUNT.
                    validate_embedding_consistency(
                        # Truyền expected_count (số vector đầu ra phải có) = POSE_COUNT.
                        [output[0] for output in observed.outputs], self.policy.min_similarity, expected_count=POSE_COUNT,
                    # Đóng phần biểu thức/tham số nhiều dòng đang mở; các dấu ngoặc không tạo bước xử lý riêng.
                    )
                # Bắt lỗi ValueError và đặt tên error để xử lý trong khối bên dưới.
                except ValueError as error:
                    # Phát sinh lỗi: tạo lỗi có mã, thông báo và mã HTTP bằng ApplicationError(...); dữ liệu/tham số: 'identity_inconsistent', f'{POSE_COUNT} ảnh chưa nhất quán danh tính. Hãy chụp lại hoặc bắt đầu phiên mới.'; from giữ lại lỗi gốc để truy nguyên.
                    raise ApplicationError("identity_inconsistent", f"{POSE_COUNT} ảnh chưa nhất quán danh tính. Hãy chụp lại hoặc bắt đầu phiên mới.") from error
                # Chỉ ghi kho chính khi mọi kiểm tra đã qua; vector đại diện và tên được ghi đồng bộ.
                # Lưu vector và tên trong cùng giao dịch bằng commit_identity(...); dữ liệu/tham số: self.service.path, session.person_id, session.display_name, num_images=result['num_images'].
                commit_identity(
                    # Tiếp phần biểu thức nhiều dòng: session.display_name (đọc thuộc tính).
                    self.service.path, session.person_id, session.display_name, result["embedding"], rt.version,
                    # Truyền num_images (số ảnh tạo nên hồ sơ) = result['num_images'].
                    num_images=result["num_images"],
                # Đóng phần biểu thức/tham số nhiều dòng đang mở; các dấu ngoặc không tạo bước xử lý riêng.
                )
            # Gán cho response: từ điển {'person_id': session.person_id, 'display_name': session.display_name, 'num_images':… (ánh xạ khóa → giá trị).
            response = {"person_id": session.person_id, "display_name": session.display_name,
                        # Tiếp phần biểu thức nhiều dòng: lấy phần tử/lát cắt result['num_images'] từ result.
                        "num_images": result["num_images"], "status": "enrolled"}
            # Xóa self.sessions[token] khỏi vùng dữ liệu hiện tại.
            del self.sessions[token]
            # Kết thúc hàm và trả response.
            return response
