# Ước lượng hướng đầu từ 5 điểm mốc: hai mắt, mũi và hai khóe miệng.
# Hoãn việc đánh giá gợi ý kiểu: tên kiểu có thể được khai báo sau mà không gây lỗi khi định nghĩa hàm.
from __future__ import annotations

# Lấy dataclass từ dataclasses để tạo lớp dữ liệu và sao chép/chuyển đổi dữ liệu của lớp.
from dataclasses import dataclass

# Nạp cv2 để xử lý ảnh, chạy ONNX và tính toán hình học camera bằng OpenCV.
import cv2
# Nạp numpy dưới tên ngắn np để tính toán mảng ảnh, vector và ma trận.
import numpy as np


# Ba góc tính theo độ: yaw quay trái/phải, pitch ngẩng/cúi, roll nghiêng đầu.
# Decorator dataclass: tự sinh hàm khởi tạo từ các trường; frozen=True chặn gán lại trường nếu được bật.
@dataclass(frozen=True)
# Định nghĩa lớp HeadPose; lớp gom dữ liệu và các hàm liên quan.
class HeadPose:
    # Chuỗi tài liệu (docstring) mô tả cách dùng; Python lưu nội dung này trong __doc__.
    """Signed yaw, pitch and roll in degrees; see the module's axis convention."""

    # Khai báo yaw có kiểu gợi ý float; giá trị được cung cấp khi tạo đối tượng.
    yaw: float
    # Khai báo pitch có kiểu gợi ý float; giá trị được cung cấp khi tạo đối tượng.
    pitch: float
    # Khai báo roll có kiểu gợi ý float; giá trị được cung cấp khi tạo đối tượng.
    roll: float


# Đơn vị độ dài tùy ý vì chỉ dùng tỷ lệ hình học; khi chưa quay, trục x hướng
# sang phải ảnh, y xuống dưới; z âm đưa mũi gần camera. Hai mắt và
# hai khóe miệng nằm hai bên; mũi nhô ra tạo chiều sâu cho mẫu 3D.
# Mẫu 3D dùng tỷ lệ tương đối: x sang phải ảnh, y xuống dưới, z âm đưa mũi gần camera.
# Gán cho _MODEL_POINTS: tạo mảng NumPy từ dữ liệu bằng np.array(...); dữ liệu/tham số: [[-32.0, -28.0, 0.0], [32.0, -28.0, 0.0], [0.0, 0.0, -24.0], [-24.0, 32.0, -5.0], [24.0, 32.0, -5.0…, dtype=np.float64.
_MODEL_POINTS = np.array([
    # Tiếp phần biểu thức nhiều dòng: danh sách [-32.0, -28.0, 0.0] ([] giữ các phần tử theo thứ tự).
    [-32.0, -28.0, 0.0],
    # Tiếp phần biểu thức nhiều dòng: danh sách [32.0, -28.0, 0.0] ([] giữ các phần tử theo thứ tự).
    [32.0, -28.0, 0.0],
    # Tiếp phần biểu thức nhiều dòng: danh sách [0.0, 0.0, -24.0] ([] giữ các phần tử theo thứ tự).
    [0.0, 0.0, -24.0],
    # Tiếp phần biểu thức nhiều dòng: danh sách [-24.0, 32.0, -5.0] ([] giữ các phần tử theo thứ tự).
    [-24.0, 32.0, -5.0],
    # Tiếp phần biểu thức nhiều dòng: danh sách [24.0, 32.0, -5.0] ([] giữ các phần tử theo thứ tự).
    [24.0, 32.0, -5.0],
# Truyền dtype (kiểu số của từng phần tử mảng) = np.float64.
], dtype=np.float64)
# Đặt cờ truy cập mảng, ở đây dùng để khóa khả năng ghi qua _MODEL_POINTS.setflags(...); dữ liệu/tham số: write=False.
_MODEL_POINTS.setflags(write=False)

# Các ngưỡng tính theo độ; khoảng hở giữa các nhóm giúp tránh chụp khi đang chuyển tư thế.
# Góc quay ngang tối đa để còn được coi là nhìn thẳng, tính bằng độ.
FRONT_MAX_YAW = 16.0
# Góc ngẩng/cúi tối đa của tư thế nhìn thẳng, tính bằng độ.
FRONT_MAX_PITCH = 10.0
# Phải quay ngang ít nhất ngưỡng này mới nhận là trái/phải; tránh chụp lúc vừa bắt đầu quay.
TURN_MIN_YAW = 18.0
# Không chấp nhận quay ngang quá ngưỡng này vì mặt có thể bị che khuất nhiều.
TURN_MAX_YAW = 50.0
# Trong lúc quay trái/phải, độ ngẩng/cúi phải nằm trong giới hạn này.
TURN_MAX_PITCH = 12.0
# Độ ngẩng/cúi nhỏ nhất để nhận là up/down; dùng trị tuyệt đối theo độ.
TILT_MIN_PITCH = 15.0
# Độ ngẩng/cúi lớn nhất còn cho phép phân loại là up/down.
TILT_MAX_PITCH = 35.0
# Khi ngẩng đầu, độ quay ngang không được vượt ngưỡng này.
UP_MAX_YAW = 12.0
# Khi cúi đầu, độ quay ngang không được vượt ngưỡng này.
DOWN_MAX_YAW = 14.0
# Không phân loại tư thế nếu đầu nghiêng quá ngưỡng roll này.
MAX_CLASSIFIED_ROLL = 15.0


# Loại điểm mốc sai vị trí/tỷ lệ; trả khoảng cách hai mắt để chuẩn hóa sai số.
# Định nghĩa hàm _valid_geometry(points, height, width). -> float | None là gợi ý kiểu kết quả, không tự ép kiểu.
def _valid_geometry(points: np.ndarray, height: int, width: int) -> float | None:
    # Chuỗi tài liệu (docstring) mô tả cách dùng; Python lưu nội dung này trong __doc__.
    """Return interocular pixel distance, or None for unreliable geometry."""
    # Chỉ chạy khối bên dưới khi np.any(points < 0) or np.any(points >= [width, height]) cho kết quả đúng.
    if np.any(points < 0) or np.any(points >= [width, height]):
        # Kết thúc hàm và trả None (chưa có/không có giá trị).
        return None
    # Lấy tọa độ mắt thứ hai trừ mắt thứ nhất để tạo vector nối hai mắt trong mặt phẳng ảnh.
    eyes = points[1] - points[0]
    # Tính độ dài vector nối hai mắt theo công thức căn(dx²+dy²), rồi chuyển số NumPy sang float của Python.
    eye_distance = float(np.linalg.norm(eyes))
    # Chỉ chạy khối bên dưới khi eye_distance < 12.0 or eyes[0] <= 0 cho kết quả đúng.
    if eye_distance < 12.0 or eyes[0] <= 0:
        # Kết thúc hàm và trả None (chưa có/không có giá trị).
        return None
    # Hệ trục theo đường hai mắt giúp kiểm tra hình học khi đầu hơi nghiêng.
    # Đổi sang hệ trục theo đường hai mắt để kiểm tra tỷ lệ ngay cả khi đầu hơi nghiêng.
    # Chia vector hai mắt cho độ dài của nó để thu vector đơn vị hướng ngang theo khuôn mặt.
    horizontal = eyes / eye_distance
    # Quay vector ngang 90° để có hướng dọc vuông góc trong hệ tọa độ khuôn mặt.
    downward = np.array([-horizontal[1], horizontal[0]])
    # Gán cho local: chia trong (points - points[:2].mean(axis=0)) @ np.column_stack([horizontal, downward]) / eye_d….
    local = (points - points[:2].mean(axis=0)) @ np.column_stack(
        # Tiếp phần biểu thức nhiều dòng: danh sách [horizontal, downward] ([] giữ các phần tử theo thứ tự).
        [horizontal, downward]
    # Tiếp phần biểu thức nhiều dòng: eye_distance (khoảng cách hai mắt tính theo pixel).
    ) / eye_distance
    # Lấy hai điểm khóe miệng ở hàng 3 và 4, tính trung bình theo từng cột để có tâm miệng.
    mouth = local[3:].mean(axis=0)
    # Lấy hoành độ khóe miệng phải trừ khóe trái để đo độ rộng miệng đã chuẩn hóa.
    mouth_width = local[4, 0] - local[3, 0]
    # Chỉ chạy khối bên dưới khi not (0.25 <= mouth_width <= 1.25 and 0.3 <= mouth[1] <= 1.8) cho kết quả đúng.
    if not (0.25 <= mouth_width <= 1.25 and 0.30 <= mouth[1] <= 1.80):
        # Kết thúc hàm và trả None (chưa có/không có giá trị).
        return None
    # Chỉ chạy khối bên dưới khi abs(mouth[0]) > 0.65 or abs(local[4, 1] - local[3, 1]) > 0.4 cho kết quả đúng.
    if abs(mouth[0]) > 0.65 or abs(local[4, 1] - local[3, 1]) > 0.40:
        # Kết thúc hàm và trả None (chưa có/không có giá trị).
        return None
    # Chỉ chạy khối bên dưới khi not -0.15 <= local[2, 1] <= 1.15 * mouth[1] cho kết quả đúng.
    if not (-0.15 <= local[2, 1] <= 1.15 * mouth[1]):
        # Kết thúc hàm và trả None (chưa có/không có giá trị).
        return None
    # Chỉ chạy khối bên dưới khi abs(local[2, 0]) > 0.85 cho kết quả đúng.
    if abs(local[2, 0]) > 0.85:
        # Kết thúc hàm và trả None (chưa có/không có giá trị).
        return None
    # Broadcast hai mảng (5,1,2) và (1,5,2) để tạo 25 vector nối từng cặp điểm; norm theo trục 2 cho bảng khoảng cách 5×5.
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    # Chỉ chạy khối bên dưới khi np.any(distances[np.triu_indices(5, 1)] < max(2.0, 0.06 * eye_distance)) cho kết quả đúng.
    if np.any(distances[np.triu_indices(5, 1)] < max(2.0, 0.06 * eye_distance)):
        # Kết thúc hàm và trả None (chưa có/không có giá trị).
        return None
    # Kết thúc hàm và trả eye_distance (khoảng cách hai mắt tính theo pixel).
    return eye_distance


# Tách ma trận quay thành yaw, pitch, roll rồi đổi từ radian sang độ.
# Định nghĩa hàm _angles(rotation). -> HeadPose là gợi ý kiểu kết quả, không tự ép kiểu.
def _angles(rotation: np.ndarray) -> HeadPose:
    # Chuỗi tài liệu (docstring) mô tả cách dùng; Python lưu nội dung này trong __doc__.
    """Decompose Rz @ Ry @ Rx without treating a Rodrigues vector as Euler."""
    # Gán cho pitch: tính góc từ hai thành phần và giữ đúng góc phần tư bằng np.arctan2(...); dữ liệu/tham số: rotation[2, 1], rotation[2, 2].
    pitch = np.arctan2(rotation[2, 1], rotation[2, 2])
    # Gán cho yaw: tính góc từ hai thành phần và giữ đúng góc phần tư bằng np.arctan2(...); dữ liệu/tham số: -rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]) (tính căn tổng bình phương hai thành phần bằng np.hypot(...)).
    yaw = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
    # Gán cho roll: tính góc từ hai thành phần và giữ đúng góc phần tư bằng np.arctan2(...); dữ liệu/tham số: rotation[1, 0], rotation[0, 0].
    roll = np.arctan2(rotation[1, 0], rotation[0, 0])
    # Kết thúc hàm và trả tạo bộ ba góc yaw, pitch, roll bằng HeadPose(...); dữ liệu/tham số: *map(float, np.rad2deg([yaw, pitch, roll])).
    return HeadPose(*map(float, np.rad2deg([yaw, pitch, roll])))


# Khớp mẫu mặt 3D với 5 điểm ảnh 2D bằng PnP; trả None nếu không đủ tin cậy.
# Định nghĩa hàm estimate_head_pose(landmarks, image_shape). -> HeadPose | None là gợi ý kiểu kết quả, không tự ép kiểu.
def estimate_head_pose(landmarks: object, image_shape: object) -> HeadPose | None:
    # Chuỗi tài liệu, dòng 1: Nhận 5 cặp tọa độ MTCNN trên ảnh gốc, trả góc đầu hoặc None nếu không đủ tin cậy.
    # Chuỗi tài liệu, dòng 2: dòng trống để tách nội dung cho dễ đọc.
    # Chuỗi tài liệu, dòng 3: image_shape nhận (cao, rộng) hoặc frame.shape là (cao, rộng, số kênh).
    # Chuỗi tài liệu, dòng 4: Chấp nhận các điểm số thực dạng mảng mà không sửa dữ liệu đầu vào.
    # Chuỗi tài liệu, dòng 5: Loại dữ liệu thiếu, sai hình dạng, NaN/vô cực, ngoài ảnh, hình học suy biến hoặc mặt quá nhỏ;
    # Chuỗi tài liệu, dòng 6: PnP lỗi, nhiều nghiệm khó phân biệt, sai số chiếu lớn hoặc góc quá cực đoan đều trả None.
    # Chuỗi tài liệu, dòng 7: Khi không đo được thì không tự giả định người dùng đang nhìn thẳng.
    # Chuỗi tài liệu, dòng 8: dòng trống để tách nội dung cho dễ đọc.
    # Chuỗi tài liệu, dòng 9: SQPnP dùng được 5 điểm không đồng phẳng; DLT khởi tạo của ITERATIVE cần 6 điểm.
    # Chuỗi tài liệu, dòng 10: Tinh chỉnh từng nghiệm SQPnP bằng LM và yêu cầu cả 5 điểm khớp ảnh;
    # Chuỗi tài liệu, dòng 11: RMS không quá 6%, sai số điểm tệ nhất không quá 12% khoảng cách hai mắt;
    # Chuỗi tài liệu, dòng 12: điểm phải ở trước camera, |yaw|≤60°, |pitch|≤45°, |roll|≤60°; riêng góc cúi
    # Chuỗi tài liệu, dòng 13: cho RMS đến 11% và sai số điểm tệ nhất đến 14%; hai nghiệm gần ngang nhau
    # Chuỗi tài liệu, dòng 14: mà lệch nhau hơn 12° bị từ chối; dòng kế tiếp dẫn tới tài liệu thuật toán.
    # Chuỗi tài liệu, dòng 15: Địa chỉ tài liệu solvePnP của OpenCV để tra cứu thêm.
    # Chuỗi tài liệu, dòng 16: Kết thúc chuỗi tài liệu.
    """Fit one original-image MTCNN ndarray (5,2), returning pose or None.

    ``image_shape`` is (height,width) or frame.shape (height,width,channels).
    Real numeric array-like points are accepted without modifying the input.
    Missing/wrong-shape/nonfinite/out-of-frame/degenerate points, tiny faces,
    failed/ambiguous PnP, large reprojection error and extreme views return None.
    There is no fallback to a fabricated frontal pose.

    SQPnP supports five non-planar points; unseeded ITERATIVE's DLT needs six.
    Refine each SQPnP candidate with LM and require all five points to fit
    (RMS <= 6% and worst residual <= 12% of interocular distance), positive
    camera depth, |yaw|<=60, |pitch|<=45, |roll|<=60. Fits within the down
    pose bounds allow up to 11% RMS and 14% worst residual. Near-equal fits
    differing by >12 degrees are rejected. See the official solver contract:
    https://docs.opencv.org/4.12.0/d5/d1f/calib3d_solvePnP.html
    """
    # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
    try:
        # Gán cho shape: tạo bộ giá trị bằng tuple(...); dữ liệu/tham số: image_shape.
        shape = tuple(image_shape)
        # Chỉ chạy khối bên dưới khi len(shape) not in (2, 3) or any((isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0 for v in … cho kết quả đúng.
        if len(shape) not in (2, 3) or any(
            # Tiếp phần biểu thức nhiều dòng: kết hợp bằng or: lấy giá trị đúng đầu tiên nếu có trong isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0.
            isinstance(v, (bool, np.bool_))
            # Bổ sung trường hợp được chấp nhận bằng or: or not isinstance(v, (int, np.integer)) or v <= 0.
            or not isinstance(v, (int, np.integer)) or v <= 0
            # Phần for của biểu thức rút gọn: duyệt từng phần tử theo thứ tự và tạo kết quả.
            for v in shape
        # Đóng phần biểu thức/tham số nhiều dòng đang mở; các dấu ngoặc không tạo bước xử lý riêng.
        ):
            # Kết thúc hàm và trả None (chưa có/không có giá trị).
            return None
        # Tách kết quả theo thứ tự và gán cho (height, width): áp dụng hàm cho từng phần tử bằng map(...); dữ liệu/tham số: int, shape[:2].
        height, width = map(int, shape[:2])
        # Gán cho raw: biểu diễn dữ liệu thành mảng NumPy, có thể dùng lại bộ nhớ sẵn có bằng np.asarray(...); dữ liệu/tham số: landmarks.
        raw = np.asarray(landmarks)
        # Chỉ chạy khối bên dưới khi raw.shape != (5, 2) or raw.dtype.kind not in 'iuf' cho kết quả đúng.
        if raw.shape != (5, 2) or raw.dtype.kind not in "iuf":
            # Kết thúc hàm và trả None (chưa có/không có giá trị).
            return None
        # Gán cho points: đưa dữ liệu về mảng liên tục trong bộ nhớ bằng np.ascontiguousarray(...); dữ liệu/tham số: raw, dtype=np.float64.
        points = np.ascontiguousarray(raw, dtype=np.float64)
        # Chỉ chạy khối bên dưới khi not np.isfinite(points).all() cho kết quả đúng.
        if not np.isfinite(points).all():
            # Kết thúc hàm và trả None (chưa có/không có giá trị).
            return None
    # Bắt lỗi (TypeError, ValueError, OverflowError) để xử lý trong khối bên dưới.
    except (TypeError, ValueError, OverflowError):
        # Kết thúc hàm và trả None (chưa có/không có giá trị).
        return None
    # Gán cho eye_distance: gọi _valid_geometry(...) với points, height, width; dữ liệu/tham số: points, height, width.
    eye_distance = _valid_geometry(points, height, width)
    # Chỉ chạy khối bên dưới khi eye_distance is None cho kết quả đúng.
    if eye_distance is None:
        # Kết thúc hàm và trả None (chưa có/không có giá trị).
        return None

    # Xấp xỉ tiêu cự bằng cạnh ảnh lớn nhất; tâm ảnh là tâm chiếu của camera.
    # Chưa hiệu chuẩn camera nên xấp xỉ tiêu cự bằng cạnh dài nhất của ảnh, tính theo pixel.
    focal = float(max(height, width))
    # Gán cho camera: tạo mảng NumPy từ dữ liệu bằng np.array(...); dữ liệu/tham số: [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]].
    camera = np.array([[focal, 0.0, width / 2.0],
                       # Tiếp phần biểu thức nhiều dòng: danh sách [0.0, focal, height / 2.0] ([] giữ các phần tử theo thứ tự).
                       [0.0, focal, height / 2.0],
                       # Tiếp phần biểu thức nhiều dòng: danh sách [0.0, 0.0, 1.0] ([] giữ các phần tử theo thứ tự).
                       [0.0, 0.0, 1.0]])
    # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
    try:
        # PnP tìm góc quay và vị trí mẫu 3D sao cho khi chiếu lên ảnh khớp các điểm 2D.
        # Tách kết quả theo thứ tự và gán cho (count, rvecs, tvecs, _): tìm các nghiệm quay/tịnh tiến khớp mẫu mặt 3D với điểm ảnh 2D bằng cv2.solvePnPGeneric(...); dữ liệu/tham số: _MODEL_POINTS, points, camera, flags=cv2.SOLVEPNP_SQPNP.
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            # Truyền flags (thuật toán/tùy chọn cho OpenCV) = cv2.SOLVEPNP_SQPNP.
            _MODEL_POINTS, points, camera, None, flags=cv2.SOLVEPNP_SQPNP
        # Đóng phần biểu thức/tham số nhiều dòng đang mở; các dấu ngoặc không tạo bước xử lý riêng.
        )
    # Bắt lỗi cv2.error để xử lý trong khối bên dưới.
    except cv2.error:
        # Kết thúc hàm và trả None (chưa có/không có giá trị).
        return None
    # Chỉ chạy khối bên dưới khi not count cho kết quả đúng.
    if not count:
        # Kết thúc hàm và trả None (chưa có/không có giá trị).
        return None

    # Khai báo candidates có kiểu gợi ý list[tuple[float, HeadPose]]; gán danh sách [] ([] giữ các phần tử theo thứ tự).
    candidates: list[tuple[float, HeadPose]] = []
    # Lặp qua zip(rvecs, tvecs); mỗi lượt gán phần tử cho (rvec, tvec).
    for rvec, tvec in zip(rvecs, tvecs):
        # Bắt đầu đoạn có thể phát sinh lỗi; các nhánh except/finally phía sau sẽ xử lý tương ứng.
        try:
            # Tách kết quả theo thứ tự và gán cho (rvec, tvec): tinh chỉnh nghiệm PnP bằng tối ưu Levenberg–Marquardt bằng cv2.solvePnPRefineLM(...); dữ liệu/tham số: _MODEL_POINTS, points, camera.
            rvec, tvec = cv2.solvePnPRefineLM(
                # Tiếp phần biểu thức nhiều dòng: _MODEL_POINTS.
                _MODEL_POINTS, points, camera, None, rvec.copy(), tvec.copy()
            # Đóng phần biểu thức/tham số nhiều dòng đang mở; các dấu ngoặc không tạo bước xử lý riêng.
            )
            # Chỉ chạy khối bên dưới khi not (np.isfinite(rvec).all() and np.isfinite(tvec).all()) cho kết quả đúng.
            if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
                # Bỏ phần còn lại của lượt hiện tại và chuyển sang lượt lặp kế tiếp.
                continue
            # Đổi vector quay rvec thành ma trận quay 3×3; dấu _ bỏ qua ma trận đạo hàm không cần dùng.
            rotation, _ = cv2.Rodrigues(rvec)
            # Nhân các điểm mẫu với ma trận quay chuyển vị, rồi cộng tịnh tiến để đưa mặt 3D sang hệ tọa độ camera.
            camera_points = _MODEL_POINTS @ rotation.T + tvec.reshape(3)
            # Chỉ chạy khối bên dưới khi np.any(camera_points[:, 2] <= 0) cho kết quả đúng.
            if np.any(camera_points[:, 2] <= 0):
                # Bỏ phần còn lại của lượt hiện tại và chuyển sang lượt lặp kế tiếp.
                continue
            # Gán cho pose: gọi _angles(...) với rotation; dữ liệu/tham số: rotation.
            pose = _angles(rotation)
            # Chỉ chạy khối bên dưới khi abs(pose.yaw) > 60 or abs(pose.pitch) > 45 or abs(pose.roll) > 60 cho kết quả đúng.
            if abs(pose.yaw) > 60 or abs(pose.pitch) > 45 or abs(pose.roll) > 60:
                # Bỏ phần còn lại của lượt hiện tại và chuyển sang lượt lặp kế tiếp.
                continue
            # Chiếu ngược nghiệm về ảnh và đo sai số; chuẩn hóa theo khoảng cách hai mắt.
            # Tách kết quả theo thứ tự và gán cho (projected, _): chiếu điểm 3D qua camera để thu tọa độ ảnh 2D bằng cv2.projectPoints(...); dữ liệu/tham số: _MODEL_POINTS, rvec, tvec.
            projected, _ = cv2.projectPoints(_MODEL_POINTS, rvec, tvec, camera, None)
            # Đưa điểm chiếu về mảng 5×2, trừ điểm thật rồi tính khoảng cách pixel cho từng điểm.
            residual = np.linalg.norm(projected.reshape(5, 2) - points, axis=1)
            # Bình phương sai số, lấy trung bình rồi căn bậc hai; chia khoảng cách hai mắt để ngưỡng không phụ thuộc mặt to/nhỏ.
            rms = float(np.sqrt(np.mean(residual ** 2)) / eye_distance)
            # Khi cúi, sai số so với mẫu mặt 5 điểm chung có thể lớn hơn
            # dù hình học các điểm mốc vẫn hợp lệ, nên dùng ngưỡng riêng.
            # Tách kết quả theo thứ tự và gán cho (rms_limit, point_limit): chọn (0.11, 0.14) nếu classify_pose(pose) == 'down'; ngược lại chọn (0.06, 0.12).
            rms_limit, point_limit = (0.11, 0.14) if classify_pose(pose) == "down" else (0.06, 0.12)
            # Chỉ chạy khối bên dưới khi rms > rms_limit or float(residual.max()) / eye_distance > point_limit cho kết quả đúng.
            if rms > rms_limit or float(residual.max()) / eye_distance > point_limit:
                # Bỏ phần còn lại của lượt hiện tại và chuyển sang lượt lặp kế tiếp.
                continue
            # Thêm một phần tử vào cuối danh sách qua candidates.append(...); dữ liệu/tham số: (rms, pose).
            candidates.append((rms, pose))
        # Bắt lỗi cv2.error để xử lý trong khối bên dưới.
        except cv2.error:
            # Bỏ phần còn lại của lượt hiện tại và chuyển sang lượt lặp kế tiếp.
            continue
    # Chỉ chạy khối bên dưới khi not candidates cho kết quả đúng.
    if not candidates:
        # Kết thúc hàm và trả None (chưa có/không có giá trị).
        return None
    # Ưu tiên nghiệm ít sai số nhất; nếu hai nghiệm gần ngang nhau nhưng khác góc lớn thì bỏ.
    # Sắp nghiệm theo phần tử thứ 0 là sai số RMS, từ nhỏ đến lớn; key=lambda chỉ định giá trị dùng để so sánh.
    candidates.sort(key=lambda candidate: candidate[0])
    # Lấy nghiệm ít sai số nhất ở đầu danh sách; tách tuple thành best_error (sai số) và best (ba góc).
    best_error, best = candidates[0]
    # Lặp qua candidates[1:]; mỗi lượt gán phần tử cho (error, other).
    for error, other in candidates[1:]:
        # Chỉ chạy khối bên dưới khi error <= best_error + 0.01 and max(abs(best.yaw - other.yaw), abs(best.pitch - other.pitch), abs(best.roll - other.roll)) > … cho kết quả đúng.
        if error <= best_error + 0.01 and max(
            # Tiếp phần biểu thức nhiều dòng: lấy trị tuyệt đối bằng abs(...).
            abs(best.yaw - other.yaw), abs(best.pitch - other.pitch),
            # Tiếp phần biểu thức nhiều dòng: lấy trị tuyệt đối bằng abs(...).
            abs(best.roll - other.roll),
        # Tiếp phần biểu thức nhiều dòng: giá trị 12.0.
        ) > 12.0:
            # Kết thúc hàm và trả None (chưa có/không có giá trị).
            return None
    # Kết thúc hàm và trả best.
    return best


# Đổi góc thành nhãn tư thế; trái/phải là phía của người dùng trong ảnh không lật gương.
# Định nghĩa hàm classify_pose(pose). -> str | None là gợi ý kiểu kết quả, không tự ép kiểu.
def classify_pose(pose: HeadPose | None) -> str | None:
    # Chuỗi tài liệu, dòng 1: Trả front/left/right/up/down, hoặc None nếu tư thế chưa rõ hay trộn nhiều hướng.
    # Chuỗi tài liệu, dòng 2: dòng trống để tách nội dung cho dễ đọc.
    # Chuỗi tài liệu, dòng 3: Trái/phải tính theo cơ thể người dùng; nhìn thẳng cần
    # Chuỗi tài liệu, dòng 4: |yaw|≤16° và |pitch|≤10°; quay ngang cần 18°≤|yaw|≤50° và |pitch|≤12°;
    # Chuỗi tài liệu, dòng 5: ngẩng/cúi cần 15°≤|pitch|≤35°, với |yaw|≤12° khi ngẩng hoặc ≤14° khi cúi.
    # Chuỗi tài liệu, dòng 6: Mọi tư thế đều phải nghiêng đầu không quá 15°.
    # Chuỗi tài liệu, dòng 7: Khoảng hở giữa các ngưỡng giúp tránh chụp quá sớm lúc đang chuyển góc.
    # Chuỗi tài liệu, dòng 8: Hàm chỉ xét góc hiện tại; việc giữ ổn định theo thời gian do bộ quản lý phiên thực hiện.
    # Chuỗi tài liệu, dòng 9: Kết thúc chuỗi tài liệu.
    """Return front/left/right/up/down, or None for uncertain/mixed poses.

    Labels left/right always refer to the USER'S ANATOMICAL side. Front needs
    |yaw|<=16 and |pitch|<=10; side turns need 18<=|yaw|<=50 and |pitch|<=12;
    up/down need 15<=|pitch|<=35 with |yaw|<=12 for up or <=14 for down.
    All need |roll|<=15.
    Gaps between these ranges deliberately prevent premature captures.
    This stateless function does not supply temporal stability/hysteresis.
    """
    # Chỉ chạy khối bên dưới khi not isinstance(pose, HeadPose) cho kết quả đúng.
    if not isinstance(pose, HeadPose):
        # Kết thúc hàm và trả None (chưa có/không có giá trị).
        return None
    # Gán cho values: biểu diễn dữ liệu thành mảng NumPy, có thể dùng lại bộ nhớ sẵn có bằng np.asarray(...); dữ liệu/tham số: [pose.yaw, pose.pitch, pose.roll], dtype=np.float64.
    values = np.asarray([pose.yaw, pose.pitch, pose.roll], dtype=np.float64)
    # Chỉ chạy khối bên dưới khi not np.isfinite(values).all() cho kết quả đúng.
    if not np.isfinite(values).all():
        # Kết thúc hàm và trả None (chưa có/không có giá trị).
        return None
    # Tách ba giá trị theo đúng thứ tự: quay trái/phải, ngẩng/cúi và nghiêng đầu.
    yaw, pitch, roll = values
    # Chỉ chạy khối bên dưới khi abs(roll) > MAX_CLASSIFIED_ROLL cho kết quả đúng.
    if abs(roll) > MAX_CLASSIFIED_ROLL:
        # Kết thúc hàm và trả None (chưa có/không có giá trị).
        return None
    # Chỉ chạy khối bên dưới khi abs(yaw) <= FRONT_MAX_YAW and abs(pitch) <= FRONT_MAX_PITCH cho kết quả đúng.
    if abs(yaw) <= FRONT_MAX_YAW and abs(pitch) <= FRONT_MAX_PITCH:
        # Kết thúc hàm và trả chuỗi 'front'.
        return "front"
    # Chỉ chạy khối bên dưới khi TURN_MIN_YAW <= abs(yaw) <= TURN_MAX_YAW and abs(pitch) <= TURN_MAX_PITCH cho kết quả đúng.
    if TURN_MIN_YAW <= abs(yaw) <= TURN_MAX_YAW and abs(pitch) <= TURN_MAX_PITCH:
        # Biểu thức điều kiện một dòng: yaw dương trả right, còn lại trả left; phía tính theo người được chụp.
        return "right" if yaw > 0 else "left"
    # Chỉ chạy khối bên dưới khi -TILT_MAX_PITCH <= pitch <= -TILT_MIN_PITCH and abs(yaw) <= UP_MAX_YAW cho kết quả đúng.
    if -TILT_MAX_PITCH <= pitch <= -TILT_MIN_PITCH and abs(yaw) <= UP_MAX_YAW:
        # Kết thúc hàm và trả chuỗi 'up'.
        return "up"
    # Chỉ chạy khối bên dưới khi TILT_MIN_PITCH <= pitch <= TILT_MAX_PITCH and abs(yaw) <= DOWN_MAX_YAW cho kết quả đúng.
    if TILT_MIN_PITCH <= pitch <= TILT_MAX_PITCH and abs(yaw) <= DOWN_MAX_YAW:
        # Kết thúc hàm và trả chuỗi 'down'.
        return "down"
    # Kết thúc hàm và trả None (chưa có/không có giá trị).
    return None
