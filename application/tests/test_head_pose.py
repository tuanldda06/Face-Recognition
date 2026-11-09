# Tạo các điểm mặt tổng hợp với góc đã biết để kiểm tra tính toán và phân loại hướng đầu.
# Lấy patch từ unittest.mock để thay tạm hàm/model và theo dõi lời gọi trong kiểm thử.
from unittest.mock import patch

# Nạp cv2 để xử lý ảnh, chạy ONNX và tính toán hình học camera bằng OpenCV.
import cv2
# Nạp numpy dưới tên ngắn np để tính toán mảng ảnh, vector và ma trận.
import numpy as np
# Nạp pytest để viết và chạy bài kiểm thử tự động.
import pytest

# Lấy HeadPose, classify_pose, estimate_head_pose từ application.head_pose.
from application.head_pose import HeadPose, classify_pose, estimate_head_pose


# Dữ liệu mẫu độc lập: không lấy mẫu mặt hoặc camera từ chính bộ ước lượng.
# Thứ tự hàng theo đầu ra O-Net đã giải mã, không dùng vector thô xen kẽ.
# Gán cho FACE: tạo mảng NumPy từ dữ liệu bằng np.array(...); dữ liệu/tham số: [[-32.0, -28.0, 0.0], [32.0, -28.0, 0.0], [0.0, 0.0, -24.0], [-24.0, 32.0, -5.0], [24.0, 32.0, -5.0….
FACE = np.array([[-32., -28., 0.], [32., -28., 0.], [0., 0., -24.],
                 # Tiếp phần biểu thức nhiều dòng: danh sách [-24.0, 32.0, -5.0] ([] giữ các phần tử theo thứ tự).
                 [-24., 32., -5.], [24., 32., -5.]])
# Gán cho SHAPE: bộ giá trị (480, 640, 3) (tuple giữ thứ tự, không gán lại từng phần tử).
SHAPE = (480, 640, 3)


# Tạo điểm ảnh 2D từ góc quay 3D đã biết, làm đáp án độc lập để so với bộ ước lượng.
# Định nghĩa hàm project(yaw, pitch, roll). Mặc định: yaw=0.0, pitch=0.0, roll=0.0. Sau dấu * phải truyền theo tên: shape=SHAPE, translation=(0.0, 0.0, 600.0), focal_scale=1.0, face=FACE.
def project(yaw=0., pitch=0., roll=0., *, shape=SHAPE, translation=(0., 0., 600.),
            # Tiếp phần biểu thức nhiều dòng: FACE.
            focal_scale=1., face=FACE):
    # Chuỗi tài liệu (docstring) mô tả cách dùng; Python lưu nội dung này trong __doc__.
    """Generate ground truth with explicit Euler matrices and cv2.projectPoints."""
    # Tách kết quả theo thứ tự và gán cho (y, p, r): đổi độ sang radian bằng np.deg2rad(...); dữ liệu/tham số: [yaw, pitch, roll].
    y, p, r = np.deg2rad([yaw, pitch, roll])
    # Gán cho rx: tạo mảng NumPy từ dữ liệu bằng np.array(...); dữ liệu/tham số: [[1.0, 0.0, 0.0], [0.0, np.cos(p), -np.sin(p)], [0.0, np.sin(p), np.cos(p)]].
    rx = np.array([[1., 0., 0.], [0., np.cos(p), -np.sin(p)],
                   # Tiếp phần biểu thức nhiều dòng: danh sách [0.0, np.sin(p), np.cos(p)] ([] giữ các phần tử theo thứ tự).
                   [0., np.sin(p), np.cos(p)]])
    # Gán cho ry: tạo mảng NumPy từ dữ liệu bằng np.array(...); dữ liệu/tham số: [[np.cos(y), 0.0, np.sin(y)], [0.0, 1.0, 0.0], [-np.sin(y), 0.0, np.cos(y)]].
    ry = np.array([[np.cos(y), 0., np.sin(y)], [0., 1., 0.],
                   # Tiếp phần biểu thức nhiều dòng: danh sách [-np.sin(y), 0.0, np.cos(y)] ([] giữ các phần tử theo thứ tự).
                   [-np.sin(y), 0., np.cos(y)]])
    # Gán cho rz: tạo mảng NumPy từ dữ liệu bằng np.array(...); dữ liệu/tham số: [[np.cos(r), -np.sin(r), 0.0], [np.sin(r), np.cos(r), 0.0], [0.0, 0.0, 1.0]].
    rz = np.array([[np.cos(r), -np.sin(r), 0.],
                   # Tiếp phần biểu thức nhiều dòng: danh sách [np.sin(r), np.cos(r), 0.0] ([] giữ các phần tử theo thứ tự).
                   [np.sin(r), np.cos(r), 0.], [0., 0., 1.]])
    # Tách kết quả theo thứ tự và gán cho (rvec, _): chuyển giữa vector quay và ma trận quay bằng cv2.Rodrigues(...); dữ liệu/tham số: rz @ ry @ rx.
    rvec, _ = cv2.Rodrigues(rz @ ry @ rx)
    # Tách kết quả theo thứ tự và gán cho (h, w): lấy phần tử/lát cắt shape[:2] từ shape.
    h, w = shape[:2]
    # Gán cho focal: nhân hoặc lặp trong max(h, w) * focal_scale.
    focal = max(h, w) * focal_scale
    # Gán cho camera: tạo mảng NumPy từ dữ liệu bằng np.array(...); dữ liệu/tham số: [[focal, 0.0, w / 2], [0.0, focal, h / 2], [0.0, 0.0, 1.0]].
    camera = np.array([[focal, 0., w / 2], [0., focal, h / 2], [0., 0., 1.]])
    # Tách kết quả theo thứ tự và gán cho (points, _): chiếu điểm 3D qua camera để thu tọa độ ảnh 2D bằng cv2.projectPoints(...); dữ liệu/tham số: face, rvec, np.array(translation) (tạo mảng NumPy từ dữ liệu bằng np.array(...)).
    points, _ = cv2.projectPoints(face, rvec, np.array(translation), camera, None)
    # Kết thúc hàm và trả đổi hình dạng mảng nhưng giữ số phần tử qua points.reshape(...); dữ liệu/tham số: 5, 2.
    return points.reshape(5, 2)


# Kiểm tra: Góc ước lượng gần góc tạo dữ liệu và được phân loại đúng nhãn.
# Decorator parametrize: chạy cùng một bài test với nhiều bộ dữ liệu liệt kê bên dưới.
@pytest.mark.parametrize("yaw,pitch,roll,label", [
    # Tiếp phần khai báo nhiều dòng: (0, 0, 0, "front"), (-28, 0, 0, "left"), (28, 0, 0, "right"),.
    (0, 0, 0, "front"), (-28, 0, 0, "left"), (28, 0, 0, "right"),
    # Tiếp phần khai báo nhiều dòng: (0, -24, 0, "up"), (0, 24, 0, "down"),.
    (0, -24, 0, "up"), (0, 24, 0, "down"),
    # Tiếp phần khai báo nhiều dòng: (-30, 5, -8, "left"), (27, -4, 9, "right"),.
    (-30, 5, -8, "left"), (27, -4, 9, "right"),
    # Tiếp phần khai báo nhiều dòng: (5, -22, 7, "up"), (-5, 22, -7, "down"),.
    (5, -22, 7, "up"), (-5, 22, -7, "down"),
    # Tiếp phần khai báo nhiều dòng: (0, 0, 25, None), (0, 0, -25, None), (26, 24, 3, None),.
    (0, 0, 25, None), (0, 0, -25, None), (26, 24, 3, None),
# Tiếp phần khai báo nhiều dòng: ]).
])
# Định nghĩa hàm test_recover_projected_angles_and_classify(yaw, pitch, roll, label).
def test_recover_projected_angles_and_classify(yaw, pitch, roll, label):
    # Gán cho pose: ước lượng ba góc đầu từ điểm mốc và kích thước ảnh bằng estimate_head_pose(...); dữ liệu/tham số: project(yaw, pitch, roll) (chiếu điểm 3D tổng hợp lên ảnh theo góc đã biết bằng project(...)), SHAPE.
    pose = estimate_head_pose(project(yaw, pitch, roll), SHAPE)
    # Kiểm chứng điều kiện isinstance(pose, HeadPose); sai thì test dừng bằng AssertionError.
    assert isinstance(pose, HeadPose)
    # Xác nhận các giá trị mảng gần bằng nhau trong sai số bằng np.testing.assert_allclose(...); dữ liệu/tham số: [pose.yaw, pose.pitch, pose.roll], [yaw, pitch, roll], atol=0.05.
    np.testing.assert_allclose([pose.yaw, pose.pitch, pose.roll],
                               # Truyền atol (sai số tuyệt đối cho phép khi so sánh số thực) = 0.05.
                               [yaw, pitch, roll], atol=0.05)
    # Kiểm chứng điều kiện classify_pose(pose) == label; sai thì test dừng bằng AssertionError.
    assert classify_pose(pose) == label


# Kiểm tra: Dấu góc quay đúng với bên trái/phải của người dùng khi camera không lật gương.
# Định nghĩa hàm test_signs_match_anatomical_instructions_in_unmirrored_camera().
def test_signs_match_anatomical_instructions_in_unmirrored_camera():
    # Bên phải người được chụp là bên trái người xem; sai dấu yaw sẽ làm test thất bại.
    # Gán cho right: chiếu điểm 3D tổng hợp lên ảnh theo góc đã biết bằng project(...); dữ liệu/tham số: yaw=28.
    right = project(yaw=28)
    # Gán cho left: chiếu điểm 3D tổng hợp lên ảnh theo góc đã biết bằng project(...); dữ liệu/tham số: yaw=-28.
    left = project(yaw=-28)
    # Kiểm chứng điều kiện right[2, 0] < right[:2, 0].mean(); sai thì test dừng bằng AssertionError.
    assert right[2, 0] < right[:2, 0].mean()
    # Kiểm chứng điều kiện left[2, 0] > left[:2, 0].mean(); sai thì test dừng bằng AssertionError.
    assert left[2, 0] > left[:2, 0].mean()
    # Kiểm chứng điều kiện classify_pose(estimate_head_pose(right, SHAPE)) == 'right'; sai thì test dừng bằng AssertionError.
    assert classify_pose(estimate_head_pose(right, SHAPE)) == "right"
    # Kiểm chứng điều kiện classify_pose(estimate_head_pose(left, SHAPE)) == 'left'; sai thì test dừng bằng AssertionError.
    assert classify_pose(estimate_head_pose(left, SHAPE)) == "left"
    # Khi cúi, mũi nhô ra dịch xuống so với đường nối hai mắt.
    # Tách kết quả theo thứ tự và gán cho (down, up): bộ giá trị (project(pitch=24), project(pitch=-24)) (tuple giữ thứ tự, không gán lại từng phần tử).
    down, up = project(pitch=24), project(pitch=-24)
    # Kiểm chứng điều kiện down[2, 1] - down[:2, 1].mean() > up[2, 1] - up[:2, 1].mean(); sai thì test dừng bằng AssertionError.
    assert down[2, 1] - down[:2, 1].mean() > up[2, 1] - up[:2, 1].mean()


# Kiểm tra: Thay độ phân giải, khoảng cách và vị trí mặt vẫn khôi phục được góc quay.
# Decorator parametrize: chạy cùng một bài test với nhiều bộ dữ liệu liệt kê bên dưới.
@pytest.mark.parametrize("shape,translation", [
    # Tiếp phần khai báo nhiều dòng: ((720, 1280, 3), (90., -50., 850.)),.
    ((720, 1280, 3), (90., -50., 850.)),
    # Tiếp phần khai báo nhiều dòng: ((800, 600), (-70., 60., 700.)),.
    ((800, 600), (-70., 60., 700.)),
    # Tiếp phần khai báo nhiều dòng: ((480, 640, 3), (0., 0., 1400.)),.
    ((480, 640, 3), (0., 0., 1400.)),
# Tiếp phần khai báo nhiều dòng: ]).
])
# Định nghĩa hàm test_original_pixel_coordinates_translation_resolution_and_scale(shape, translation).
def test_original_pixel_coordinates_translation_resolution_and_scale(shape, translation):
    # Gán cho points: chiếu điểm 3D tổng hợp lên ảnh theo góc đã biết bằng project(...); dữ liệu/tham số: -26, 4, 6, shape=shape, translation=translation.
    points = project(-26, 4, 6, shape=shape, translation=translation)
    # Gán cho pose: ước lượng ba góc đầu từ điểm mốc và kích thước ảnh bằng estimate_head_pose(...); dữ liệu/tham số: points, shape.
    pose = estimate_head_pose(points, shape)
    # Kiểm chứng điều kiện pose is not None; sai thì test dừng bằng AssertionError.
    assert pose is not None
    # Xác nhận các giá trị mảng gần bằng nhau trong sai số bằng np.testing.assert_allclose(...); dữ liệu/tham số: [pose.yaw, pose.pitch, pose.roll], [-26, 4, 6], atol=0.05.
    np.testing.assert_allclose([pose.yaw, pose.pitch, pose.roll], [-26, 4, 6], atol=.05)
    # Kiểm chứng điều kiện classify_pose(pose) == 'left'; sai thì test dừng bằng AssertionError.
    assert classify_pose(pose) == "left"


# Kiểm tra: Nhận được mảng float32 không liên tục mà không sửa dữ liệu đầu vào.
# Định nghĩa hàm test_accepts_mtcnn_float32_noncontiguous_array_and_does_not_modify_input().
def test_accepts_mtcnn_float32_noncontiguous_array_and_does_not_modify_input():
    # Gán cho backing: tạo mảng gồm các số 0 bằng np.zeros(...); dữ liệu/tham số: (5, 4), dtype=np.float32.
    backing = np.zeros((5, 4), dtype=np.float32)
    # Gán cho backing[:, ::2]: chiếu điểm 3D tổng hợp lên ảnh theo góc đã biết bằng project(...); dữ liệu/tham số: 28.
    backing[:, ::2] = project(28)
    # Gán cho points: lấy phần tử/lát cắt backing[:, ::2] từ backing.
    points = backing[:, ::2]
    # Kiểm chứng điều kiện not points.flags.c_contiguous; sai thì test dừng bằng AssertionError.
    assert not points.flags.c_contiguous
    # Gán cho before: tạo bản sao để thay đổi không tác động dữ liệu gốc qua backing.copy(...).
    before = backing.copy()
    # Kiểm chứng điều kiện classify_pose(estimate_head_pose(points, SHAPE)) == 'right'; sai thì test dừng bằng AssertionError.
    assert classify_pose(estimate_head_pose(points, SHAPE)) == "right"
    # Xác nhận hai mảng bằng nhau hoàn toàn bằng np.testing.assert_array_equal(...); dữ liệu/tham số: backing, before.
    np.testing.assert_array_equal(backing, before)


# Kiểm tra: Nhiễu nhỏ và tiêu cự xấp xỉ vẫn giữ đúng 5 nhãn tư thế.
# Decorator parametrize: chạy cùng một bài test với nhiều bộ dữ liệu liệt kê bên dưới.
@pytest.mark.parametrize("yaw,pitch,label", [
    # Tiếp phần khai báo nhiều dòng: (0, 0, "front"), (-30, 0, "left"), (30, 0, "right"),.
    (0, 0, "front"), (-30, 0, "left"), (30, 0, "right"),
    # Tiếp phần khai báo nhiều dòng: (0, -25, "up"), (0, 25, "down"),.
    (0, -25, "up"), (0, 25, "down"),
# Tiếp phần khai báo nhiều dòng: ]).
])
# Định nghĩa hàm test_mild_landmark_noise_and_camera_approximation_keep_five_pose_labels(yaw, pitch, label).
def test_mild_landmark_noise_and_camera_approximation_keep_five_pose_labels(yaw, pitch, label):
    # Gán cho rng: gọi np.random.default_rng(...) với 41; dữ liệu/tham số: 41.
    rng = np.random.default_rng(41)
    # Lặp qua (0.85, 1.15); mỗi lượt gán phần tử cho focal_scale.
    for focal_scale in (.85, 1.15):
        # Lặp qua range(6); mỗi lượt gán phần tử cho _.
        for _ in range(6):
            # Gán cho points: chiếu điểm 3D tổng hợp lên ảnh theo góc đã biết bằng project(...); dữ liệu/tham số: yaw, pitch, focal_scale=focal_scale.
            points = project(yaw, pitch, focal_scale=focal_scale)
            # Cập nhật points: cộng thêm rng.normal(0, 0.35, size=(5, 2)) và lưu lại vào chính nó.
            points += rng.normal(0, .35, size=(5, 2))
            # Gán cho pose: ước lượng ba góc đầu từ điểm mốc và kích thước ảnh bằng estimate_head_pose(...); dữ liệu/tham số: points, SHAPE.
            pose = estimate_head_pose(points, SHAPE)
            # Kiểm chứng điều kiện pose is not None; sai thì test dừng bằng AssertionError.
            assert pose is not None
            # Kiểm chứng điều kiện abs(pose.yaw - yaw) < 5; sai thì test dừng bằng AssertionError.
            assert abs(pose.yaw - yaw) < 5
            # Kiểm chứng điều kiện abs(pose.pitch - pitch) < 5; sai thì test dừng bằng AssertionError.
            assert abs(pose.pitch - pitch) < 5
            # Kiểm chứng điều kiện classify_pose(pose) == label; sai thì test dừng bằng AssertionError.
            assert classify_pose(pose) == label


# Kiểm tra: Khác biệt nhẹ tỷ lệ mặt không biến góc quay trái thành nhìn thẳng.
# Định nghĩa hàm test_mild_independent_face_proportion_change_does_not_force_frontal_turn().
def test_mild_independent_face_proportion_change_does_not_force_frontal_turn():
    # Gán cho face: tạo bản sao để thay đổi không tác động dữ liệu gốc qua FACE.copy(...).
    face = FACE.copy()
    # Cập nhật face[2, 2]: nhân với 0.9 và lưu lại vào chính nó.
    face[2, 2] *= .9
    # Cập nhật face[3:, 0]: nhân với 1.05 và lưu lại vào chính nó.
    face[3:, 0] *= 1.05
    # Gán cho pose: ước lượng ba góc đầu từ điểm mốc và kích thước ảnh bằng estimate_head_pose(...); dữ liệu/tham số: project(-30, face=face) (chiếu điểm 3D tổng hợp lên ảnh theo góc đã biết bằng project(...)), SHAPE.
    pose = estimate_head_pose(project(-30, face=face), SHAPE)
    # Kiểm chứng điều kiện pose is not None; sai thì test dừng bằng AssertionError.
    assert pose is not None
    # Kiểm chứng điều kiện abs(pose.yaw + 30) < 6; sai thì test dừng bằng AssertionError.
    assert abs(pose.yaw + 30) < 6
    # Kiểm chứng điều kiện classify_pose(pose) == 'left'; sai thì test dừng bằng AssertionError.
    assert classify_pose(pose) == "left"


# Kiểm tra: Điểm mốc sai dạng, kiểu hoặc giá trị phải trả None.
# Decorator parametrize: chạy cùng một bài test với nhiều bộ dữ liệu liệt kê bên dưới.
@pytest.mark.parametrize("points", [
    # Tiếp phần khai báo nhiều dòng: None, [], np.zeros((4, 2)), np.zeros((1, 5, 2)), np.zeros(10),.
    None, [], np.zeros((4, 2)), np.zeros((1, 5, 2)), np.zeros(10),
    # Tiếp phần khai báo nhiều dòng: np.zeros((5, 2)), np.ones((5, 2), dtype=bool),.
    np.zeros((5, 2)), np.ones((5, 2), dtype=bool),
    # Tiếp phần khai báo nhiều dòng: np.ones((5, 2), dtype=complex), np.full((5, 2), "1"),.
    np.ones((5, 2), dtype=complex), np.full((5, 2), "1"),
    # Tiếp phần khai báo nhiều dòng: np.full((5, 2), np.nan), np.full((5, 2), np.inf),.
    np.full((5, 2), np.nan), np.full((5, 2), np.inf),
    # Tiếp phần khai báo nhiều dòng: np.array([[100, 100], [150, 100], [125, 100], [110, 100], [140, 100]]),.
    np.array([[100, 100], [150, 100], [125, 100], [110, 100], [140, 100]]),
# Tiếp phần khai báo nhiều dòng: ]).
])
# Định nghĩa hàm test_invalid_landmarks_return_none(points).
def test_invalid_landmarks_return_none(points):
    # Kiểm chứng điều kiện estimate_head_pose(points, SHAPE) is None; sai thì test dừng bằng AssertionError.
    assert estimate_head_pose(points, SHAPE) is None
    # Kiểm chứng điều kiện classify_pose(estimate_head_pose(points, SHAPE)) is None; sai thì test dừng bằng AssertionError.
    assert classify_pose(estimate_head_pose(points, SHAPE)) is None


# Kiểm tra: Kích thước ảnh sai định dạng phải trả None.
# Decorator parametrize: chạy cùng một bài test với nhiều bộ dữ liệu liệt kê bên dưới.
@pytest.mark.parametrize("shape", [None, (), (480,), (0, 640), (-1, 640, 3),
                                  # Tiếp phần khai báo nhiều dòng: (480., 640), (True, 640), (480, 640, 0),.
                                  (480., 640), (True, 640), (480, 640, 0),
                                  # Tiếp phần khai báo nhiều dòng: (480, 640, 3, 1), "480x640"]).
                                  (480, 640, 3, 1), "480x640"])
# Định nghĩa hàm test_invalid_image_shape_returns_none(shape).
def test_invalid_image_shape_returns_none(shape):
    # Kiểm chứng điều kiện estimate_head_pose(project(), shape) is None; sai thì test dừng bằng AssertionError.
    assert estimate_head_pose(project(), shape) is None


# Kiểm tra: Điểm bị trùng, đảo thứ tự hoặc lệch hình học bị từ chối.
# Decorator parametrize: chạy cùng một bài test với nhiều bộ dữ liệu liệt kê bên dưới.
@pytest.mark.parametrize("change", [
    # Tiếp phần khai báo nhiều dòng: lambda p: p.__setitem__(0, p[1]),.
    lambda p: p.__setitem__(0, p[1]),
    # Tiếp phần khai báo nhiều dòng: lambda p: p.__setitem__(2, p[3]),.
    lambda p: p.__setitem__(2, p[3]),
    # Tiếp phần khai báo nhiều dòng: lambda p: p.__setitem__(2, [-1, 100]),.
    lambda p: p.__setitem__(2, [-1, 100]),
    # Tiếp phần khai báo nhiều dòng: lambda p: p.__setitem__(2, [640, 240]),.
    lambda p: p.__setitem__(2, [640, 240]),
    # Tiếp phần khai báo nhiều dòng: lambda p: p.__setitem__(2, [320, 480]),.
    lambda p: p.__setitem__(2, [320, 480]),
    # Tiếp phần khai báo nhiều dòng: lambda p: p.__setitem__([0, 1], p[[1, 0]]),.
    lambda p: p.__setitem__([0, 1], p[[1, 0]]),
    # Tiếp phần khai báo nhiều dòng: lambda p: p.__setitem__([3, 4], p[[4, 3]]),.
    lambda p: p.__setitem__([3, 4], p[[4, 3]]),
    # Tiếp phần khai báo nhiều dòng: lambda p: p.__setitem__(2, p[2] + [0, 70]),.
    lambda p: p.__setitem__(2, p[2] + [0, 70]),
    # Tiếp phần khai báo nhiều dòng: lambda p: p.__setitem__(4, p[4] + [0, -35]),.
    lambda p: p.__setitem__(4, p[4] + [0, -35]),
# Tiếp phần khai báo nhiều dòng: ]).
])
# Định nghĩa hàm test_bad_geometry_and_wrong_correspondences_rejected(change).
def test_bad_geometry_and_wrong_correspondences_rejected(change):
    # Gán cho points: chiếu điểm 3D tổng hợp lên ảnh theo góc đã biết bằng project(...).
    points = project()
    # Gọi change(...) với points; dữ liệu/tham số: points.
    change(points)
    # Kiểm chứng điều kiện estimate_head_pose(points, SHAPE) is None; sai thì test dừng bằng AssertionError.
    assert estimate_head_pose(points, SHAPE) is None


# Kiểm tra: Mặt quá nhỏ bị từ chối dù phép chiếu toán học khớp hoàn hảo.
# Định nghĩa hàm test_tiny_face_rejected_even_with_perfect_projection().
def test_tiny_face_rejected_even_with_perfect_projection():
    # Kiểm chứng điều kiện estimate_head_pose(project(translation=(0.0, 0.0, 5000.0)), SHAPE) is None; sai thì test dừng bằng AssertionError.
    assert estimate_head_pose(project(translation=(0., 0., 5000.)), SHAPE) is None


# Kiểm tra: Hình học trông hợp lý nhưng sai số chiếu lớn vẫn bị loại.
# Định nghĩa hàm test_reprojection_gate_rejects_plausible_but_inconsistent_geometry().
def test_reprojection_gate_rejects_plausible_but_inconsistent_geometry():
    # Gán cho points: chiếu điểm 3D tổng hợp lên ảnh theo góc đã biết bằng project(...).
    points = project()
    # Cập nhật points[3]: cộng thêm [14, -8] và lưu lại vào chính nó.
    points[3] += [14, -8]
    # Mắt vẫn tách biệt, mũi giữa mắt/miệng và khóe miệng đúng thứ tự; lỗi nằm ở sai số chiếu.
    # Kiểm chứng điều kiện estimate_head_pose(points, SHAPE) is None; sai thì test dừng bằng AssertionError.
    assert estimate_head_pose(points, SHAPE) is None


# Kiểm tra: Tư thế cúi vẫn được nhận với tỷ lệ mặt biến đổi trong mức cho phép.
# Định nghĩa hàm test_downward_view_accepts_plausible_face_proportions().
def test_downward_view_accepts_plausible_face_proportions():
    # Gán cho face: tạo bản sao để thay đổi không tác động dữ liệu gốc qua FACE.copy(...).
    face = FACE.copy()
    # Cập nhật face[2, 1]: cộng thêm 20 và lưu lại vào chính nó.
    face[2, 1] += 20
    # Cập nhật face[3:, 1]: cộng thêm 10 và lưu lại vào chính nó.
    face[3:, 1] += 10
    # Gán cho pose: ước lượng ba góc đầu từ điểm mốc và kích thước ảnh bằng estimate_head_pose(...); dữ liệu/tham số: project(yaw=8, pitch=20, face=face) (chiếu điểm 3D tổng hợp lên ảnh theo góc đã biết bằng project(...)), SHAPE.
    pose = estimate_head_pose(project(yaw=8, pitch=20, face=face), SHAPE)
    # Kiểm chứng điều kiện classify_pose(pose) == 'down'; sai thì test dừng bằng AssertionError.
    assert classify_pose(pose) == "down"


# Kiểm tra: Góc quay hoặc ngẩng/cúi quá lớn bị từ chối.
# Decorator parametrize: chạy cùng một bài test với nhiều bộ dữ liệu liệt kê bên dưới.
@pytest.mark.parametrize("yaw,pitch", [(72, 0), (0, 55), (0, -55)])
# Định nghĩa hàm test_extreme_views_rejected(yaw, pitch).
def test_extreme_views_rejected(yaw, pitch):
    # Kiểm chứng điều kiện estimate_head_pose(project(yaw, pitch), SHAPE) is None; sai thì test dừng bằng AssertionError.
    assert estimate_head_pose(project(yaw, pitch), SHAPE) is None


# Kiểm tra: Lật ảnh mà không sửa thứ tự điểm mốc phải bị từ chối.
# Định nghĩa hàm test_mirrored_pixels_with_unchanged_correspondences_are_rejected().
def test_mirrored_pixels_with_unchanged_correspondences_are_rejected():
    # Gán cho points: chiếu điểm 3D tổng hợp lên ảnh theo góc đã biết bằng project(...); dữ liệu/tham số: 28.
    points = project(28)
    # Gán cho points[:, 0]: trừ trong SHAPE[1] - 1 - points[:, 0].
    points[:, 0] = SHAPE[1] - 1 - points[:, 0]
    # Kiểm chứng điều kiện estimate_head_pose(points, SHAPE) is None; sai thì test dừng bằng AssertionError.
    assert estimate_head_pose(points, SHAPE) is None


# Kiểm tra: PnP thất bại phải trả None, không mặc định coi là nhìn thẳng.
# Định nghĩa hàm test_solver_failure_does_not_become_front().
def test_solver_failure_does_not_become_front():
    # Mở khối with để thay tạm đối tượng theo đường dẫn import trong bài kiểm thử bằng patch(...); dữ liệu/tham số: 'application.head_pose.cv2.solvePnPGeneric', side_effect=cv2.error('failed'); tự kết thúc ngữ cảnh/giải phóng tài nguyên khi ra khỏi khối.
    with patch("application.head_pose.cv2.solvePnPGeneric", side_effect=cv2.error("failed")):
        # Kiểm chứng điều kiện estimate_head_pose(project(), SHAPE) is None; sai thì test dừng bằng AssertionError.
        assert estimate_head_pose(project(), SHAPE) is None


# Kiểm tra: Kiểm tra biên ngưỡng, khoảng trống giữa nhãn và giá trị góc không hữu hạn.
# Decorator parametrize: chạy cùng một bài test với nhiều bộ dữ liệu liệt kê bên dưới.
@pytest.mark.parametrize("pose,label", [
    # Tiếp phần khai báo nhiều dòng: (HeadPose(16, -10, 15), "front"),.
    (HeadPose(16, -10, 15), "front"),
    # Tiếp phần khai báo nhiều dòng: (HeadPose(-14.85, 6.1, -5), "front"),.
    (HeadPose(-14.85, 6.1, -5), "front"),
    # Tiếp phần khai báo nhiều dòng: (HeadPose(-18, 12, -15), "left"), (HeadPose(50, -12, 15), "right"),.
    (HeadPose(-18, 12, -15), "left"), (HeadPose(50, -12, 15), "right"),
    # Tiếp phần khai báo nhiều dòng: (HeadPose(-12, -15, 15), "up"), (HeadPose(12, 35, -15), "down"),.
    (HeadPose(-12, -15, 15), "up"), (HeadPose(12, 35, -15), "down"),
    # Tiếp phần khai báo nhiều dòng: (HeadPose(14, 16, 0), "down"), (HeadPose(15, 16, 0), None),.
    (HeadPose(14, 16, 0), "down"), (HeadPose(15, 16, 0), None),
    # Tiếp phần khai báo nhiều dòng: (HeadPose(13, -16, 0), None),.
    (HeadPose(13, -16, 0), None),
    # Tiếp phần khai báo nhiều dòng: (HeadPose(17, 0, 0), None), (HeadPose(0, 12, 0), None),.
    (HeadPose(17, 0, 0), None), (HeadPose(0, 12, 0), None),
    # Tiếp phần khai báo nhiều dòng: (HeadPose(28, 20, 0), None), (HeadPose(51, 0, 0), None),.
    (HeadPose(28, 20, 0), None), (HeadPose(51, 0, 0), None),
    # Tiếp phần khai báo nhiều dòng: (HeadPose(0, 36, 0), None), (HeadPose(0, 0, 16), None),.
    (HeadPose(0, 36, 0), None), (HeadPose(0, 0, 16), None),
    # Tiếp phần khai báo nhiều dòng: (HeadPose(float("nan"), 0, 0), None), (HeadPose(0, float("inf"), 0), None),.
    (HeadPose(float("nan"), 0, 0), None), (HeadPose(0, float("inf"), 0), None),
    # Tiếp phần khai báo nhiều dòng: (HeadPose(0, 0, float("nan")), None), (None, None),.
    (HeadPose(0, 0, float("nan")), None), (None, None),
# Tiếp phần khai báo nhiều dòng: ]).
])
# Định nghĩa hàm test_classification_thresholds_gaps_and_invalid_angles(pose, label).
def test_classification_thresholds_gaps_and_invalid_angles(pose, label):
    # Kiểm chứng điều kiện classify_pose(pose) == label; sai thì test dừng bằng AssertionError.
    assert classify_pose(pose) == label
