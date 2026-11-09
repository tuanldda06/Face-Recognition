# Lỗi có cấu trúc dùng chung cho API và giao diện: mã lỗi, thông báo và trạng thái HTTP.
# Mang thông tin lỗi nghiệp vụ để API và giao diện hiển thị thống nhất.
# Định nghĩa lớp ApplicationError kế thừa Exception; lớp gom dữ liệu và các hàm liên quan.
class ApplicationError(Exception):
    # Khởi tạo đối tượng và giữ các phụ thuộc/trạng thái cần dùng trong những lần gọi sau.
    # Định nghĩa hàm __init__(self, code, message, status). self là đối tượng đang gọi phương thức. Mặc định: status=422.
    def __init__(self, code: str, message: str, status: int = 422):
        # Gọi super().__init__(...) với message; dữ liệu/tham số: message.
        super().__init__(message)
        # Tách kết quả theo thứ tự và gán cho (self.code, self.message, self.status): bộ giá trị (code, message, status) (tuple giữ thứ tự, không gán lại từng phần tử).
        self.code, self.message, self.status = code, message, status

    # Đưa lỗi về dict có thể chuyển thành JSON khi trả cho phía gọi.
    # Định nghĩa hàm payload(self). self là đối tượng đang gọi phương thức.
    def payload(self):
        # Kết thúc hàm và trả từ điển {'error': {'code': self.code, 'message': self.message}} (ánh xạ khóa → giá trị).
        return {"error": {"code": self.code, "message": self.message}}
