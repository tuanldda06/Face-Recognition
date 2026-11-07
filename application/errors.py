class ApplicationError(Exception):
    """A safe public error, separate from exceptions raised by model inference."""

    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status

    def payload(self):
        return {"error": {"code": self.code, "message": self.message}}
