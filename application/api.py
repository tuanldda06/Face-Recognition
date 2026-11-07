from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from .errors import ApplicationError
from .images import decode_image
from .service import ApplicationService


class BodyLimitMiddleware:
    """Bound multipart body before parsing, including chunked requests."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not (
            scope["path"] == "/recognize" or scope["path"].endswith("/upload")
            or scope["path"].startswith("/enrollment-sessions")
        ):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = self.limit + 1
        body = bytearray()
        if length <= self.limit:
            while True:
                event = await receive()
                if event["type"] == "http.disconnect":
                    return
                body.extend(event.get("body", b""))
                if len(body) > self.limit or not event.get("more_body", False):
                    break
        if length > self.limit or len(body) > self.limit:
            response = JSONResponse(ApplicationError("request_too_large", "Tổng upload tối đa 32 MiB.", 413).payload(), 413)
            return await response(scope, receive, send)
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


class EnrollmentStart(BaseModel):
    person_id: str = Field(max_length=100)
    display_name: str = Field(max_length=150)


class EnrollmentConfirmation(BaseModel):
    confirmed: bool = False
    revision: int


def create_app(service=None, *, with_ui=False):
    service = service or ApplicationService()

    @asynccontextmanager
    async def lifespan(app):
        await run_in_threadpool(service.start)
        yield

    app = FastAPI(title="MTCNN + FaceNet Application", version="1.0.0", lifespan=lifespan)
    app.state.service = service
    app.add_middleware(BodyLimitMiddleware, limit=service.settings.max_request_bytes)

    @app.exception_handler(ApplicationError)
    async def application_error(request, error):
        return JSONResponse(error.payload(), status_code=error.status)

    @app.get("/health")
    def health():
        result = service.health()
        return JSONResponse(result, status_code=200 if result["status"] == "ok" else 503)

    @app.get("/persons")
    def persons():
        return {"persons": service.persons()}

    @app.delete("/persons/{person_id}")
    def delete(person_id: str):
        return service.delete(person_id)

    async def read_image(upload):
        data = await upload.read(service.settings.max_file_bytes + 1)
        return await run_in_threadpool(decode_image, data, service.settings)

    enrollment = service.webcam_enrollment

    @app.post("/enrollment-sessions", status_code=201)
    def start_enrollment(body: EnrollmentStart):
        return enrollment.start(body.person_id, body.display_name)

    @app.get("/enrollment-sessions/{token}")
    def enrollment_status(token: str):
        return enrollment.status(token)

    @app.post("/enrollment-sessions/{token}/frames")
    async def enrollment_frame(token: str, revision: Annotated[int, Form()], image: Annotated[UploadFile, File()]):
        try:
            try:
                rgb = await read_image(image)
            except ApplicationError:
                # An invalid frame must break a previously stable run as well.
                await run_in_threadpool(enrollment.frame, token, None, revision)
                raise
            return await run_in_threadpool(enrollment.frame, token, rgb, revision)
        finally:
            await image.close()

    @app.get("/enrollment-sessions/{token}/review")
    def enrollment_review(token: str):
        import base64
        from io import BytesIO
        from PIL import Image
        items = []
        snapshot = enrollment.review_snapshot(token)
        for pose, image in snapshot["images"]:
            data = BytesIO()
            Image.fromarray(image).save(data, format="JPEG")
            items.append({"pose": pose, "image": "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode("ascii")})
        return {"revision": snapshot["revision"], "images": items}

    @app.post("/enrollment-sessions/{token}/retake/{pose}")
    def retake_enrollment(token: str, pose: str):
        return enrollment.retake(token, pose)

    @app.post("/enrollment-sessions/{token}/pause")
    def pause_enrollment(token: str):
        return enrollment.pause(token)

    @app.post("/enrollment-sessions/{token}/resume")
    def resume_enrollment(token: str):
        return enrollment.resume(token)

    @app.delete("/enrollment-sessions/{token}")
    def cancel_enrollment(token: str):
        return enrollment.cancel(token)

    @app.post("/enrollment-sessions/{token}/confirm", status_code=201)
    def confirm_enrollment(token: str, body: EnrollmentConfirmation):
        return enrollment.confirm(token, body.confirmed, body.revision)

    @app.post("/recognize")
    async def recognize(image: Annotated[UploadFile, File()]):
        try:
            rgb = await read_image(image)
            return await run_in_threadpool(service.recognize, rgb)
        finally:
            await image.close()

    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse("/ui/" if with_ui else "/docs")

    if with_ui:
        import gradio as gr
        from .ui import build_ui, UI_CSS, UI_THEME
        app = gr.mount_gradio_app(app, build_ui(service), path="/ui",
                                 css=UI_CSS, theme=UI_THEME,
                                 max_file_size=service.settings.max_file_bytes,
                                 show_error=False, run_history=False,
                                 footer_links=[], enable_monitoring=False)
    return app


def create_demo_app():
    return create_app(with_ui=True)
