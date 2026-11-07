"""Stateless per-frame processor; stream timing state belongs to each UI session."""
from time import perf_counter

import numpy as np
from PIL import Image

from .errors import ApplicationError
from .images import validate_rgb
from .visualization import annotate


def process_frame(service, frame, state=None):
    started = perf_counter()
    if frame is None:
        raise ApplicationError("camera_unavailable", "Chưa có frame. Cho phép camera trong trình duyệt và dùng localhost hoặc HTTPS.")
    validate_rgb(frame, service.settings)
    height, width = frame.shape[:2]
    if width > service.settings.webcam_width:
        target = service.settings.webcam_width
        frame = np.asarray(Image.fromarray(frame).resize((target, max(20, round(height * target / width))), Image.Resampling.BILINEAR))
    result = service.recognize(frame)
    annotated = annotate(frame, result["faces"])
    finished = perf_counter()
    state = dict(state or {})
    state.setdefault("first_received", started)
    state["frames"] = state.get("frames", 0) + 1
    elapsed = finished - started
    result["frame_latency_ms"] = elapsed * 1000
    result["processing_fps"] = 1 / elapsed
    result["stream_fps"] = state["frames"] / (finished - state["first_received"])
    result["frames_processed"] = state["frames"]
    return annotated, result, state
