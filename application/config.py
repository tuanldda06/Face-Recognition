from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "FaceNet/models/facenet.onnx"
MODEL_SHA256 = "f6688ba6c95465fac18df53fdf8e094048fa3f1dd2c09f1cc6dd734a1c8dedf8"
MODEL_SIZE = 93987560
MODEL_URL = "https://media.axelera.ai/artifacts/model_cards/weights/face_recognition/facenet.onnx"


@dataclass(frozen=True)
class Settings:
    registry: Path = ROOT / "application/data/registry.sqlite"
    max_file_bytes: int = 8 * 1024 * 1024
    max_request_bytes: int = 32 * 1024 * 1024
    max_images: int = 10
    max_pixels: int = 8_000_000
    max_dimension: int = 4096
    max_faces: int = 20
    cpu_threads: int = 2
    webcam_width: int = 640
    lock_timeout: float = 30

    @classmethod
    def from_env(cls):
        return cls(
            registry=Path(os.environ.get("FACE_REGISTRY", str(cls.registry))).absolute(),
            cpu_threads=int(os.environ.get("FACE_CPU_THREADS", "2")),
        )

    def __post_init__(self):
        if not self.registry.is_absolute():
            raise ValueError("Registry path must be absolute.")
        for value in (self.max_file_bytes, self.max_request_bytes, self.max_images,
                      self.max_pixels, self.max_dimension, self.max_faces,
                      self.cpu_threads, self.webcam_width, self.lock_timeout):
            if value <= 0:
                raise ValueError("Application limits must be positive.")
