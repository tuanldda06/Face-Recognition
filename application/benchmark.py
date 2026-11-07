"""Standalone benchmark of the actual service/core, never run inside the web server.

Temporary wrappers time existing functions without changing source or calculations.
Stage sums exclude fingerprint/registry/adapter/serialization overhead; E2E includes it.
"""
import argparse
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
from time import perf_counter
from unittest.mock import patch
import uuid

import numpy as np

from FaceNet import facenet_model, recognition
from integration import recognition_pipeline
from .config import ROOT, Settings
from .errors import ApplicationError
from .images import decode_image
from .service import ApplicationService
from .visualization import annotate


def summarize(values):
    if not values:
        return {"status": "not_measured", "mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {"status": "measured", "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95))}


class TimedDetector:
    def __init__(self, detector, timings):
        self.detector, self.timings = detector, timings

    def __getattr__(self, name):
        return getattr(self.detector, name)

    def detect(self, *args, **kwargs):
        return timed(self.timings, "detection", self.detector.detect)(*args, **kwargs)

    def extract_faces_rgb(self, *args, **kwargs):
        return timed(self.timings, "alignment", self.detector.extract_faces_rgb)(*args, **kwargs)


def timed(timings, name, function):
    def call(*args, **kwargs):
        started = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            timings[name] = timings.get(name, 0.0) + (perf_counter() - started) * 1000
    return call


def run_case(service, path, repetitions, warmup):
    data = path.read_bytes()
    image = decode_image(data, service.settings)
    timings = {}
    original_detector = service.runtime.detector
    measured = {key: [] for key in ("detection", "alignment", "facenet_preprocessing", "embedding",
                                    "matching", "end_to_end")}
    counts, samples = [], []
    with ExitStack() as stack:
        service.runtime.detector = TimedDetector(original_detector, timings)
        stack.callback(setattr, service.runtime, "detector", original_detector)
        extraction = timed(timings, "embedding_including_preprocessing", facenet_model.extract_embedding)
        stack.enter_context(patch.object(recognition, "extract_embedding", extraction))
        stack.enter_context(patch.object(recognition_pipeline, "extract_embedding", extraction))
        stack.enter_context(patch.object(facenet_model, "preprocess_face",
                                        timed(timings, "facenet_preprocessing", facenet_model.preprocess_face)))
        stack.enter_context(patch.object(recognition, "detect_unknown",
                                        timed(timings, "matching", recognition.detect_unknown)))
        for index in range(warmup + repetitions):
            timings.clear()
            started = perf_counter()
            result = service.recognize(image)
            annotate(image, result["faces"])
            timings["end_to_end"] = (perf_counter() - started) * 1000
            if "embedding_including_preprocessing" in timings:
                timings["embedding"] = timings["embedding_including_preprocessing"] - timings.get("facenet_preprocessing", 0)
            if index >= warmup:
                counts.append(result["face_count"])
                samples.append(dict(timings))
                for name in measured:
                    if name in timings:
                        measured[name].append(timings[name])
    return {"status": "measured", "image": path.name,
            "image_sha256": hashlib.sha256(data).hexdigest(), "dimensions": [image.shape[1], image.shape[0]],
            "detected_faces_each_run": counts, "repetitions": repetitions, "warmup": warmup,
            "latency": {name: summarize(values) for name, values in measured.items()},
            "fps": repetitions / (sum(measured["end_to_end"]) / 1000), "samples_ms": samples}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", default=[
        ROOT / "integration/tests/fixtures/mtcnn_single.png",
        ROOT / "application/tests/fixtures/group_two_faces.png",
        ROOT / "integration/tests/fixtures/mtcnn_multi.jpg"])
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--temporary-gallery", action="store_true", help="Create a disposable gallery using the bundled single face; never modify a user registry")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.warmup < 1 or (args.temporary_gallery and args.registry):
        parser.error("repeat/warmup >= 1; choose either --registry or --temporary-gallery")
    report = {"schema": "application-benchmark/1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "device": "CPU", "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "not_measured"),
              "logical_cpus": os.cpu_count(),
              "os": platform.platform(), "python": platform.python_version(), "cpu_threads": args.cpu_threads,
              "runtime": {name: importlib.metadata.version(name) for name in
                          ("numpy", "opencv-python", "Pillow", "torch", "fastapi", "gradio")},
              "scope": "Service recognition + visualization; excludes model load, image decode, HTTP and browser transport. Serial warm-cache calls.",
              "stage_definitions": {"embedding": "extract_embedding minus preprocess_face; includes forward, validation and L2",
                                    "matching": "existing detect_unknown, cosine/ranking/threshold",
                                    "alignment": "existing extract_faces_rgb align=True"},
              "accuracy_evaluation": False, "cases": []}
    with ExitStack() as stack:
        settings = replace(Settings.from_env(), cpu_threads=args.cpu_threads)
        if args.registry:
            settings = replace(settings, registry=args.registry.absolute())
        if args.temporary_gallery:
            from FaceNet import initialize_enrollment_registry
            folder = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="face_benchmark_")))
            settings = replace(settings, registry=folder / "registry.sqlite")
            initialize_enrollment_registry(settings.registry)
        service = ApplicationService(settings)
        service.start()
        try:
            service._require_model()
            if args.temporary_gallery:
                sample = ROOT / "integration/tests/fixtures/mtcnn_single.png"
                service.enroll("benchmark_fixture", [decode_image(sample.read_bytes(), settings)])
            report["gallery_size"] = len(service.persons())
            report["model_version_sha256"] = hashlib.sha256(service.runtime.version.encode()).hexdigest()
            for path in args.images:
                try:
                    report["cases"].append(run_case(service, path, args.repeat, args.warmup))
                except (ApplicationError, OSError) as error:
                    report["cases"].append({"image": path.name, "status": "blocked", "reason": str(error)})
            report["status"] = "measured" if all(c["status"] == "measured" for c in report["cases"]) else "partial"
        except ApplicationError as error:
            report.update(status="blocked", reason=error.message, error_code=error.code)
    seen = {count for case in report["cases"] for count in case.get("detected_faces_each_run", [])}
    report["face_count_coverage"] = {str(n): "measured" if n in seen else "blocked: no verified input with this face count" for n in (1, 2, 5)}
    output = args.output or ROOT / "application/reports" / ("benchmark_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8] + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f'{report["status"]}: {output}')
    return 0 if report["status"] == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
