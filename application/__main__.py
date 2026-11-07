import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import urllib.request

from .assets import verify_model
from .config import MODEL_PATH, MODEL_SIZE, MODEL_URL, Settings
from .errors import ApplicationError


def restore_model():
    if MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as stream:
            is_pointer = stream.read(200).startswith(b"version https://git-lfs.github.com/spec/v1")
        if not is_pointer:
            return verify_model()  # Never overwrite a real but different model.
    # Keep the temporary download if verification fails, for diagnosis.
    with tempfile.NamedTemporaryFile(prefix="facenet_", suffix=".download", dir=MODEL_PATH.parent, delete=False) as temporary:
        download = Path(temporary.name)
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            size = 0
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > MODEL_SIZE:
                    raise ValueError(f"Artifact exceeds expected size; pointer preserved. Download: {download}")
                temporary.write(block)
    result = verify_model(download)
    if MODEL_PATH.exists():
        backup = MODEL_PATH.with_suffix(".onnx.lfs-pointer")
        if not backup.exists():
            with backup.open("xb") as output:
                output.write(MODEL_PATH.read_bytes())
    download.replace(MODEL_PATH)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Face Recognition application (run from repository root)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-assets")
    sub.add_parser("restore-model")
    initialize = sub.add_parser("init-registry")
    initialize.add_argument("--registry", type=Path)
    serve = sub.add_parser("serve")
    serve.add_argument("--registry", type=Path)
    serve.add_argument("--api-only", action="store_true")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.command == "check-assets":
            from .service import load_runtime
            runtime = load_runtime(Settings.from_env())
            result = {**verify_model(), "models": "loaded", "runtime": json.loads(runtime.version)["runtime"]}
        elif args.command == "restore-model":
            result = restore_model()
        else:
            from .service import ApplicationService
            settings = Settings.from_env()
            if args.registry:
                settings = replace(settings, registry=args.registry.absolute())
            if args.command == "init-registry":
                from FaceNet import initialize_enrollment_registry
                service = ApplicationService(settings)
                with service.guard():
                    initialize_enrollment_registry(service.path)
                result = {"status": "created", "registry": str(service.path)}
            else:
                import uvicorn
                from .api import create_app
                settings = replace(settings, cpu_threads=args.cpu_threads)
                app = create_app(ApplicationService(settings), with_ui=not args.api_only)
                uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False)
                return 0
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ApplicationError as error:
        print(json.dumps(error.payload(), ensure_ascii=False), file=sys.stderr)
    except (ValueError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
