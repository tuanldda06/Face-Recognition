"""Chạy tests của bản final; mọi dữ liệu ghi chỉ nằm trong vùng tạm."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=["all", "facenet", "integration", "mtcnn", "adapter", "embedding"])
    args = parser.parse_args()
    parent = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix="recognition_verification_", dir=parent) as owned:
        temporary = Path(owned).resolve()
        assert temporary.parent == parent and temporary.name.startswith("recognition_verification_")
        os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                          PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", TEMP=str(temporary),
                          TMP=str(temporary), CUDA_VISIBLE_DEVICES="")
        tempfile.tempdir = str(temporary)
        sqlite_writes = set()
        def writable(value):
            return isinstance(value, (str, bytes, os.PathLike)) and Path(os.fsdecode(value)).resolve().is_relative_to(temporary)
        def guard(event, values):
            if event == "open":
                path, mode, flags = values
                writing = ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                           or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
                if writing and not isinstance(path, int) and os.fsdecode(path).lower() not in ("nul", "nul:") and not writable(path):
                    raise PermissionError("Tests may write only their owned temporary directory.")
            if event in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod"):
                if not writable(values[0]):
                    raise PermissionError("Filesystem change outside test temporary directory.")
            if event in ("os.rename", "os.link", "os.symlink"):
                if not all(writable(p) for p in values[:2]):
                    raise PermissionError("Filesystem change outside test temporary directory.")
            if event in ("socket.connect", "socket.bind"):
                raise PermissionError("Network disabled during verification.")
            if event == "sqlite3.connect":
                value = str(values[0])
                if value == ":memory:":
                    return
                uri = value.startswith("file:")
                path = unquote(urlsplit(value).path).lstrip("/") if uri else value
                if not (uri and "mode=ro" in value):
                    if not writable(path):
                        raise PermissionError("No writes to real/source SQLite registries.")
                    sqlite_writes.add(str(Path(path).resolve()))
        sys.addaudithook(guard)
        def unit_suite(name, directory, pattern, expected):
            suite = unittest.TestLoader().discover(str(directory), pattern=pattern)
            if suite.countTestCases() != expected:
                raise AssertionError(f"{name}: {suite.countTestCases()} != {expected}")
            run = unittest.TextTestRunner(verbosity=2).run(suite)
            summary = {"suite": name, "tests": run.testsRun, "failures": len(run.failures),
                       "errors": len(run.errors), "skipped": len(run.skipped)}
            print("SUITE_RESULT " + json.dumps(summary), flush=True)
            if not run.wasSuccessful() or run.skipped:
                raise SystemExit(1)
        selected = ["facenet", "integration", "mtcnn"] if args.suite == "all" else [args.suite]
        for name in selected:
            if name == "facenet":
                unit_suite(name, ROOT / "FaceNet/tests", "test_*.py", 115)
            elif name == "integration":
                unit_suite(name, ROOT / "integration/tests", "test_*.py", 130)
            elif name in ("adapter", "embedding"):
                pattern = "test_mtcnn_adapter.py" if name == "adapter" else "test_end_to_end_embedding.py"
                unit_suite(name, ROOT / "integration/tests", pattern, 26 if name == "adapter" else 3)
            else:
                # Test fixtures remain read-only at the original MTCNN source.
                import MTCNN.mtcnn
                import pytest
                class Results:
                    passed = failed = skipped = 0
                    def pytest_runtest_logreport(self, report):
                        if report.when == "call" and report.passed:
                            self.passed += 1
                        if report.failed:
                            self.failed += 1
                        if report.skipped:
                            self.skipped += 1
                results = Results()
                code = pytest.main(["-q", "-p", "no:cacheprovider", "-c",
                    "D:/MTCNN/MTCNN/pytest.ini", "-o", "log_file=" + (temporary / "pytest.log").as_posix(),
                    "--basetemp", str(temporary / "pytest"), "D:/MTCNN/MTCNN/tests"],
                    plugins=[results])
                print("SUITE_RESULT " + json.dumps({"suite": name, "tests": results.passed,
                    "failures": results.failed, "skipped": results.skipped}), flush=True)
                if code != 0 or results.passed != 30 or results.failed or results.skipped:
                    raise SystemExit(1)
        if any(Path(path).exists() for path in sqlite_writes):
            raise AssertionError("Temporary SQLite not disposed by tests.")
    assert not Path(owned).exists()
    print("TEMP_CLEANUP PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
