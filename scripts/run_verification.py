
import argparse
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [Path("D:/MTCNN").resolve(),Path("D:/Facenet-Recognition").resolve()]
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"
TEMP_ROOT = (ROOT / ".work/tmp").resolve()
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(TEMP_ROOT)
os.environ["TEMP"] = str(TEMP_ROOT)
os.environ["TMP"] = str(TEMP_ROOT)
sqlite_paths = set()


def path_in_source(value):
    if not isinstance(value,(str,bytes,os.PathLike)):
        return False
    path = Path(os.fsdecode(value)).resolve()
    return any(path.is_relative_to(source) for source in SOURCES)


def guard(event,args):
    if event == "open":
        mode,flags = args[1:3]
        writing = ((isinstance(mode,str) and any(c in mode for c in "wax+")) or
                   (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if writing and path_in_source(args[0]):
            raise AssertionError("Cấm ghi project nguồn")
    if event in ("os.mkdir","os.remove","os.rmdir","os.rename"):
        if any(path_in_source(value) for value in args[:2]):
            raise AssertionError("Cấm sửa filesystem project nguồn")
    if event == "socket.connect":
        raise AssertionError("Không dùng mạng khi kiểm thử model")
    if event == "sqlite3.connect":
        value = str(args[0])
        if any(str(root).lower().replace("\\","/") in value.lower().replace("\\","/") for root in SOURCES):
            if "?mode=ro" not in value:
                raise AssertionError("SQLite nguồn chỉ được mở read-only")
        # Tests Phase 3 chỉ tạo DB dưới vùng tạm riêng, đã có cleanup guard.
        if not value.startswith("file:"):
            path = Path(value).resolve()
            if not path.is_relative_to(TEMP_ROOT):
                raise AssertionError("SQLite ghi chỉ được ở thư mục tạm")
            sqlite_paths.add(str(path))
        elif "?mode=ro" not in value:
            from urllib.parse import unquote, urlsplit
            raw = unquote(urlsplit(value).path).lstrip("/")
            path = Path(raw).resolve()
            if not path.is_relative_to(TEMP_ROOT):
                raise AssertionError("SQLite ghi URI chỉ được ở thư mục tạm")
            sqlite_paths.add(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite",choices=["facenet","mtcnn","adapter","embedding"])
    args = parser.parse_args()
    sys.addaudithook(guard)
    if args.suite == "facenet":
        # Test gốc dùng fixtures/source gốc, chỉ đọc; runtime copies được hash riêng.
        sys.path.insert(0,"D:/Facenet-Recognition/tests")
        suite = unittest.TestLoader().discover("D:/Facenet-Recognition/tests",pattern="test_*.py")
        expected = 115
    elif args.suite == "mtcnn":
        import pytest
        code = pytest.main(["-q","-p","no:cacheprovider","-c",
                            "D:/MTCNN/MTCNN/pytest.ini","D:/MTCNN/MTCNN/tests"])
        return int(code)
    else:
        sys.path[:0] = [str(ROOT),str(ROOT/"src")]
        pattern = "test_mtcnn_adapter.py" if args.suite=="adapter" else "test_end_to_end_embedding.py"
        suite = unittest.TestLoader().discover(str(ROOT/"tests"),pattern=pattern)
        expected = 26 if args.suite=="adapter" else 3
    if suite.countTestCases()!=expected:
        raise AssertionError(f"Test count {suite.countTestCases()} != {expected}")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    cleanup = all(not Path(path).exists() for path in sqlite_paths)
    print("SUITE_RESULT "+json.dumps({
        "suite":args.suite,"tests":result.testsRun,"failures":len(result.failures),
        "errors":len(result.errors),"skipped":len(result.skipped),
        "sqlite_temporary_paths":sorted(sqlite_paths),"sqlite_cleanup":cleanup,
        "python":sys.executable,
    },sort_keys=True))
    if not cleanup:
        raise AssertionError("Database tạm chưa cleanup")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
