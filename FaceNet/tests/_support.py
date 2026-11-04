"""Hỗ trợ tests: chỉ đọc notebook và lấy đúng các định nghĩa đã chọn vào RAM."""

import ast
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path("D:/Facenet-Recognition")  # Read-only notebook/data oracle.
SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def notebook_namespace(filename, function_names, constant_names=()):
    """Không chạy cell top-level, load model, ghi ảnh hoặc mở SQLite của notebook."""
    path = PROJECT_ROOT / "notebooks" / filename
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"np": np, "cv2": cv2, "Path": Path}
    found = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            selected = isinstance(node, ast.FunctionDef) and node.name in function_names
            if isinstance(node, ast.Assign):
                selected = any(
                    isinstance(target, ast.Name) and target.id in constant_names
                    for target in node.targets
                )
            if selected:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
                if isinstance(node, ast.FunctionDef):
                    found.add(node.name)
    if found != set(function_names):
        raise AssertionError(f"Thiếu định nghĩa nguồn: {set(function_names) - found}")
    return namespace


def function_body(source, name):
    """Đối chiếu thân hàm, không tính docstring/type hints hoặc định dạng."""
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def notebook_function_body(filename, name):
    notebook = json.loads((PROJECT_ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if any(isinstance(n, ast.FunctionDef) and n.name == name for n in ast.parse(source).body):
            return function_body(source, name)
    raise AssertionError(f"Không tìm thấy {name} trong {filename}")


def protected_hashes():
    """Băm asset/tài liệu gốc; không ghi snapshot vào ổ đĩa."""
    files = [PROJECT_ROOT / name for name in ("README.md", "requirements.txt", ".gitignore")]
    for name in ("models", "data", "notebooks"):
        files.extend(p for p in (PROJECT_ROOT / name).rglob("*") if p.is_file())
    result = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result.update({"runtime/" + str(p.relative_to(SOURCE_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (SOURCE_ROOT / "FaceNet").rglob("*") if p.is_file()})
    return result
