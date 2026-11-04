"""FaceNet: implementation local; import không nạp model hoặc dữ liệu."""

from importlib import import_module

_MODULES = ("contracts", "facenet_model", "matching", "database", "enrollment", "recognition")
_APIS = {
    "load_facenet_model": (".facenet_model", "load_facenet_model"),
    "build_model_version": ("integration.enrollment_pipeline", "build_model_version"),
    "initialize_enrollment_registry": ("integration.enrollment_pipeline", "initialize_enrollment_registry"),
    "enroll_person_from_images": ("integration.enrollment_pipeline", "enroll_person_from_images"),
    "recognize_image": ("integration.recognition_runtime", "recognize_image"),
}
__all__ = [*_MODULES, *_APIS]


def __getattr__(name):
    if name in _MODULES:
        value = import_module(f".{name}", __name__)
    elif name in _APIS:
        module, attribute = _APIS[name]
        value = getattr(import_module(module, __name__), attribute)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
