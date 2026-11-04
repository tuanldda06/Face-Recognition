"""Cosine, ranking và Known/Unknown giữ nguyên từ notebook 07–10.

Chỉ nhận embedding đã trích xuất, không đọc SQLite/ảnh hoặc nạp model. Threshold
luôn do bên gọi truyền; không có ngưỡng mặc định hoặc calibration trong module.
Các validator giữ tại đây để không thay đổi contracts.py đã duyệt ở Phase 1.
"""

from __future__ import annotations

import numpy as np


EMBEDDING_SIZE = 512


def validate_person_id(person_id: str) -> None:
    """Giữ quy tắc ID của notebook: chuỗi không rỗng, không có khoảng trắng hai đầu."""
    if not isinstance(person_id, str):
        raise TypeError("person_id phải là chuỗi.")
    if not person_id.strip():
        raise ValueError("person_id không được rỗng hoặc chỉ chứa khoảng trắng.")
    if person_id != person_id.strip():
        raise ValueError("person_id không được có khoảng trắng ở đầu/cuối.")


def validate_database_embedding(embedding: np.ndarray) -> None:
    """Kiểm tra float32 (512,), hữu hạn, norm gần 1; không tự chuẩn hóa.

    Giữ tên hàm cũ để đối chiếu notebook; hàm này không truy cập database.
    """
    if not isinstance(embedding, np.ndarray):
        raise TypeError("Embedding phải là numpy.ndarray.")
    if embedding.shape != (EMBEDDING_SIZE,):
        raise ValueError("Embedding phải có shape (512,).")
    if embedding.dtype != np.float32:
        raise TypeError("Embedding phải có dtype float32.")
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding chứa NaN hoặc Inf.")
    norm = float(np.linalg.norm(embedding.astype(np.float64)))
    if not np.isclose(norm, 1.0, atol=1e-6, rtol=0):
        raise ValueError("Embedding phải có L2 norm xấp xỉ 1.")


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Trả cosine [-1,1], tính float64 và clip sai số nhỏ như notebook 07."""
    validate_database_embedding(embedding_a)
    validate_database_embedding(embedding_b)
    a = embedding_a.astype(np.float64)
    b = embedding_b.astype(np.float64)
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if min(norm_a, norm_b) <= 1e-12:
        raise ValueError("Cosine không xác định cho vector có norm bằng 0.")
    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    if not np.isfinite(similarity) or not -1.0 - 1e-12 <= similarity <= 1.0 + 1e-12:
        raise ValueError("Cosine không hữu hạn hoặc vượt khoảng hợp lệ.")
    return float(np.clip(similarity, -1.0, 1.0))


def match_embedding(
    query_embedding: np.ndarray, registered_embeddings: dict[str, np.ndarray]
) -> list[dict[str, str | float | int]]:
    """Trả toàn bộ ranking (person_id, similarity, rank), rank bắt đầu từ 1.

    Sắp giảm theo similarity, hòa điểm theo person_id tăng dần. Registry rỗng
    gây ValueError; không tự trả Unknown hoặc lọc top-k.
    """
    validate_database_embedding(query_embedding)
    if not isinstance(registered_embeddings, dict):
        raise TypeError("registered_embeddings phải là dictionary person_id -> embedding.")
    if not registered_embeddings:
        raise ValueError("Không có danh tính đăng ký để matching.")
    for person_id, embedding in registered_embeddings.items():
        validate_person_id(person_id)
        try:
            validate_database_embedding(embedding)
        except (TypeError, ValueError) as error:
            raise type(error)(f"Embedding đăng ký của {person_id}: {error}") from error
    results = [
        {"person_id": person_id, "similarity": cosine_similarity(query_embedding, embedding)}
        for person_id, embedding in registered_embeddings.items()
    ]
    results.sort(key=lambda result: (-result["similarity"], result["person_id"]))
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results


def validate_threshold(threshold: float) -> float:
    """Trả Python float hữu hạn thuộc [-1,1]; giữ việc từ chối bool/chuỗi."""
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        raise TypeError("threshold phải là số thực, không phải bool hoặc chuỗi.")
    value = float(threshold)
    if not np.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("threshold phải hữu hạn trong [-1,1].")
    return value


def detect_unknown(
    query_embedding: np.ndarray,
    registered_embeddings: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, object]:
    """Chấp nhận top-1 khi score >= threshold, còn lại predicted_id='Unknown'.

    best_match_id luôn là ứng viên gần nhất, không xác nhận danh tính khi bị từ
    chối. Giữ thứ tự validation và sáu trường kết quả của notebook 08–10.
    """
    threshold = validate_threshold(threshold)
    ranking = match_embedding(query_embedding, registered_embeddings)
    if "Unknown" in registered_embeddings:
        raise ValueError("person_id 'Unknown' được dành riêng cho kết quả từ chối.")
    best = ranking[0]
    is_known = bool(best["similarity"] >= threshold)
    return {
        "predicted_id": best["person_id"] if is_known else "Unknown",
        "best_match_id": best["person_id"],
        "similarity": best["similarity"],
        "threshold": threshold,
        "is_known": is_known,
        "ranking": ranking,
    }
