"""In-memory consistency gate for webcam enrollment embeddings.

This gate checks pairwise similarity, not identity, pose diversity or liveness.
Callers must use the same FaceNet weights, RGB preprocessing, MTCNN alignment
and model_version as recognition. No inference or persistence happens here.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from FaceNet.matching import (
    EMBEDDING_SIZE,
    cosine_similarity,
    validate_database_embedding,
    validate_threshold,
)


def validate_embedding_consistency(
    embeddings: list[np.ndarray] | tuple[np.ndarray, ...] | np.ndarray,
    min_similarity: float = 0.65,
    *,
    expected_count: int = 5,
) -> list[np.ndarray]:
    """Return independent float32 L2 embeddings, or raise ValueError.

    Accept exactly expected_count real numeric (512,) vectors in a list,
    tuple, or 2D array.
    Every vector must be finite and nonzero; vectors are normalized on copies
    in float64, then returned as float32, matching the FaceNet contract.
    All pairwise cosine scores must be >= min_similarity (finite [-1,1]).
    Errors identify zero-based embedding indices and never modify input.

    The default 0.65 is a configurable consistency policy, not a calibrated
    recognition threshold. Different people can pass; severe pose/lighting
    changes for one person can fail. Repeated frames can also pass this gate.
    """
    try:
        threshold = validate_threshold(min_similarity)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("min_similarity must be a finite real number in [-1,1].") from error

    if type(expected_count) is not int or expected_count < 2:
        raise ValueError("expected_count must be an integer of at least 2.")

    if (not isinstance(embeddings, (list, tuple, np.ndarray))
            or (isinstance(embeddings, np.ndarray) and embeddings.ndim != 2)
            or len(embeddings) != expected_count):
        raise ValueError(f"Exactly {expected_count} embeddings of shape (512,) are required.")

    normalized = []
    for index, embedding in enumerate(embeddings):
        try:
            vector = np.asarray(embedding)
            if vector.shape != (EMBEDDING_SIZE,):
                raise ValueError(f"Embedding must have shape ({EMBEDDING_SIZE},).")
            if vector.dtype.kind not in "iuf":
                raise ValueError("Embedding must contain real numbers, not bool/string/object/complex.")
            with np.errstate(over="ignore", invalid="ignore"):
                vector = vector.astype(np.float64, copy=True)
            if not np.isfinite(vector).all():
                raise ValueError("Embedding must be finite; NaN/Inf are not allowed.")
            scale = float(np.max(np.abs(vector)))
            if scale == 0:
                raise ValueError("Embedding must be nonzero.")
            # Scaling first avoids overflow/underflow for finite nonzero input.
            vector /= scale
            vector /= np.linalg.norm(vector)
            result = vector.astype(np.float32)
            validate_database_embedding(result)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"embedding_index {index}: {error}") from error
        normalized.append(result)

    for first, second in combinations(range(len(normalized)), 2):
        similarity = cosine_similarity(normalized[first], normalized[second])
        if similarity < threshold:
            raise ValueError(
                f"embedding_index {first} and {second}: cosine similarity "
                f"{similarity:.8g} is below min_similarity {threshold:.8g}; "
                "the samples are not consistent."
            )
    return normalized
