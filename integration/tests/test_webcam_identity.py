"""Webcam identity gate: invalid input, all-pairs policy and core compatibility."""

from itertools import combinations
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from FaceNet.enrollment import representative_from_embeddings
from FaceNet.matching import cosine_similarity, validate_database_embedding
from integration.webcam_identity import validate_embedding_consistency


def unit(index=0):
    vector = np.zeros(512, np.float32)
    vector[index] = 1
    return vector


def same_identity():
    vectors = []
    for index in range(5):
        vector = unit() * np.float32(0.9)
        vector[index + 1] = np.float32(np.sqrt(1 - 0.9**2))
        vectors.append(vector)
    return vectors


@pytest.mark.parametrize("container", [list, tuple, np.stack])
def test_consistent_samples_return_core_compatible_copies_in_order(container):
    source = same_identity()
    embeddings = container([vector * (index + 2) for index, vector in enumerate(source)])
    before = [vector.copy() for vector in embeddings]
    result = validate_embedding_consistency(embeddings)

    assert isinstance(result, list)
    assert len(result) == 5
    for index, vector in enumerate(result):
        validate_database_embedding(vector)
        np.testing.assert_allclose(vector, source[index], atol=1e-7, rtol=0)
        np.testing.assert_array_equal(embeddings[index], before[index])
        assert all(not np.shares_memory(vector, item) for item in embeddings)
    representative = representative_from_embeddings(np.stack(result))
    validate_database_embedding(representative)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32])
def test_real_numeric_inputs_produce_float32(dtype):
    result = validate_embedding_consistency(np.stack([unit() * 2] * 5).astype(dtype))
    for vector in result:
        np.testing.assert_array_equal(vector, unit())
        assert vector.dtype == np.float32


def test_nested_real_numeric_lists_are_supported():
    result = validate_embedding_consistency([unit().tolist() for _ in range(5)])
    assert len(result) == 5
    for vector in result:
        validate_database_embedding(vector)


@pytest.mark.parametrize("count", [0, 1, 3, 4, 6])
def test_requires_exactly_five_samples(count):
    with pytest.raises(ValueError, match="Exactly 5"):
        validate_embedding_consistency([unit()] * count)


@pytest.mark.parametrize("embeddings", [None, 5, {}, "abcde", iter([unit()] * 5),
                                          np.array(5), np.ones(512),
                                          np.ones((5, 1, 512))])
def test_invalid_outer_container_raises_value_error(embeddings):
    with pytest.raises(ValueError):
        validate_embedding_consistency(embeddings)


@pytest.mark.parametrize("shape", [(), (0,), (511,), (513,), (1, 512), (512, 1)])
def test_rejects_wrong_vector_shape_without_flattening(shape):
    embeddings = [unit(), unit(), np.ones(shape, np.float32), unit(), unit()]
    with pytest.raises(ValueError, match="embedding_index 2:.*shape"):
        validate_embedding_consistency(embeddings)


@pytest.mark.parametrize("bad", [None, [None] * 512, np.ones(512, bool),
                                    np.ones(512, complex), np.full(512, "1"),
                                    unit().astype(object)])
def test_invalid_vector_type_raises_value_error(bad):
    with pytest.raises(ValueError, match="embedding_index 3"):
        validate_embedding_consistency([unit(), unit(), unit(), bad, unit()])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_vector_rejected_without_mutation(value):
    embeddings = np.stack([unit()] * 5)
    embeddings[4, 7] = value
    before = embeddings.copy()
    with pytest.raises(ValueError, match="embedding_index 4:.*finite"):
        validate_embedding_consistency(embeddings)
    np.testing.assert_array_equal(embeddings, before)


def test_zero_vector_rejected_even_with_permissive_threshold():
    with pytest.raises(ValueError, match="embedding_index 4:.*nonzero"):
        validate_embedding_consistency([unit()] * 4 + [np.zeros(512)], -1)


@pytest.mark.parametrize("scale", [np.finfo(np.float64).max,
                                      np.nextafter(np.float64(0), np.float64(1))])
def test_finite_extreme_nonzero_vectors_normalize_without_overflow(scale):
    vector = np.zeros(512, np.float64)
    vector[:2] = scale
    with np.errstate(all="raise"):
        result = validate_embedding_consistency([vector] * 5)
    for item in result:
        validate_database_embedding(item)
        np.testing.assert_allclose(item[:2], 1 / np.sqrt(2), atol=1e-7, rtol=0)


@pytest.mark.parametrize("threshold", [None, True, np.bool_(False), "0.65", [],
                                          np.nan, np.inf, -np.inf, -1.01, 1.01,
                                          10**1000])
def test_invalid_threshold_raises_value_error(threshold):
    with pytest.raises(ValueError, match="min_similarity"):
        validate_embedding_consistency([unit()] * 5, threshold)


@pytest.mark.parametrize("threshold", [-1, 0, 1, np.float32(.65), np.float64(.65)])
def test_numeric_threshold_domain_and_endpoints(threshold):
    assert len(validate_embedding_consistency([unit()] * 5, threshold)) == 5


def test_default_is_point_65_and_override_changes_policy():
    vector = unit() * np.float32(0.6)
    vector[1] = np.float32(0.8)
    embeddings = [unit()] * 4 + [vector]
    with pytest.raises(ValueError, match="min_similarity 0.65"):
        validate_embedding_consistency(embeddings)
    assert len(validate_embedding_consistency(embeddings, min_similarity=.59)) == 5


@pytest.mark.parametrize("outlier", [unit(1), -unit()])
def test_mixed_identity_outlier_rejected_without_mutation(outlier):
    embeddings = np.stack([unit()] * 4 + [outlier])
    before = embeddings.copy()
    with pytest.raises(ValueError, match="embedding_index 0 and 4"):
        validate_embedding_consistency(embeddings)
    np.testing.assert_array_equal(embeddings, before)


def test_all_pairs_checked_instead_of_only_first_frame_or_mean():
    positive = unit() * np.float32(0.8)
    positive[1] = np.float32(0.6)
    negative = positive.copy()
    negative[1] *= -1
    embeddings = [unit(), positive, unit(), unit(), negative]
    assert all(cosine_similarity(embeddings[0], item) > .65 for item in embeddings)
    representative = representative_from_embeddings(np.stack(embeddings))
    assert all(cosine_similarity(representative, item) > .65 for item in embeddings)
    with pytest.raises(ValueError, match="embedding_index 1 and 4"):
        validate_embedding_consistency(embeddings)


@pytest.mark.parametrize("pair", list(combinations(range(5), 2)))
def test_each_of_ten_pairs_can_independently_reject(pair):
    first = unit() * np.float32(.8)
    first[1] = np.float32(.6)
    second = first.copy()
    second[1] *= -1
    embeddings = [unit() for _ in range(5)]
    embeddings[pair[0]], embeddings[pair[1]] = first, second
    with pytest.raises(ValueError, match=f"embedding_index {pair[0]} and {pair[1]}"):
        validate_embedding_consistency(embeddings)


def test_threshold_equality_and_adjacent_values_use_greater_equal():
    vector = unit() * np.float32(.8)
    vector[1] = np.float32(.6)
    embeddings = [unit()] * 4 + [vector]
    normalized = validate_embedding_consistency(embeddings, min_similarity=-1)
    score = cosine_similarity(normalized[0], normalized[-1])
    for threshold in (np.nextafter(score, -np.inf), score):
        assert len(validate_embedding_consistency(embeddings, threshold)) == 5
    with pytest.raises(ValueError):
        validate_embedding_consistency(embeddings, np.nextafter(score, np.inf))


def test_readonly_noncontiguous_input_and_outputs_are_independent():
    base = np.zeros((5, 1024), np.float32)
    embeddings = base[:, ::2]
    embeddings[:, 0] = 3
    before = base.copy()
    base.setflags(write=False)
    embeddings.setflags(write=False)
    result = validate_embedding_consistency(embeddings)
    result[0][:] = 0
    np.testing.assert_array_equal(base, before)
    for item in result[1:]:
        np.testing.assert_array_equal(item, unit())
        assert not np.shares_memory(item, result[0])


def test_repeated_input_reference_returns_independent_arrays():
    vector = unit() * 3
    embeddings = [vector] * 5
    result = validate_embedding_consistency(embeddings)
    result[0][:] = 0
    np.testing.assert_array_equal(vector, unit() * 3)
    for item in result[1:]:
        np.testing.assert_array_equal(item, unit())


def test_import_and_call_do_not_load_models_or_access_business_assets():
    code = f"""
import os, sys
sys.path.insert(0, {str(ROOT)!r})
def guard(event, args):
    if event == 'open':
        mode, flags = args[1:3]
        if str(args[0]).lower().endswith(('.db', '.sqlite', '.onnx', '.pt', '.png', '.jpg')):
            raise AssertionError('Business asset access')
        if ((isinstance(mode, str) and any(c in mode for c in 'wax+')) or
                (isinstance(flags, int) and flags &
                 (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
            raise AssertionError('File write')
    if event in ('sqlite3.connect', 'os.mkdir', 'socket.connect'):
        raise AssertionError('Unexpected I/O')
sys.addaudithook(guard)
import numpy as np
from integration.webcam_identity import validate_embedding_consistency
vector = np.zeros(512, np.float32)
vector[0] = 1
assert len(validate_embedding_consistency([vector] * 5)) == 5
for name in ('cv2', 'torch', 'sqlite3', 'MTCNN', 'FaceNet.facenet_model', 'FaceNet.database'):
    assert name not in sys.modules, name
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
