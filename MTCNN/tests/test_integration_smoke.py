"""Real pretrained MTCNN smoke test. Runs on CPU with the included face.jpg."""

from pathlib import Path

import numpy as np
import pytest

from MTCNN.mtcnn import MTCNN, validate_detection_result
from MTCNN.preprocessing.image_processing import read_image


@pytest.mark.integration
def test_end_to_end_detection_and_uint8_alignment():
    folder = Path(__file__).resolve().parents[1]
    if not (folder / 'face.jpg').is_file() or not all(
        (folder / 'weights' / f'{name}.pt').is_file()
        for name in ('pnet', 'rnet', 'onet')
    ):
        pytest.skip('Demo image or pretrained MTCNN weights not installed')

    detector = MTCNN(device='cpu')
    image = read_image(folder / 'face.jpg')
    result = validate_detection_result(detector.detect(image))
    assert len(result.boxes) >= 1
    faces = detector.extract_faces_rgb(image, result, size=160, align=True)
    assert faces.shape == (len(result.boxes), 160, 160, 3)
    assert faces.dtype == np.uint8
    assert np.isfinite(result.landmarks).all()

    tensor = detector.extract_faces(image, result, size=160, align=True)
    assert tuple(tensor.shape) == (len(result.boxes), 3, 160, 160)
    assert tensor.dtype.is_floating_point
