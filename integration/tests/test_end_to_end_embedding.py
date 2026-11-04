
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT,):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import cv2
import torch
import MTCNN.mtcnn as mtcnn_module
import integration.mtcnn_adapter as adapter_module
import FaceNet.facenet_model as model_module
from FaceNet.contracts import validate_mtcnn_output
from MTCNN.preprocessing.image_processing import read_image


class TestEndToEndEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for module in (mtcnn_module, adapter_module, model_module):
            if not Path(module.__file__).resolve().is_relative_to(ROOT):
                raise AssertionError("Phải dùng module từ project tích hợp")
        if torch.version.cuda is not None or torch.cuda.is_available():
            raise AssertionError("Environment phải CPU-only")
        cls.detector = mtcnn_module.MTCNN(device="cpu")
        cls.net = model_module.load_facenet_model(ROOT / "FaceNet/models/facenet.onnx")
        for net in (cls.detector.pnet, cls.detector.rnet, cls.detector.onet):
            if str(next(net.parameters()).device) != "cpu":
                raise AssertionError("MTCNN phải chạy CPU")

    def pipeline(self, image):
        original = image.copy()
        result = self.detector.detect(image)
        result_before = deepcopy(result)
        faces = self.detector.extract_faces_rgb(image, result, size=160, align=True)
        faces_before = faces.copy()
        output = adapter_module.mtcnn_to_facenet_detections(image, result, faces)
        validate_mtcnn_output(output)
        self.assertEqual(len(output), len(result.boxes))
        embeddings = []
        order_max_difference = 0.0
        for index, item in enumerate(output):
            np.testing.assert_array_equal(item["face_rgb"], faces[index])
            embedding = model_module.extract_embedding(item["face_rgb"], self.net)
            # Đối chiếu từng index với aligned face tương ứng, không mock model.
            direct = model_module.extract_embedding(faces[index], self.net)
            difference = float(np.max(np.abs(embedding - direct)))
            order_max_difference = max(order_max_difference, difference)
            np.testing.assert_allclose(embedding, direct, atol=1e-6, rtol=1e-5)
            self.assertEqual(embedding.shape, (512,))
            self.assertEqual(embedding.dtype, np.float32)
            self.assertTrue(np.isfinite(embedding).all())
            self.assertAlmostEqual(float(np.linalg.norm(embedding.astype(np.float64))), 1.0, places=6)
            embeddings.append(embedding)
        matrix = np.stack(embeddings) if embeddings else np.empty((0,512),np.float32)
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(faces, faces_before)
        for name in ("boxes", "scores", "landmarks"):
            np.testing.assert_array_equal(getattr(result,name),getattr(result_before,name))
        return result, output, matrix, order_max_difference

    def compare_twice(self, name, image, expected_count):
        start = time.perf_counter()
        first = self.pipeline(image)
        second = self.pipeline(image)
        result, output, embeddings, order_error = first
        self.assertEqual(len(output), expected_count)
        self.assertEqual(embeddings.shape, (expected_count,512))
        self.assertEqual(len(second[1]), expected_count)
        for field in ("boxes", "scores", "landmarks"):
            np.testing.assert_allclose(getattr(first[0],field),getattr(second[0],field),atol=1e-6,rtol=1e-5)
        for a,b in zip(output,second[1]):
            np.testing.assert_array_equal(a["face_rgb"],b["face_rgb"])
            self.assertEqual({k:v for k,v in a.items() if k!="face_rgb"},
                             {k:v for k,v in b.items() if k!="face_rgb"})
        difference = float(np.max(np.abs(embeddings-second[2]))) if expected_count else 0.0
        norm_error = float(np.max(np.abs(np.linalg.norm(embeddings.astype(np.float64),axis=1)-1))) if expected_count else 0.0
        np.testing.assert_allclose(embeddings,second[2],atol=1e-6,rtol=1e-5)
        print("END_TO_END_EMBEDDING " + json.dumps({
            "sample":name,"pipeline_runs":2,"detections":len(result.boxes),
            "aligned_faces":len(output),"embeddings_shape":list(embeddings.shape),
            "dtype":str(embeddings.dtype),"finite":bool(np.isfinite(embeddings).all()),
            "max_abs_difference":difference,"max_l2_norm_error":norm_error,
            "face_order_max_difference":max(order_error,second[3]),
            "input_unchanged":True,"seconds":time.perf_counter()-start,
            "landmarks_omitted":sum("landmarks" not in item for item in output),
            "bbox_clipped":sum(not np.array_equal(item["bbox"],result.boxes[i]) for i,item in enumerate(output)),
        },sort_keys=True))

    def test_one_face_twice(self):
        image = read_image(ROOT / "integration/tests/fixtures/mtcnn_single.png")
        self.compare_twice("single",image,1)

    def test_many_faces_twice(self):
        image = read_image(ROOT / "integration/tests/fixtures/mtcnn_multi.jpg")
        self.compare_twice("multi",image,5)

    def test_no_face_twice(self):
        self.compare_twice("black_in_RAM",np.zeros((240,320,3),np.uint8),0)


if __name__ == "__main__":
    unittest.main()
