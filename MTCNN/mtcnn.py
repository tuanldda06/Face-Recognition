"""MTCNN: P-Net -> R-Net -> O-Net. Ảnh RGB uint8, boxes [x1,y1,x2,y2]."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from MTCNN.model_loader import load_pretrained_models
from MTCNN.detection.image_pyramid import build_image_pyramid
from MTCNN.detection.bounding_box import (
    generate_bounding_box, nms, calibrate_boxes, square_boxes,
    valid_box_indices, decode_landmarks,
)
from MTCNN.preprocessing.image_processing import (
    read_image, check_rgb, normalize_pixels, crop_and_resize,
)


@dataclass
class DetectionResult:
    boxes: np.ndarray       # (N, 4)
    scores: np.ndarray      # (N,)
    landmarks: np.ndarray   # (N, 5, 2)

    @staticmethod
    def empty():
        return DetectionResult(
            np.empty((0, 4), np.float32),
            np.empty(0, np.float32),
            np.empty((0, 5, 2), np.float32),
        )

class MTCNN:
    def __init__(self, device="cpu", weights_dir=None, models=None,
                 min_face_size=20, factor=0.709,
                 thresholds=(0.6, 0.7, 0.7), batch_size=256):
        self.device = torch.device(device)
        self.min_face_size, self.factor = min_face_size, factor
        self.thresholds, self.batch_size = thresholds, batch_size
        if len(thresholds) != 3 or batch_size < 1 or min_face_size <= 0 or not 0 < factor < 1:
            raise ValueError("Tham số MTCNN không hợp lệ")
        nets = models if models is not None else load_pretrained_models(
            device=self.device, weights_dir=weights_dir
        )
        if len(nets) != 3:
            raise ValueError("Cần đúng 3 model: P-Net, R-Net, O-Net")
        self.pnet, self.rnet, self.onet = [net.to(self.device).eval() for net in nets]

    def _predict(self, model, crops):
        """Nhận NHWC đã normalize, chạy CNN theo batch, trả về NumPy."""
        outputs = []
        for start in range(0, len(crops), self.batch_size):
            batch = np.ascontiguousarray(crops[start:start + self.batch_size])
            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
            outputs.append([value.cpu().numpy() for value in model(tensor)])
        return [np.concatenate(parts) for parts in zip(*outputs)]

    def _run_pnet(self, image):
        proposals = []
        for scale, scaled in build_image_pyramid(image, self.min_face_size, self.factor):
            offsets, probs = self._predict(self.pnet, normalize_pixels(scaled)[None])
            boxes, scores, reg = generate_bounding_box(
                probs[0, 1], offsets[0].transpose(1, 2, 0), scale, self.thresholds[0]
            )
            if len(boxes):
                keep = nms(boxes, scores, 0.5)
                proposals.append((boxes[keep], scores[keep], reg[keep]))

        if not proposals:
            return np.empty((0, 4), np.float32)
        boxes, scores, offsets = [np.concatenate(x) for x in zip(*proposals)]
        keep = nms(boxes, scores, 0.7)
        boxes = square_boxes(calibrate_boxes(boxes[keep], offsets[keep]))
        return boxes[valid_box_indices(boxes)]

    def _run_rnet(self, image, boxes):
        crops, actual = crop_and_resize(image, boxes, 24)
        offsets, probs = self._predict(self.rnet, crops)
        selected = probs[:, 1] >= self.thresholds[1]
        if not selected.any():
            return actual[:0]

        boxes, offsets, scores = actual[selected], offsets[selected], probs[selected, 1]
        keep = nms(boxes, scores, 0.7)
        boxes = square_boxes(calibrate_boxes(boxes[keep], offsets[keep]))
        return boxes[valid_box_indices(boxes)]

    def _run_onet(self, image, boxes):
        crops, actual = crop_and_resize(image, boxes, 48)
        offsets, points, probs = self._predict(self.onet, crops)
        selected = probs[:, 1] >= self.thresholds[2]
        if not selected.any():
            return DetectionResult.empty()

        boxes, offsets = actual[selected], offsets[selected]
        scores, points = probs[selected, 1], points[selected]
        landmarks = decode_landmarks(boxes, points)  # Trước khi calibrate boxes
        boxes = calibrate_boxes(boxes, offsets)
        valid = valid_box_indices(boxes)
        boxes, scores, landmarks = boxes[valid], scores[valid], landmarks[valid]
        if not len(boxes):
            return DetectionResult.empty()
        keep = nms(boxes, scores, 0.7, mode="min")
        return DetectionResult(boxes[keep], scores[keep], landmarks[keep])

    @torch.inference_mode()
    def detect(self, image):
        if isinstance(image, (str, Path)):
            image = read_image(image)
        image = check_rgb(image)
        boxes = self._run_pnet(image)
        if len(boxes):
            boxes = self._run_rnet(image, boxes)
        return self._run_onet(image, boxes) if len(boxes) else DetectionResult.empty()

    @torch.inference_mode()
    @torch.inference_mode()
    def extract_faces(self, image, result=None, size=160):
        if isinstance(image, (str, Path)):
            image = read_image(image)

        image = check_rgb(image)

        if result is None:
            result = self.detect(image)

        crops, _ = crop_and_resize(
        image, result.boxes, size
        )

        tensor = torch.from_numpy(np.ascontiguousarray(crops))

        return tensor.permute(0, 3, 1, 2)
