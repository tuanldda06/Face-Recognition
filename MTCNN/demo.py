"""Chạy demo: từ root project, dùng `python -m MTCNN.demo --help`."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from MTCNN.mtcnn import MTCNN
from MTCNN.preprocessing.image_processing import read_image


SAMPLE_IMAGE = Path(__file__).resolve().parent / "face.jpg"


def main():
    parser = argparse.ArgumentParser(description="MTCNN face detection demo")
    parser.add_argument(
        "image", nargs="?", type=Path, default=SAMPLE_IMAGE,
        help="Đường dẫn ảnh RGB (mặc định: MTCNN/face.jpg)",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu",
        help="Thiết bị chạy inference (mặc định: cpu)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Lưu ảnh có bounding boxes và 5 landmarks, ví dụ detected.jpg",
    )
    args = parser.parse_args()

    image = read_image(args.image)
    detector = MTCNN(device=args.device)
    result = detector.detect(image)

    print("Boxes:", result.boxes)
    print("Scores:", result.scores)
    print("Landmarks:", result.landmarks)
    faces = detector.extract_faces(image, result)
    print("FaceNet input shape:", tuple(faces.shape))

    if args.output:
        annotated = Image.fromarray(image).copy()
        draw = ImageDraw.Draw(annotated)
        for box, landmarks in zip(result.boxes, result.landmarks):
            draw.rectangle(tuple(float(v) for v in box), outline="lime", width=3)
            for x, y in landmarks:
                x, y = float(x), float(y)
                draw.ellipse((x-3, y-3, x+3, y+3), fill="red")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(args.output)
        print("Đã lưu ảnh kết quả:", args.output)


if __name__ == "__main__":
    main()
