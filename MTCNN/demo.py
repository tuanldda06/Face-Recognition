"""Demo from project root: python -m MTCNN.demo --help"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from MTCNN.mtcnn import MTCNN
from MTCNN.preprocessing.image_processing import read_image


SAMPLE_IMAGE = Path(__file__).resolve().parent / "face.jpg"


def main():
    parser = argparse.ArgumentParser(description="MTCNN detection and optional 5-point alignment")
    parser.add_argument("image", nargs="?", type=Path, default=SAMPLE_IMAGE,
                        help="Đường dẫn ảnh (mặc định: MTCNN/face.jpg)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="Lưu ảnh đã vẽ boxes/landmarks")
    parser.add_argument("--faces-dir", type=Path, help="Lưu ảnh mặt RGB uint8 (N files)")
    parser.add_argument("--align", action="store_true",
                        help="Dùng 5 landmarks để căn chỉnh ảnh mặt trước khi xuất")
    args = parser.parse_args()

    image = read_image(args.image)
    detector = MTCNN(device=args.device)
    result = detector.detect(image)

    print("Boxes:", result.boxes)
    print("Scores:", result.scores)
    print("Landmarks:", result.landmarks)

    rgb_faces = detector.extract_faces_rgb(image, result, align=args.align)
    print("FaceNet input RGB:", tuple(rgb_faces.shape), rgb_faces.dtype)

    if args.faces_dir:
        args.faces_dir.mkdir(parents=True, exist_ok=True)
        for i, face in enumerate(rgb_faces):
            path = args.faces_dir / f"face_{i + 1}.png"
            Image.fromarray(face).save(path)
            print("Đã lưu ảnh mặt RGB:", path)

    if args.output:
        annotated = Image.fromarray(image).copy()
        draw = ImageDraw.Draw(annotated)
        for box, landmarks in zip(result.boxes, result.landmarks):
            draw.rectangle(tuple(float(v) for v in box), outline="lime", width=3)
            for x, y in landmarks:
                x, y = float(x), float(y)
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(args.output)
        print("Đã lưu ảnh có boxes/landmarks:", args.output)


if __name__ == "__main__":
    main()
