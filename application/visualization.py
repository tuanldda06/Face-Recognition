from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def annotate(image_rgb, faces):
    image = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(image)
    size = max(13, min(22, image.width // 40))
    font = ImageFont.load_default(size=size)
    for path in (Path("C:/Windows/Fonts/segoeui.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if path.is_file():
            font = ImageFont.truetype(str(path), size)
            break
    for face in faces:
        known = face["status"] == "recognized"
        color = (22, 163, 74) if known else (234, 110, 30)
        x1, y1, x2, y2 = face["bbox"]
        name = face["person_id"] if known else "Unknown"
        score = "n/a" if face["similarity"] is None else f'{face["similarity"]:.3f}'
        label = f'{name} | det {face["confidence"]:.2f} | cos {score}'
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=3)
        y = max(0, y1 - size - 8)
        bounds = draw.textbbox((x1, y), label, font=font)
        draw.rectangle(bounds, fill=color)
        draw.text((x1, y), label, font=font, fill="white")
    return np.asarray(image)
