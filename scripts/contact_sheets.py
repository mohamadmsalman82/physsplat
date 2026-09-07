"""Build labeled contact sheets from converted photos so they can be
visually classified into the capture-guide sets."""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SRC = Path("data/photos/converted")
OUT = Path("data/photos/_sheets")
OUT.mkdir(exist_ok=True)

THUMB = 340
COLS, ROWS = 5, 5  # 25 per sheet

files = sorted(SRC.glob("*.jpg"))
print(f"{len(files)} photos")

try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
except Exception:
    font = ImageFont.load_default()

for s in range(0, len(files), COLS * ROWS):
    batch = files[s : s + COLS * ROWS]
    sheet = Image.new("RGB", (COLS * THUMB, ROWS * (THUMB + 40)), "white")
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(batch):
        im = Image.open(f)
        im.thumbnail((THUMB, THUMB))
        x = (i % COLS) * THUMB
        y = (i // COLS) * (THUMB + 40)
        sheet.paste(im, (x + (THUMB - im.width) // 2, y))
        label = f.stem.replace("IMG_", "")
        d.text((x + 8, y + THUMB + 4), label, fill="black", font=font)
    n = s // (COLS * ROWS)
    sheet.save(OUT / f"sheet_{n:02d}.png")
    print(f"sheet_{n:02d}.png: {batch[0].stem} .. {batch[-1].stem}")
