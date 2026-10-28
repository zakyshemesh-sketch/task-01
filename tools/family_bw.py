"""Family photo for the end of the field, in two versions: black and white (drives the field) and colour
(shown at the end, faded in CSS). The children's faces are replaced by random grey cells in both, so
nothing of their faces survives in either file. Run from the repo root."""
import random
from PIL import Image, ImageOps, ImageDraw
faces = [(620, 820, 1050, 1380), (300, 1350, 810, 2020), (670, 1410, 1165, 2090)]   # baby, boy, toddler, with margin
cell = 28
def cover(im, seed):
    random.seed(seed); d = ImageDraw.Draw(im)
    for x0, y0, x1, y1 in faces:
        for y in range(y0, y1, cell):
            for x in range(x0, x1, cell):
                v = random.choice([12, 20, 30, 45, 70, 110, 160, 220]) if random.random() < .7 else 12
                d.rectangle([x, y, x + cell - 2, y + cell - 2], fill=(v, v, v) if im.mode == "RGB" else v)
    return im.resize((1050, 1400), Image.LANCZOS)
src = Image.open("raw/private/family.jpg").convert("RGB")
bw = cover(ImageOps.autocontrast(ImageOps.grayscale(src), cutoff=1), 11)
bw.save("assets/family-bw.jpg", "JPEG", quality=82, optimize=True, progressive=True)
col = cover(src.copy(), 11)
col.save("assets/family-color.jpg", "JPEG", quality=82, optimize=True, progressive=True)
print(bw.size, col.size)
