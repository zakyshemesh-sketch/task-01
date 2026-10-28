"""Black and white family photo for the end of the field. The children's faces are replaced by random
grey cells (not blurred, not pixelated: nothing of the face survives in the file). Run from the repo root."""
import random
from PIL import Image, ImageOps, ImageDraw
random.seed(11)
im = ImageOps.grayscale(Image.open("raw/private/family.jpg"))
im = ImageOps.autocontrast(im, cutoff=1)
d = ImageDraw.Draw(im)
faces = [(620, 820, 1050, 1380), (300, 1350, 810, 2020), (670, 1410, 1165, 2090)]   # baby, boy, toddler, with margin
cell = 28
for x0, y0, x1, y1 in faces:
    for y in range(y0, y1, cell):
        for x in range(x0, x1, cell):
            v = random.choice([12, 20, 30, 45, 70, 110, 160, 220]) if random.random() < .7 else 12
            d.rectangle([x, y, x + cell - 2, y + cell - 2], fill=v)
im = im.resize((1050, 1400), Image.LANCZOS)
im.save("assets/family-bw.jpg", "JPEG", quality=82, optimize=True, progressive=True)
print(im.size)
