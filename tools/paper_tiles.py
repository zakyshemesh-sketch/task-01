"""Tiles of my own torn paper for the pattern modules: crops taken along the torn edges (not the straight
outer edges of the sheet), in grey, with the backdrop removed. One atlas image, 8 x 4 tiles. Run from the repo root."""
import random, numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
random.seed(5)
TW, TH = 180, 300                       # one module is 0.6 wide for every 1 high
names = ["tear-vertical-gentle", "tear-hole", "tear-diagonal", "tear-curve-deep", "tear-strips"]
tiles = []
for n in names:
    im = Image.open(f"raw/tears/{n}-cutout.png").convert("RGBA")
    a = np.asarray(im)[..., 3] > 128
    ys, xs = np.where(a); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    edge = a ^ ndi.binary_erosion(a, iterations=2)
    inner = np.zeros_like(a); inner[y0 + 90:y1 - 90, x0 + 90:x1 - 90] = True     # skip the straight outer edges
    ey, ex = np.where(edge & inner)
    picks = random.sample(range(len(ey)), min(400, len(ey)))
    got = 0
    for k in picks:
        cx, cy = ex[k], ey[k]
        sw = random.choice([1.0, 1.4, 1.9]); w, h = int(TW * sw), int(TH * sw)
        box = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
        if box[0] < 0 or box[1] < 0 or box[2] > im.width or box[3] > im.height: continue
        crop = im.crop(box)
        cov = (np.asarray(crop)[..., 3] > 128).mean()
        if not .35 < cov < .8: continue                      # both paper and gap in the tile
        g = ImageOps.autocontrast(ImageOps.grayscale(crop.convert("RGB")), cutoff=2)
        t = Image.merge("RGBA", (g, g, g, crop.split()[3])).resize((TW, TH), Image.LANCZOS)
        if random.random() < .5: t = ImageOps.mirror(t)
        tiles.append(t); got += 1
        if got >= 7: break
random.shuffle(tiles); tiles = tiles[:32]
atlas = Image.new("RGBA", (TW * 8, TH * 4), (0, 0, 0, 0))
for i, t in enumerate(tiles): atlas.paste(t, ((i % 8) * TW, (i // 8) * TH))
atlas.save("assets/paper-tiles.webp", "WEBP", quality=82, method=6)
print(len(tiles), atlas.size)
