"""Version 20: the children's faces as a glitch instead of grey squares. Each face area is first destroyed
(heavy blur, then reduced to a few pixels and smoothed back), and only then cut into horizontal bands that
slide sideways with a split between the colour channels. Nothing of the faces survives in the files.
Run from the repo root. The source stays in raw/private (not in the repo)."""
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
faces = [(620, 820, 1050, 1380), (300, 1350, 810, 2020), (670, 1410, 1125, 2090)]   # baby, boy, toddler, with margin
RIGHT = 1135   # my face starts just right of the toddler: no band may run past this line (v23)

def destroy(region):
    w, h = region.size
    r = region.filter(ImageFilter.GaussianBlur(45))
    return r.resize((max(2, w // 70), max(2, h // 70)), Image.BILINEAR).resize((w, h), Image.BICUBIC)

def glitch(src, seed):
    random.seed(seed); rng = np.random.default_rng(seed)
    im = np.asarray(src.convert("RGB")).astype(np.float32)
    H, W, _ = im.shape
    for x0, y0, x1, y1 in faces:
        pad = 90
        X0, X1 = max(0, x0 - pad), min(W, x1 + pad)
        base = np.asarray(destroy(src.convert("RGB").crop((X0, y0, X1, y1)))).astype(np.float32)
        bw_ = X1 - X0
        y = y0
        while y < y1:
            bh = random.choice([3, 5, 8, 12, 18, 26, 40, 60])
            ya, yb = y, min(y1, y + bh); y = yb
            # this band covers the face, and sometimes runs past it
            a = max(X0, x0 - random.randint(0, pad)); b = min(X1, RIGHT, x1 + random.randint(0, pad))
            shift = int(rng.normal(0, 40)); split = random.choice([0, 4, 8, 14, 22])
            band = base[ya - y0:yb - y0]
            out = np.empty_like(band)
            for ch, d in zip(range(3), (split, 0, -split)):
                out[..., ch] = np.roll(band[..., ch], shift + d, axis=1)
            out *= random.uniform(.8, 1.2)
            if random.random() < .12: out[:] = out.mean(axis=(0, 1)) * random.choice([.3, 1.6])   # a dead or burnt line
            out += rng.normal(0, 6, out.shape)
            seg = out[:, a - X0:b - X0]
            # soft ends, so the band fades into the photo instead of ending in a hard edge
            ramp = np.ones(b - a, np.float32); f = min(24, (b - a) // 4)
            ramp[:f] = np.linspace(0, 1, f)
            if b < RIGHT: ramp[-f:] = np.linspace(1, 0, f)
            m = ramp[None, :, None]
            # the face itself is always fully covered
            core = np.zeros(b - a, bool); core[max(0, x0 - a):max(0, x1 - a)] = True
            m = np.where(core[None, :, None], 1, m)
            im[ya:yb, a:b] = im[ya:yb, a:b] * (1 - m) + seg * m
    return Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))

src = Image.open("raw/private/family.jpg").convert("RGB")
col = glitch(src, 11).resize((1050, 1400), Image.LANCZOS)
col.save("assets/family-color.jpg", "JPEG", quality=82, optimize=True, progressive=True)
bw = ImageOps.autocontrast(ImageOps.grayscale(glitch(src, 11)), cutoff=1).resize((1050, 1400), Image.LANCZOS)
bw.save("assets/family-bw.jpg", "JPEG", quality=82, optimize=True, progressive=True)
print(col.size, src.size)
