"""Build the web assets for the page from the raw photos. Run from the repo root."""
import numpy as np, random
from PIL import Image, ImageFilter, ImageOps, ImageChops
from scipy import ndimage as ndi
OUT = "assets/"
random.seed(7)

def bbox_crop(rgba):
    a = np.asarray(rgba)[..., 3]
    ys, xs = np.where(a > 20)
    return rgba.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

def save_webp(im, name, q=82):
    im.save(OUT + name, "WEBP", quality=q, method=6)

def mask_from(rgba, erode_px):
    a = np.asarray(rgba)[..., 3] > 128
    a = ndi.binary_erosion(a, iterations=erode_px)
    m = Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    out = Image.new("RGBA", rgba.size, (0, 0, 0, 0)); out.putalpha(m)
    return out

def components(rgba, min_frac=0.02):
    a = np.asarray(rgba)[..., 3] > 128
    lab, n = ndi.label(a)
    parts = []
    for i in range(1, n + 1):
        comp = lab == i
        if comp.mean() < min_frac * (1 if min_frac > .01 else 1):
            continue
        ys, xs = np.where(comp)
        parts.append((xs.mean(), comp))
    parts.sort(key=lambda t: t[0])
    return [c for _, c in parts]

def piece(rgba, comp):
    arr = np.asarray(rgba).copy()
    arr[..., 3] = np.where(comp, arr[..., 3], 0)
    return Image.fromarray(arr)

H = 1600  # working height for sheets

def sheet(name, split):
    im = bbox_crop(Image.open(f"raw/tears/{name}-cutout.png").convert("RGBA"))
    im = im.resize((round(im.width * H / im.height), H), Image.LANCZOS)
    if not split:
        save_webp(im, f"{name}-paper.webp"); save_webp(mask_from(im, 7), f"{name}-mask.webp", 90)
        return im.size
    halves = components(im, 0.05)
    for side, comp in zip(("a", "b"), halves[:2]):
        p = piece(im, comp)
        save_webp(p, f"{name}-{side}-paper.webp"); save_webp(mask_from(p, 7), f"{name}-{side}-mask.webp", 90)
    return im.size

sizes = {}
sizes["tear-hole"] = sheet("tear-hole", False)
for n in ("tear-vertical-gentle", "tear-diagonal", "tear-curve-deep"):
    sizes[n] = sheet(n, True)

# strips: left and right narrow strips, rotated to lie horizontally
im = bbox_crop(Image.open("raw/tears/tear-strips-cutout.png").convert("RGBA"))
parts = components(im, 0.01)
strip_imgs = [bbox_crop(piece(im, parts[0])), bbox_crop(piece(im, parts[-1]))]
k = 0
for s in strip_imgs:
    for flip in (False, True):
        r = s.rotate(90, expand=True)
        if flip: r = ImageOps.flip(ImageOps.mirror(r))
        r = r.resize((1600, round(r.height * 1600 / r.width)), Image.LANCZOS)
        save_webp(r, f"strip-{k}.webp"); k += 1
        sizes[f"strip-{k-1}"] = r.size

# photos
def photo(src, name, maxw=1400, q=80):
    p = Image.open(src).convert("RGB"); p.thumbnail((maxw, maxw), Image.LANCZOS)
    p.save(OUT + name, "JPEG", quality=q, optimize=True, progressive=True); sizes[name] = p.size
photo("raw/real/selfie-park.jpg", "selfie.jpg")
photo("raw/real/living-room-morning.jpg", "room-morning.jpg")
photo("raw/real/portrait-sofa-styled.jpg", "room-styled.jpg")
for f in ("portrait-a", "portrait-b", "portrait-c", "portrait-d"):
    photo(f"process/03-ai-described/{f}.jpg", f"ai-{f}.jpg", 700)
photo("process/01-ai-default/home-b.jpg", "ai-home.jpg")
photo("process/01-ai-default/childhood-a.jpg", "ai-child-a.jpg", 900)
photo("process/02-ai-with-reference/childhood-b.jpg", "ai-child-b.jpg", 900)
photo("process/01-ai-default/street-a.jpg", "ai-street-a.jpg", 900)
photo("process/02-ai-with-reference/street-a.jpg", "ai-street-b.jpg", 900)
photo("process/02-ai-with-reference/street-b.jpg", "ai-street-c.jpg", 900)

# family: cover the children's faces with torn paper, baked into the pixels
fam = Image.open("raw/private/family.jpg").convert("RGB")
faces = [(650, 850, 1020, 1350), (330, 1380, 780, 1990), (700, 1440, 1180, 2060)]   # baby, boy, toddler (x0,y0,x1,y1)
riso = [(255, 212, 0), (255, 90, 168), (255, 212, 0)]
src = Image.open("raw/tears/tear-strips-cutout.png").convert("RGBA")
mid = bbox_crop(piece(bbox_crop(src), components(bbox_crop(src), 0.01)[1]))
for (x0, y0, x1, y1), col in zip(faces, riso):
    w, h = x1 - x0, y1 - y0
    # a torn-edged scrap from the middle sheet, big enough to cover with margin
    sx = random.randint(0, mid.width - w - 90); sy = random.randint(0, mid.height - h - 90)
    scrap = mid.crop((sx, sy, sx + w + 90, sy + h + 90))
    # torn edges: borrow the alpha of a real torn strip, stretched over the scrap
    strip = Image.open(OUT + f"strip-{random.randint(0,3)}.webp").convert("RGBA")
    if random.random() < .5: strip = strip.rotate(90, expand=True)
    edge = strip.split()[3].resize(scrap.size, Image.LANCZOS)
    core = Image.new("L", scrap.size, 0)
    core.paste(255, (45, 45, scrap.width - 45, scrap.height - 45))      # the face box itself is always fully covered
    edge = ImageChops.lighter(edge, core)
    scrap.putalpha(edge)
    tint = Image.new("RGB", scrap.size, col)
    rgb = ImageChops.multiply(scrap.convert("RGB"), tint)
    scrap = Image.merge("RGBA", (*rgb.split(), edge))
    scrap = scrap.rotate(random.uniform(-9, 9), expand=True, resample=Image.BICUBIC)
    shadow = Image.new("RGBA", scrap.size, (0, 0, 0, 0)); sa = scrap.split()[3].filter(ImageFilter.GaussianBlur(14))
    shadow.putalpha(sa.point(lambda v: v * 0.45))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    pos = (cx - scrap.width // 2, cy - scrap.height // 2)
    fam.paste(shadow, (pos[0] + 10, pos[1] + 16), shadow)
    fam.paste(scrap, pos, scrap)
fam.thumbnail((1400, 1400), Image.LANCZOS)
fam.save(OUT + "family-covered.jpg", "JPEG", quality=80, optimize=True, progressive=True); sizes["family"] = fam.size
for k, v in sizes.items(): print(k, v)

# plain paper texture from the middle sheet of the strips photo (an untouched area)
tex = mid.convert("RGB")
cx, cy = tex.width // 2, tex.height // 2
tex = tex.crop((cx - 450, cy - 600, cx + 450, cy + 600))
tex.save(OUT + "paper-texture.jpg", "JPEG", quality=78, optimize=True)
