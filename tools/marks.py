"""Second sheet, red crayon (raw/handwriting/lines-shapes-upright.jpg): the final letters ך ם ן, lines and shapes.
- The three final letters are added to assets/hand-glyphs.png, scaled to sit with the blue alphabet.
- Lines and shapes go to assets/hand-marks.png + .json (white strokes on transparent), for the route.
Run from the repo root."""
import json, numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi
a = np.asarray(Image.open("raw/handwriting/lines-shapes-upright.jpg").convert("RGB")).astype(np.int16)
R, G, B = a[..., 0], a[..., 1], a[..., 2]
ink = ndi.binary_opening((R - G > 45) & (R - B > 25), iterations=1)
soft = np.clip((R - G - 25) / 55.0, 0, 1) * ndi.binary_dilation(ink, iterations=3)

def cut(x0, y0, x1, y1, clean=False):
    g = soft[y0:y1, x0:x1].copy()
    if clean:                                          # drop crumbs of crayon next to a letter
        lab, n = ndi.label(ndi.binary_dilation(g > .15, iterations=4))
        sizes = ndi.sum(g > .15, lab, range(1, n + 1))
        g *= np.isin(lab, [i + 1 for i, v in enumerate(sizes) if v > .2 * sizes.max()])
    ys, xs = np.nonzero(g > .15)
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return Image.fromarray((g * 255).astype(np.uint8))

# boxes found by segmenting the sheet (x0, y0, x1, y1), with margins
FINALS = {"ך": (1870, 255, 2030, 540), "ם": (1985, 280, 2185, 490), "ן": (2135, 285, 2290, 590)}
MARKS = {
    "circle": (245, 295, 485, 490), "x": (525, 310, 695, 490), "square": (750, 305, 960, 505),
    "circle-s": (1060, 265, 1170, 370), "circle-m": (1035, 350, 1210, 505), "circle-l": (1015, 500, 1220, 680),
    "short-1": (230, 500, 850, 580), "short-2": (230, 578, 780, 650), "short-3": (230, 640, 710, 712),
    "line-1": (230, 700, 2340, 805), "line-2": (230, 890, 2340, 985), "line-3": (230, 1085, 2360, 1172),
    "line-4": (230, 1212, 2340, 1312), "line-5": (230, 1400, 2340, 1495), "line-6": (230, 1555, 2360, 1640),
}
# the short lines sit close to the shapes: keep only what is inside each line's own band
# ---- final letters, into the glyph atlas ----
atlas = Image.open("assets/hand-glyphs.png"); meta = json.load(open("assets/hand-glyphs.json"))
CW, CH = meta["cw"], meta["ch"]
ref = atlas.getchannel("A").crop((meta["chars"].index("מ") * CW, 0, meta["chars"].index("מ") * CW + CW, CH)).getbbox()   # where מ sits in its cell
mem = cut(*FINALS["ם"], clean=True); sc = (ref[3] - ref[1]) / mem.height
out = Image.new("RGBA", (atlas.width + 3 * CW, CH), (255, 255, 255, 0)); out.paste(atlas, (0, 0))
chars = meta["chars"]
for i, ch in enumerate("ךםן"):
    if ch in chars: continue
    g = cut(*FINALS[ch], clean=True); g = g.resize((max(1, int(g.width * sc)), max(1, int(g.height * sc))), Image.LANCZOS)
    if g.height > CH - 4: g = g.resize((max(1, int(g.width * (CH - 4) / g.height)), CH - 4), Image.LANCZOS)
    g = g.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(.6))   # the red crayon was thinner than the blue one
    white = Image.new("RGBA", g.size, (255, 255, 255, 255)); white.putalpha(g)
    x = len(chars) * CW + (CW - g.width) // 2
    out.alpha_composite(white, (x, min(ref[1], CH - g.height))); chars += ch
out = out.crop((0, 0, len(chars) * CW, CH))
out.save("assets/hand-glyphs.png", optimize=True)
json.dump({"cw": CW, "ch": CH, "chars": chars}, open("assets/hand-glyphs.json", "w"), ensure_ascii=False)
print("glyphs", len(chars), chars, "scale", round(sc, 3))
# ---- lines and shapes, packed in rows ----
items, x, y, rowh, W = {}, 0, 0, 0, 2200
imgs = {k: cut(*b) for k, b in MARKS.items()}
for k, g in imgs.items():
    if x + g.width > W: x, y, rowh = 0, y + rowh + 4, 0
    items[k] = [x, y, g.width, g.height]; x += g.width + 4; rowh = max(rowh, g.height)
sheet = Image.new("RGBA", (W, y + rowh), (255, 255, 255, 0))
for k, g in imgs.items():
    white = Image.new("RGBA", g.size, (255, 255, 255, 255)); white.putalpha(g); sheet.paste(white, tuple(items[k][:2]))
sheet.save("assets/hand-marks.png", optimize=True)
json.dump(items, open("assets/hand-marks.json", "w"))
print({k: v[2:] for k, v in items.items()}, sheet.size)
