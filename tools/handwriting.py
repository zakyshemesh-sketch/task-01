"""Cut my handwritten alphabet and digits (blue crayon on paper) into separate glyphs for the field.
Output: assets/hand-glyphs.png, one row of cells (white strokes on transparent), and assets/hand-glyphs.json
with the character each cell holds. Run from the repo root."""
import json, numpy as np
from PIL import Image
from scipy import ndimage as ndi
im = Image.open("raw/handwriting/alphabet-upright.jpg").convert("RGB")
a = np.asarray(im).astype(np.int16)
R, G, B = a[..., 0], a[..., 1], a[..., 2]
ink = (B - R > 40)                                   # blue crayon, not the paper and not the wooden table
ink = ndi.binary_opening(ink, iterations=1)
soft = np.clip((B - R - 18) / 45.0, 0, 1)               # the crayon's own texture, kept as soft alpha
rows_txt = ["אבגדהוזחטיכלמ", "נסעפצקרשת", "123456789"]
rtl = [True, True, False]
DIL = [4, 9, 9]; OVL = [.5, 0, 9]; MINA = [250, 250, 900]          # tuned per row: row 1 has a separate י next to ט, rows 2-3 have letters drawn in two strokes
# row bands from the vertical profile
prof = ink.sum(axis=1); on = prof > 25
bands, start = [], None
for y, v in enumerate(on):
    if v and start is None: start = y
    if not v and start is not None:
        if y - start > 40: bands.append((start, y))
        start = None
print("bands", bands)
cells, chars = [], []
CW, CH = 120, 160
for (y0, y1), txt, r2l, dil, ovl, mina in zip(bands, rows_txt, rtl, DIL, OVL, MINA):
    band = ink[max(0, y0 - 20):y1 + 20]
    lab, n = ndi.label(ndi.binary_dilation(band, iterations=dil))
    raw = [[sl[1].start, sl[1].stop, sl[0].start, sl[0].stop] for sl in ndi.find_objects(lab) if (sl[1].stop - sl[1].start) * (sl[0].stop - sl[0].start) > mina]
    # pieces of one letter sit above or inside each other: merge boxes that overlap a lot on the x axis
    raw.sort(key=lambda b: b[0]); boxes = []
    for b in raw:
        if boxes:
            p_ = boxes[-1]; ov = min(p_[1], b[1]) - max(p_[0], b[0])
            if ov > ovl * min(p_[1] - p_[0], b[1] - b[0]):
                boxes[-1] = [min(p_[0], b[0]), max(p_[1], b[1]), min(p_[2], b[2]), max(p_[3], b[3])]; continue
        boxes.append(b)
    boxes = sorted(boxes, key=lambda b: -b[0] if r2l else b[0])
    print(txt, len(txt), "found", len(boxes))
    yoff = max(0, y0 - 20)
    sc = (CH - 16) / (y1 - y0 + 40)                    # one scale per row, so small letters stay small
    for ch, (x0, x1, by0, by1) in zip(txt, boxes):
        g = soft[yoff + by0:yoff + by1, x0:x1] * ndi.binary_dilation(band[by0:by1, x0:x1], iterations=3)
        alpha = Image.fromarray((g * 255).astype(np.uint8))
        w, h = alpha.size
        alpha = alpha.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.LANCZOS)
        if alpha.width > CW - 8: alpha = alpha.resize((CW - 8, max(1, int(alpha.height * (CW - 8) / alpha.width))), Image.LANCZOS)
        cell = Image.new("L", (CW, CH), 0)
        cell.paste(alpha, ((CW - alpha.width) // 2, 8 + int((by0 - 20) * sc) if by0 > 20 else 8))
        cells.append(cell); chars.append(ch)
sheet = Image.new("RGBA", (CW * len(cells), CH), (255, 255, 255, 0))
for i, c in enumerate(cells):
    white = Image.new("RGBA", (CW, CH), (255, 255, 255, 255)); white.putalpha(c); sheet.paste(white, (i * CW, 0))
sheet.save("assets/hand-glyphs.png", optimize=True)
json.dump({"cw": CW, "ch": CH, "chars": "".join(chars)}, open("assets/hand-glyphs.json", "w"), ensure_ascii=False)
print(len(cells), "".join(chars))
