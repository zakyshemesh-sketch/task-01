"""The answer I wrote to the model's last question, "מי אתה?" (raw/handwriting/who-am-i-photo.jpg, blue marker).
Each line is cut whole into assets/hand-answer.png + .json, to be written out on the screen line by line.
(Second take: the first sheet had a spelling mistake, and my signature, which never goes online.) Run from the repo root."""
import json, numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage as ndi
im = ImageOps.exif_transpose(Image.open("raw/handwriting/who-am-i-photo.jpg")).convert("RGB")
a = np.asarray(im).astype(np.int16); R, G, B = a[..., 0], a[..., 1], a[..., 2]
k = im.width / 1500
ink = ndi.binary_opening(B - R > 45, iterations=1)
ink[int(1600 * k):] = False                              # below the last line
soft = np.clip((B - R - 30) / 60.0, 0, 1)
CENTRES = [160, 335, 440, 560, 650, 740, 860, 975, 1090, 1210, 1330, 1470]
cy = np.array(CENTRES) * k
lab, n = ndi.label(ndi.binary_dilation(ink, iterations=3))
masks = [np.zeros(ink.shape, bool) for _ in CENTRES]
for i, sl in enumerate(ndi.find_objects(lab)):
    comp = lab[sl] == i + 1
    if comp.sum() < 40: continue                     # keep the yods and the dots
    ys = np.nonzero(comp)[0] + sl[0].start
    masks[int(np.argmin(abs(cy - np.percentile(ys, 40))))][sl] |= comp
out = []
for m in masks:
    ys, xs = np.nonzero(m)
    al = (soft * ndi.binary_dilation(m, iterations=2))[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    img = Image.fromarray((al * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))
    out.append(img.resize((img.width // 2, img.height // 2), Image.LANCZOS))
W = max(i.width for i in out); y = 0; items = []
sheet = Image.new("RGBA", (W, sum(i.height + 6 for i in out)), (255, 255, 255, 0))
for img in out:
    wh = Image.new("RGBA", img.size, (255, 255, 255, 255)); wh.putalpha(img)
    sheet.paste(wh, (0, y)); items.append([0, y, img.width, img.height]); y += img.height + 6
sheet.save("assets/hand-answer.png", optimize=True)
json.dump(items, open("assets/hand-answer.json", "w"))
print(items)

# ---- v44: a time map, so each line is written the way a hand writes it ----
# For every line: strokes (connected pieces of ink) are taken from right to left; inside a stroke the pen
# follows the stroke's skeleton from its rightmost end. Every ink pixel gets the moment the pen passes it.
# Saved as assets/hand-answer-time.png, same layout as hand-answer.png: red = time (1..255), 0 = no ink.
from skimage.morphology import skeletonize
from collections import deque
atlas = np.asarray(Image.open("assets/hand-answer.png").getchannel("A")).astype(float) / 255
tmap = np.zeros(atlas.shape, np.uint8)
meta = []
for x0, y0, w, h in items:
    al = atlas[y0:y0 + h, x0:x0 + w]
    ink = al > .18
    lab, n = ndi.label(ndi.binary_dilation(ink, iterations=2))   # crayon breaks: a stroke is what touches
    comps = []
    for i in range(1, n + 1):
        m = (lab == i) & ink
        if m.sum() < 4: continue
        comps.append((np.nonzero(m)[1].max(), m))
    comps.sort(key=lambda c: -c[0])                      # right to left, like I write
    t_line = np.full(ink.shape, -1.0); t = 0.0
    for _, m in comps:
        sk = skeletonize(m)
        if sk.sum() == 0: sk = m
        ys, xs = np.nonzero(sk)
        start = int(np.argmax(xs - .3 * ys))             # the pen goes in at the stroke's top right
        dist = np.full(sk.shape, -1.0); q = deque([(ys[start], xs[start])]); dist[ys[start], xs[start]] = 0
        while q:                                         # walk along the skeleton
            y, x = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < sk.shape[0] and 0 <= xx < sk.shape[1] and sk[yy, xx] and dist[yy, xx] < 0:
                        dist[yy, xx] = dist[y, x] + (1.41 if dy and dx else 1); q.append((yy, xx))
        dist[sk & (dist < 0)] = dist.max() + 1           # a broken skeleton: the rest comes right after
        _, (iy, ix) = ndi.distance_transform_edt(~sk, return_indices=True)
        local = dist[iy, ix]
        t_line[m] = t + local[m]
        t = t_line[m].max() + 14                         # lifting the pen costs a little
    total = max(t, 1)
    tm = np.where(ink, 1 + np.clip(t_line, 0, None) / total * 254, 0)
    tmap[y0:y0 + h, x0:x0 + w] = tm.astype(np.uint8)
    meta.append(round(total, 1))                         # the length the pen travels, in pixels
Image.fromarray(np.dstack([tmap, tmap * 0, tmap * 0, (tmap > 0) * 255]).astype(np.uint8), "RGBA").save("assets/hand-answer-time.png", optimize=True)
json.dump({"lines": items, "travel": meta}, open("assets/hand-answer.json", "w"))
print("travel", meta)
