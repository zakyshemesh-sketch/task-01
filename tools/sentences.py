"""Third sheet (raw/handwriting/sentences-photo.jpg): short lines I wrote about myself in red crayon.
Each line (or two-line answer) is cut whole, as I wrote it, into assets/hand-lines.png + .json.
Run from the repo root."""
import json, numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage as ndi
im = ImageOps.exif_transpose(Image.open("raw/handwriting/sentences-photo.jpg")).convert("RGB")
a = np.asarray(im).astype(np.int16); R, G, B = a[..., 0], a[..., 1], a[..., 2]
ink = ndi.binary_opening((R - G > 45) & (R - B > 25), iterations=1)
ink[:330, :900] = False                                  # the jar in the corner
ink[:, 1700:] = False                                    # the dashes I put before each answer
soft = np.clip((R - G - 25) / 55.0, 0, 1)
k = 1932 / 1500                                          # centres measured on the preview, in full-size pixels
CENTRES = [205, 325, 525, 665, 745, 885, 1020, 1160, 1280, 1400, 1545, 1660, 1770, 1880]
GROUP = [0, 1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11]    # two-line answers stay together
cy = np.array(CENTRES) * k
lab, n = ndi.label(ndi.binary_dilation(ink, iterations=3))
masks = [np.zeros(ink.shape, bool) for _ in range(max(GROUP) + 1)]
for i, sl in enumerate(ndi.find_objects(lab)):
    comp = lab[sl] == i + 1
    if comp.sum() < 60: continue
    ys = np.nonzero(comp)[0] + sl[0].start
    g = GROUP[int(np.argmin(abs(cy - ys.mean())))]
    masks[g][sl] |= comp
items, out = [], []
for g, m in enumerate(masks):
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    alpha = (soft * ndi.binary_dilation(m, iterations=2))[y0:y1, x0:x1]
    img = Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))
    s = 0.5; img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    out.append(img)
W = max(i.width for i in out); y = 0
sheet = Image.new("RGBA", (W, sum(i.height + 6 for i in out)), (255, 255, 255, 0))
for img in out:
    white = Image.new("RGBA", img.size, (255, 255, 255, 255)); white.putalpha(img)
    sheet.paste(white, (0, y)); items.append([0, y, img.width, img.height]); y += img.height + 6
sheet.save("assets/hand-lines.png", optimize=True)
json.dump(items, open("assets/hand-lines.json", "w"))
print(len(items), sheet.size, [i[2:] for i in items])
