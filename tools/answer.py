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
