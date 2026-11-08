"""Third sheet (raw/handwriting/sentences-photo.jpg): short lines I wrote about myself in red crayon.
Each line (or two-line answer) is cut whole, as I wrote it, into assets/hand-lines.png + .json.
Run from the repo root."""
import json, numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage as ndi
SHEETS = [   # (photo, centres of the lines measured on a 1500px-wide preview, which answer each line belongs to, jar box)
    ("raw/handwriting/sentences-photo.jpg", [205, 325, 525, 665, 745, 885, 1020, 1160, 1280, 1400, 1545, 1660, 1770, 1880],
     [0, 1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11], (330, 900)),
    ("raw/handwriting/sentences-2-photo.jpg", [140, 280, 380, 545, 665, 830, 985, 1150, 1265, 1360, 1475, 1590, 1705],
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11], (0, 0)),   # the crossed-out words stay: they are part of what I wrote
    # answers sheet: green-blue marker on the back of a worksheet (its print shows through faintly); numbers on the right
    ("raw/handwriting/answers-photo.jpg", [325, 410, 565, 715, 870, 1050, 1240, 1360, 1500, 1610, 1720],
     [0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 6], (0, 0), "dark"),
]
masks_all = []
for path, CENTRES, GROUP, jar, *mode in SHEETS:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    a = np.asarray(im).astype(np.int16); R, G, B = a[..., 0], a[..., 1], a[..., 2]
    if mode:                                             # dark marker: darker than the paper around it, and not the red mark
        Lm = (R + G + B) / 3.0
        d = ndi.grey_closing(Lm, size=(45, 45)) - Lm      # the paper without the strokes, minus the strokes
        ink = ndi.binary_opening((d > 38) & ~(R - G > 40), iterations=1)
        ink[:int(150 * im.width / 1500)] = False; ink[int(1800 * im.width / 1500):] = False   # the page edges
        x1 = int(1185 * im.width / 1500); ink[:, x1:] = False   # the question numbers
        soft_d = np.clip((d - 22) / 55.0, 0, 1)
    else:
        ink = ndi.binary_opening((R - G > 45) & (R - B > 25), iterations=1)
    ink[:jar[0], :jar[1]] = False                        # a jar in the corner of the first photo
    if not mode: ink[:, 1700:] = False                   # the dashes I put before each answer
    soft = soft_d if mode else np.clip((R - G - 25) / 55.0, 0, 1)
    cy = np.array(CENTRES) * (im.width / 1500)
    lab, n = ndi.label(ndi.binary_dilation(ink, iterations=3))
    masks = [np.zeros(ink.shape, bool) for _ in range(max(GROUP) + 1)]
    for i, sl in enumerate(ndi.find_objects(lab)):
        comp = lab[sl] == i + 1
        if comp.sum() < 300: continue                 # crumbs of crayon
        ys = np.nonzero(comp)[0] + sl[0].start
        masks[GROUP[int(np.argmin(abs(cy - np.percentile(ys, 20))))]][sl] |= comp   # by its upper part: long descenders belong to the line above
    masks_all += [(m, soft, .36 if mode else .5) for m in masks]   # the marker letters are larger than the crayon ones
items, out = [], []
for g, (m, soft, sc) in enumerate(masks_all):
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    alpha = (soft * ndi.binary_dilation(m, iterations=2))[y0:y1, x0:x1]
    img = Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))
    s = sc; img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    out.append(img)
W = max(i.width for i in out); y = 0
sheet = Image.new("RGBA", (W, sum(i.height + 6 for i in out)), (255, 255, 255, 0))
for img in out:
    white = Image.new("RGBA", img.size, (255, 255, 255, 255)); white.putalpha(img)
    sheet.paste(white, (0, y)); items.append([0, y, img.width, img.height]); y += img.height + 6
sheet.save("assets/hand-lines-2.png", optimize=True)
json.dump(items, open("assets/hand-lines-2.json", "w"))
print(len(items), sheet.size, [i[2:] for i in items])
