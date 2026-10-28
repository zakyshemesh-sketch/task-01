import sys, numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi
src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src).convert('RGB')
a = np.asarray(im).astype(np.float32)
h, w, _ = a.shape
# sample the backdrop from the four borders, then measure distance to it
border = np.concatenate([a[:40].reshape(-1,3), a[-40:].reshape(-1,3), a[:, :40].reshape(-1,3), a[:, -40:].reshape(-1,3)])
bg = np.median(border, axis=0)
lum = a.mean(axis=2); bgl = bg.mean()
# backdrop = close to backdrop colour AND not much brighter than it (paper in shade stays paper)
d = np.sqrt(((a - bg) ** 2).sum(axis=2))
isbg = (d < 55) & (lum < bgl + 45)
isbg = ndi.binary_opening(isbg, iterations=2)
lab, n = ndi.label(isbg)
sizes = ndi.sum(isbg, lab, range(1, n + 1))
keep = np.isin(lab, 1 + np.where(sizes > h * w * 0.0015)[0])   # drop specks inside paper
paper = ~keep
paper = ndi.binary_fill_holes(ndi.binary_closing(paper, iterations=1)) & ~keep | paper
pl, pn = ndi.label(paper)
psz = ndi.sum(paper, pl, range(1, pn + 1))
paper = np.isin(pl, 1 + np.where(psz > h * w * 0.01)[0])        # drop backdrop specks read as paper
m = Image.fromarray((paper * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
rgba = im.copy(); rgba.putalpha(m); rgba.save(dst)
print(dst, 'bg', bg.round(), f"paper={paper.mean():.2%}")
