# Scans a plain RGBA rendering for denoiser block artefacts on the shadow-catcher ground: tiles of 24 px that lie entirely on
# the ground (away from every object), mean absolute second difference of the alpha along columns and rows (x 255).
# usage: python3 renderings/artifact_scan.py <render.png> [...]      requires numpy, scipy, pillow
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.
import sys
import numpy as np
from PIL import Image
from scipy import ndimage
T = 24
import os
for f in sys.argv[1:]:
    a = np.asarray(Image.open(f).convert('RGBA')).astype(float)[..., 3] / 255.0
    mf = os.path.splitext(f)[0] + '_mask.png'
    m = np.asarray(Image.open(mf).convert('RGBA')).astype(float)[..., 3] / 255.0 if os.path.exists(mf) else (a > 0.97)
    obj = ndimage.binary_dilation(m > 0.02, iterations=8)
    dv = np.zeros_like(a); dh = np.zeros_like(a)
    dv[1:-1, :] = np.abs(a[2:, :] - 2 * a[1:-1, :] + a[:-2, :])
    dh[:, 1:-1] = np.abs(a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2])
    H, W = a.shape; rows = []
    for y in range(0, H - T, T):
        for x in range(0, W - T, T):
            if obj[y:y + T, x:x + T].any(): continue
            v = 255 * dv[y:y + T, x:x + T].mean(); h = 255 * dh[y:y + T, x:x + T].mean()
            g = np.diff(a[y:y + T, x:x + T], axis=0); sg = np.sign(np.where(np.abs(g) > 1.5 / 255, g, 0))
            nz = sg != 0; flips = 0.0
            for col in range(T):
                c = sg[:, col][nz[:, col]]; flips += np.sum(c[1:] != c[:-1]) if len(c) > 1 else 0
            rows.append((x, y, v, h, flips / T))
    R = np.array(rows)
    from scipy.ndimage import median_filter
    hp = a - median_filter(a, size=3); blot = []          # noise residual: a median filter keeps edges, removes noise
    for (x, y, *_r) in rows:
        blot.append(255 * np.sqrt(np.mean(hp[y:y + T, x:x + T] ** 2)))
    blot = np.array(blot); bmed = np.median(blot)
    sd = np.array([255 * a[int(r_[1]):int(r_[1]) + T, int(r_[0]):int(r_[0]) + T].std() for r_ in rows])
    cand = (blot > max(0.6, 3.5 * bmed)) & (sd < 8)           # noisy but smooth tile (a shadow edge has a large spread)
    pos = {(int(rows[k][0]), int(rows[k][1])) for k in np.where(cand)[0]}
    blotch = np.array([cand[k] and any((int(rows[k][0]) + dx, int(rows[k][1]) + dy) in pos for dx, dy in ((T, 0), (-T, 0), (0, T), (0, -T)))
                       for k in range(len(rows))], bool)        # at least two adjacent tiles
    # denoiser streaks: strong vertical second difference with weak horizontal one (horizontal stripes); a shadow edge has both
    streak = (R[:, 2] > 1.5) & (R[:, 2] > 4 * np.maximum(R[:, 3], 0.05)) & (R[:, 4] > 3.0)   # many alternations along each column
    print('%s: ground tiles %d, median %.2f, max %.2f, streak tiles %d %s, blotch tiles %d (hp rms median %.2f, max %.2f) %s' % (
          f, len(R), np.median(R[:, 2] + R[:, 3]), (R[:, 2] + R[:, 3]).max(), int(streak.sum()),
          [(int(r[0]), int(r[1]), round(r[2], 1), round(r[3], 1), round(r[4], 1)) for r in R[streak][:6]],
          int(blotch.sum()), bmed, blot.max(), [(int(R[k, 0]), int(R[k, 1]), round(blot[k], 2)) for k in np.where(blotch)[0][:6]]))
