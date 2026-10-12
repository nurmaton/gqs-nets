# Clear height under the panels of a deployed hall, seen along the rows of blocks (standing frame of halls.py).
# Used by checks/check_clear_height.py and figures/sections.py. Requires numpy and scipy.
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import numpy as np
from scipy.spatial import ConvexHull

DX = 0.0005          # m: step of the x grid


def shadows(Ps):
    """convex polygons (x, z) of the panels seen along the rows (the standing frame's y)"""
    out = []
    for q in Ps:
        Q = np.unique(np.round(q[:, [0, 2]], 12), axis=0)
        out.append(Q[ConvexHull(Q).vertices])
    return out


def grid(polys):
    allx = np.concatenate([q[:, 0] for q in polys])
    return np.round(np.arange(np.floor(allx.min() / DX) * DX, allx.max() + DX, DX), 7)


def vertical_extents(polys, xg):
    """per polygon, the lowest and highest z at each x of the grid (nan where the vertical line misses it)"""
    LO = np.full((len(polys), len(xg)), np.nan); HI = np.full((len(polys), len(xg)), np.nan)
    for p, Q in enumerate(polys):
        Qc = np.vstack([Q, Q[:1]])
        for a, b in zip(Qc[:-1], Qc[1:]):
            x1, x2 = sorted((a[0], b[0]))
            sel = (xg >= x1) & (xg <= x2)
            if not sel.any() or abs(b[0] - a[0]) < 1e-12: continue
            z = a[1] + (b[1] - a[1]) * ((xg[sel] - a[0]) / (b[0] - a[0]))
            LO[p, sel] = np.fmin(LO[p, sel], z); HI[p, sel] = np.fmax(HI[p, sel], z)
    return LO, HI


def knee_walls(Ps, thick=0.08):
    """x ranges of the knee walls: thick wide, centred under the lowest panel corner of each side"""
    flat = Ps.reshape(-1, 3); out = []
    for sgn in (-1, 1):
        side = flat[sgn * flat[:, 0] > 0]
        xl = float(side[np.argmin(side[:, 2]), 0])
        out.append((xl - thick / 2, xl + thick / 2))
    return out


def clear_curve(LO, HI, xg, hw, walls=()):
    """h(x) above the floor with the foot line hw above it: the lowest panel material above the floor (0 where material, a
    plinth or knee wall reaches the floor); walls: (xa, xb) ranges of the plinths or knee walls, where h = 0 when hw > 0"""
    lo = LO + hw; hi = HI + hw
    h = np.where(np.isnan(lo), np.inf, np.where(hi < 0, np.inf, np.maximum(lo, 0.0))).min(0)
    if hw > 0:
        for xa, xb in walls:
            h = np.where((xg >= xa) & (xg <= xb), 0.0, h)
    return h


def runs(ok):
    """(start, end) index pairs of the runs of True"""
    r = []; k = 0
    while k < len(ok):
        if ok[k]:
            a = k
            while k + 1 < len(ok) and ok[k + 1]: k += 1
            r.append((a, k))
        k += 1
    return r


def strips(h, xg, level):
    """width of the widest run with h >= level, the total width of all runs, and the x range of the widest"""
    rr = runs(h >= level - 1e-12)
    widths = [(b - a + 1) * DX for a, b in rr]
    if not widths: return 0.0, 0.0, None
    k = int(np.argmax(widths)); a, b = rr[k]
    return float(max(widths)), float(sum(widths)), (float(xg[a] - DX / 2), float(xg[b] + DX / 2))


def all_strips(h, xg, level):
    return [(float(xg[a] - DX / 2), float(xg[b] + DX / 2)) for a, b in runs(h >= level - 1e-12)]
