# y-range of the panel material that lies above a knee wall: over every panel (convex hull of its points), minimise and
# maximise y subject to xa <= x <= xb and z <= ztop (barycentric LP); used to build the walls (make_geo.py) and to test them.
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.
import numpy as np
from scipy.optimize import linprog


def wall_y_range(slabs, xa, xb, ztop):
    lo, hi = np.inf, -np.inf
    for q in slabs:
        q = np.asarray(q, float)
        if q[:, 0].max() < xa or q[:, 0].min() > xb or q[:, 2].min() > ztop: continue
        n = len(q); A = np.vstack([q[:, 0], -q[:, 0], q[:, 2]]); b = [xb, -xa, ztop]
        for sgn in (1, -1):
            r = linprog(sgn * q[:, 1], A_ub=A, b_ub=b, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(0, None)] * n, method='highs')
            if r.status == 0:
                y = sgn * r.fun; lo = min(lo, y); hi = max(hi, y)
    return lo, hi


def z_min_over(slabs, xa, xb):
    """lowest point of the panel material with xa <= x <= xb"""
    zmin = np.inf
    for q in slabs:
        q = np.asarray(q, float)
        if q[:, 0].max() < xa or q[:, 0].min() > xb: continue
        n = len(q)
        r = linprog(q[:, 2], A_ub=np.vstack([q[:, 0], -q[:, 0]]), b_ub=[xb, -xa], A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(0, None)] * n, method='highs')
        if r.status == 0: zmin = min(zmin, r.fun)
    return zmin
