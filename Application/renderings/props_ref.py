# Reference layout of the schematic props (containers and trailers) and its comparison with a scene dump. The reference
# (reference/props_reference.npz) holds, for one container and one trailer of the printed renderings, every part's vertices
# relative to the unit's anchor (x: centre of the unit's bounding box, y: its minimum, z: 0), sorted, with its triangle count.
# A scene passes only if every unit in it has exactly these parts, with the same shape and size at the same place (1e-5 m),
# and nothing else is in the scene (make_reference writes a new reference).
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.
import os, json
import numpy as np

CONTAINER_BASES = ('c_cast', 'c_cross', 'c_doorpost', 'c_endrail', 'c_floorplate', 'c_header', 'c_post', 'c_roof', 'c_siderail',
                   'c_toprail_x', 'c_toprail_y', 'c_wall_l', 'c_wall_r', 'c_wall_rear')
TRUCK_BASES = ('deck', 'wellbeam', 'wellfloor', 'rearbeam', 'light', 'mudflap', 'neck', 'gooseneck', 'kingpin', 'rearstep', 'reardeck', 'tyre', 'rim', 'hanger', 'axle', 'leg', 'leg_pad', 'underrun',
               'underrun_strut')
TOL = 1e-5


def base(n): return n.split('.')[0]


def units(Z, names, bases):
    """the props of one kind, clustered along x (units at least 1 m apart): list of lists of object names"""
    objs = sorted([n for n in names if base(n) in bases], key=lambda n: Z[n + '__V'][:, 0].min())
    groups, hi = [], -np.inf
    for n in objs:
        lo_, hi_ = Z[n + '__V'][:, 0].min(), Z[n + '__V'][:, 0].max()
        if not groups or lo_ > hi + 1.0: groups.append([n])
        else: groups[-1].append(n)
        hi = max(hi, hi_) if groups[-1] != [n] else hi_
    return groups


def anchor(Z, grp):
    V = np.concatenate([Z[n + '__V'] for n in grp])
    return np.array([0.5 * (V[:, 0].min() + V[:, 0].max()), V[:, 1].min(), 0.0])


def signature(Z, n, a):
    V = np.asarray(Z[n + '__V'], float) - a
    V = V[np.lexsort((V[:, 2], V[:, 1], V[:, 0]))]
    return V, int(len(Z[n + '__F']))


def make_reference(dumps, out):
    """dumps: {'container': (npz path), 'truck': (npz path)}; writes the reference of the first unit of each kind"""
    R = {}
    for kind, path in dumps.items():
        Z = np.load(path); names = sorted({k[:-3] for k in Z.files})
        grp = units(Z, names, CONTAINER_BASES if kind == 'container' else TRUCK_BASES)[0]; a = anchor(Z, grp)
        for k, n in enumerate(sorted(grp)):
            V, nf = signature(Z, n, a)
            R['%s|%s|%d|V' % (kind, base(n), k)] = V; R['%s|%s|%d|F' % (kind, base(n), k)] = np.array(nf)
    np.savez_compressed(out, **R)


def load_reference(path):
    R = np.load(path); ref = {}
    for k in R.files:
        kind, b, i, what = k.split('|')
        ref.setdefault(kind, {}).setdefault((b, int(i)), {})[what] = R[k]
    return {kind: [(b, d['V'], int(d['F'])) for (b, i), d in sorted(v.items())] for kind, v in ref.items()}


def open_edges(F):
    from collections import Counter
    e = Counter(tuple(sorted((int(f[i]), int(f[(i + 1) % 3])))) for f in F for i in range(3))
    return sum(1 for c in e.values() if c != 2)


def compare_unit(Z, grp, ref_parts):
    """every reference part matched by exactly one part of the unit (same base, same triangle count, vertices within TOL),
    every part of the unit used, every part a closed mesh.  Returns (problems, anchor)"""
    a = anchor(Z, grp); probs = []
    left = list(grp)
    for b, Vr, nfr in ref_parts:
        best, bd = None, np.inf
        for n in left:
            if base(n) != b: continue
            V, nf = signature(Z, n, a)
            if nf != nfr or V.shape != Vr.shape: continue
            # vertex sets compared without relying on their order (sorting is unstable for ties such as a cylinder's cap)
            from scipy.spatial import cKDTree
            d = float(max(cKDTree(Vr).query(V, p=np.inf)[0].max(), cKDTree(V).query(Vr, p=np.inf)[0].max()))
            if d < bd: best, bd = n, d
        if best is None or bd > TOL:
            probs.append('%s: no part matches the reference (closest %s, %.2e m)' % (b, best, bd))
        else:
            left.remove(best)
    for n in left: probs.append('%s: not in the reference (moved, resized, reshaped or extra)' % n)
    for n in grp:
        if open_edges(Z[n + '__F']): probs.append('%s: mesh not closed' % n)
    return probs, a
