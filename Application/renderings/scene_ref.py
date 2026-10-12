# Whole-scene reference of each rendering: every dumped object with all its triangles (and their orientation), and the
# scene meta. A dump passes only if it has exactly the reference objects, each with the same triangles (every vertex within
# TOL, same winding), and the same meta numbers: this locks the complete picture geometry (panels, person, walls, floor,
# containers, trailers, their places and the layout) to the scenes of the printed renderings (make_reference writes one).
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.
import os, json
import numpy as np
from scipy.spatial import cKDTree
TOL = 1e-5


def tris(Z, n):
    V = np.asarray(Z[n + '__V'], float); F = np.asarray(Z[n + '__F'], int)
    return V[F]                                                    # (m, 3, 3)


def make_reference(dump, out):
    Z = np.load(dump); meta = json.load(open(os.path.splitext(dump)[0] + '.meta.json'))
    names = sorted({k[:-3] for k in Z.files})
    np.savez_compressed(out, **{n: tris(Z, n) for n in names}, __meta__=np.array(json.dumps(meta, sort_keys=True)))


def _num_diff(a, b, path=''):
    """differences between two json-like values (numbers within 1e-9)"""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b: out.append(path + '.' + str(k))
            else: out += _num_diff(a[k], b[k], path + '.' + str(k))
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b): return [path]
        return [q for i, (x, y) in enumerate(zip(a, b)) for q in _num_diff(x, y, '%s[%d]' % (path, i))]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return [] if abs(a - b) <= 1e-9 else [path]
    return [] if a == b else [path]


def same_triangles(A, B):
    """A, B: (m, 3, 3).  True if there is a one-to-one matching of triangles with every vertex within TOL (any vertex order)
    and the same winding (normals in the same direction).  Returns (ok, worst distance, reason)"""
    if A.shape != B.shape: return False, np.inf, 'triangle count %d, reference %d' % (len(B), len(A))
    if len(A) == 0: return True, 0.0, ''
    ca, cb = A.mean(1), B.mean(1)
    tree = cKDTree(cb); used = np.zeros(len(B), bool); worst = 0.0
    for i in range(len(A)):
        cand = tree.query_ball_point(ca[i], r=TOL * 2, p=np.inf)
        best = None
        for j in cand:
            if used[j]: continue
            d = np.abs(A[i][:, None, :] - B[j][None, :, :]).max(-1)        # 3 x 3 vertex distances
            if d.min(1).max() <= TOL and d.min(0).max() <= TOL:
                na = np.cross(A[i][1] - A[i][0], A[i][2] - A[i][0]); nb = np.cross(B[j][1] - B[j][0], B[j][2] - B[j][0])
                if np.dot(na, nb) < 0 and np.linalg.norm(na) > 1e-12: continue          # flipped triangle
                best = j; worst = max(worst, float(d.min(1).max())); break
        if best is None: return False, np.inf, 'triangle %d of the reference has no match (moved, reshaped, flipped or re-triangulated)' % i
        used[best] = True
    return True, worst, ''


def compare(dump, ref_path):
    """returns (problems, worst vertex distance)"""
    Z = np.load(dump); meta = json.load(open(os.path.splitext(dump)[0] + '.meta.json'))
    R = np.load(ref_path); probs = []; worst = 0.0
    rmeta = json.loads(str(R['__meta__']))
    md = _num_diff(rmeta, meta)
    if md: probs.append('scene meta differs from the reference: %s' % md[:6])
    names = sorted({k[:-3] for k in Z.files}); rnames = sorted(k for k in R.files if k != '__meta__')
    miss = [n for n in rnames if n not in names]; extra = [n for n in names if n not in rnames]
    if miss: probs.append('objects missing: %s' % miss[:6])
    if extra: probs.append('objects not in the reference: %s' % extra[:6])
    for n in rnames:
        if n not in names: continue
        ok, w, why = same_triangles(np.asarray(R[n], float), tris(Z, n))
        if not ok: probs.append('%s: %s' % (n, why))
        else: worst = max(worst, w)
    return probs, worst
