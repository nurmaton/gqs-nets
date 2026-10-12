# Shared definitions for the three arch halls (i), (ii), (iii): the net data (data/halls.json), the construction of the
# hall faces with ../gqs.py, the standing frame of a position, the midline, the profile length and the period-two translation.
# The checks in checks/ and the figure scripts import this module. Requires numpy (and gqs.py: numpy, matplotlib).
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, json, math, importlib.util

sys.dont_write_bytecode = True                       # no __pycache__ next to ../gqs.py or in this folder
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
OUT = os.environ.get('APP_OUT', os.path.join(ROOT, 'out'))
GQS_PATH = os.environ.get('GQS_PY', os.path.join(ROOT, '..', 'gqs.py'))
GQS_MD5 = '5b67bf1f19d85f93f47e72181c81955e'          # the gqs.py these scripts were run with
HALL_KEYS = ('i', 'ii', 'iii')
STATES = ('deployed', 'transport')

with open(os.path.join(DATA, 'halls.json')) as _fh:
    SPEC = json.load(_fh)
HALLS = SPEC['halls']

_G = []


def gqs():
    """the module ../gqs.py, loaded read-only (no bytecode written); its interactive part is not started"""
    if not _G:
        path = os.path.abspath(GQS_PATH)
        if not os.path.exists(path):
            sys.exit('gqs.py not found at %s (set GQS_PY to its path)' % path)
        spec = importlib.util.spec_from_file_location('gqs_supplement', path)
        G = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(G)
        _G.append(G)
    return _G[0]


def gqs_md5():
    import hashlib
    with open(os.path.abspath(GQS_PATH), 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


# ------------------------------------------------------------------------------------------------ the net and its faces
def window(h):
    """vertex index ranges (i0, i1, j0, j1) of the hall: the corners of the faces F_ij, faces_i[0] <= i <= faces_i[1], ..."""
    H = HALLS[h]
    return H['faces_i'][0], H['faces_i'][1] + 1, H['faces_j'][0], H['faces_j'][1] + 1


def face_labels(h):
    """the faces F_ij of the hall in the order of the panel arrays (i-major)"""
    H = HALLS[h]
    return [(i, j) for i in range(H['faces_i'][0], H['faces_i'][1] + 1) for j in range(H['faces_j'][0], H['faces_j'][1] + 1)]


def hinge_list(h):
    """the common edges of two faces of the hall, as ((i, j), (i', j')) pairs of vertex labels"""
    i0, i1, j0, j1 = window(h)
    out = []
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            if i0 < i < i1 and j < j1: out.append(((i, j), (i, j + 1)))          # edge between F_{i-1,j} and F_ij
            if j0 < j < j1 and i < i1: out.append(((i, j), (i + 1, j)))          # edge between F_{i,j-1} and F_ij
    return out


class Net:
    """an m x n net of gqs.py for one hall: flat angles assembled once, positions at any t (gqs.py units)"""

    def __init__(self, h, rows=None):
        G = gqs()
        H = HALLS[h]
        self.h, self.G = h, G
        self.ang_deg = tuple(float(a) for a in H['angles_deg'])
        self.m, self.n, self.base = H['m'], H['n'], tuple(H['base'])
        self.signs, self.e0 = tuple(H['signs']), H['e0']
        self.rows = rows if rows is not None else H['rows']
        assert len(self.rows) == self.n - 2, (len(self.rows), self.n)
        rd = G.check_parameters(self.m, self.n, self.base, {k + 1: r for k, r in enumerate(self.rows)}, verbose=False)
        sub = G.assemble_net(self.ang_deg, self.m, self.n, self.base, rd)
        self.ang = G.vertex_face_angles(sub)
        self.subs = {kn: G.Block(kn[0], kn[1], q) for kn, q in sub.items()}
        # the free lengths of the boundary faces are recorded once, at the parameter where gqs.py starts (as its window does);
        # the hall uses no boundary face, so they do not enter the hall
        lo, hi = G.admissible_t_range(self.subs[self.base])
        t0 = None
        for f_ in (0.15, 0.05, 0.25, 0.35, 0.5, 0.65, 0.8, 0.92):
            if G.synchronize(self.subs, self.base, self.signs, self.e0, lo + f_ * (hi - lo)) is not None:
                t0 = lo + f_ * (hi - lo); break
        assert t0 is not None, 'no admissible start parameter'
        th0 = G.synchronize(self.subs, self.base, self.signs, self.e0, t0)
        _, _, _, self.free = G.build_net(self.ang, th0, self.m, self.n, self.base, G.BASE_EDGE, None)

    def positions(self, t):
        """vertex positions {(i, j): xyz} of the whole net at t and the largest dihedral-angle mismatch of the build, or None"""
        G = self.G
        th = G.synchronize(self.subs, self.base, self.signs, self.e0, t)
        if th is None: return None
        pos, worst, _, _ = G.build_net(self.ang, th, self.m, self.n, self.base, G.BASE_EDGE, self.free)
        return pos, worst

    def hall_grid(self, t):
        """the hall vertices V_ij as an array (i - i0, j - j0, 3), gqs.py units, and the build mismatch"""
        r = self.positions(t)
        if r is None: return None
        pos, worst = r
        i0, i1, j0, j1 = window(self.h)
        return np.array([[pos[(i, j)] for j in range(j0, j1 + 1)] for i in range(i0, i1 + 1)], float), worst


def theta1_deg(t):
    return math.degrees(2 * math.atan2(1.0, t))


# ------------------------------------------------------------------------------------------------ geometry helpers
def kabsch(A, B):
    """proper rotation R and shift c minimising |A R^T + c - B| (rows are points); returns R, c, largest residual"""
    A = np.asarray(A, float).reshape(-1, 3); B = np.asarray(B, float).reshape(-1, 3)
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    c = cb - R @ ca
    return R, c, float(np.linalg.norm(A @ R.T + c - B, axis=1).max())


def period_two(V):
    """mean of V_{i+2,j} - V_ij over all i, j, and the largest deviation of a single difference from it"""
    D = (V[2:] - V[:-2]).reshape(-1, 3)
    T = D.mean(0)
    return T, float(np.linalg.norm(D - T, axis=1).max())


def midline3(V):
    """per vertex column j: the mean of the polyline of the even-i vertices and that of the odd-i vertices (3D)"""
    return (V[0::2].mean(0) + V[1::2].mean(0)) / 2


def standing_rotation(V, x_sign=1):
    """rows x, y, z of the standing frame of a position V (any frame): y along the period-two translation, x along the
    foot chord of the midline (made orthogonal to y, times x_sign), z up (towards the midline), y = z x x"""
    T, _ = period_two(V)
    y0 = T / np.linalg.norm(T)
    mid = midline3(V)
    ch = mid[-1] - mid[0]; ch = ch - (ch @ y0) * y0
    x0 = ch / np.linalg.norm(ch)
    z0 = np.cross(x0, y0)
    up = 1.0 if np.mean((mid - mid[0]) @ z0) > 0 else -1.0
    z = up * z0; x = x_sign * x0; y = np.cross(z, x)
    return np.array([x, y, z])


def to_standing(V, panels, h, state):
    """V (ni, nj, 3) and panels (np, k, 3) in one frame -> both in the standing frame of the hall (see data/halls.json)"""
    R = standing_rotation(V, HALLS[h]['frame_x_sign'])
    Vs = V @ R.T; Ps = None if panels is None else panels @ R.T
    pts = Vs.reshape(-1, 3) if Ps is None else np.vstack([Vs.reshape(-1, 3), Ps.reshape(-1, 3)])
    lo, hi = pts.min(0), pts.max(0)
    if state == 'deployed':
        mid = midline3(Vs)
        shift = -np.array([(mid[0, 0] + mid[-1, 0]) / 2, lo[1], (mid[0, 2] + mid[-1, 2]) / 2])
    else:
        shift = -np.array([(lo[0] + hi[0]) / 2, lo[1], lo[2]])
    return Vs + shift, (None if Ps is None else Ps + shift), R, shift


def midline(Vs):
    """the midline in the standing frame: (nj, 2) array of (x, z)"""
    return midline3(Vs)[:, [0, 2]]


def polyline_length(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def face_quads(V):
    """(nfaces, 4, 3): corners V_ij, V_{i+1,j}, V_{i+1,j+1}, V_{i,j+1} of every face, i-major (the panel order)"""
    ni, nj = V.shape[:2]
    return np.array([[V[i, j], V[i + 1, j], V[i + 1, j + 1], V[i, j + 1]] for i in range(ni - 1) for j in range(nj - 1)])


# ------------------------------------------------------------------------------------------------ stored data
def load_V(h, state):
    return np.load(os.path.join(DATA, 'hall_%s_%s_V.npy' % (h, state)))


def load_panels(h, state):
    """(npanels, k, 3): points whose convex hull is the 80 mm panel of each face (repeated points pad to a common k)"""
    return np.load(os.path.join(DATA, 'hall_%s_%s_panels.npy' % (h, state)))


def t_of(h, state):
    return HALLS[h]['t_%s' % state]


# ------------------------------------------------------------------------------------------------ results
def write_result(name, obj):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + '.json')

    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=1, default=conv)
    return path


def header(title):
    print('=' * 100)
    print(title)
    print('=' * 100)
