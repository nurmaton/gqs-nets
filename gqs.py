# Interactive visualization of flexible m x n nets assembled from 3 x 3 GQS-nets.
#
# Input: the flat angles (alpha_1, beta_1, gamma_1, delta_1) in degrees at the vertex A_1 of the base 3 x 3 block, the one with the
#        central face F_{kappa_0 nu_0}; the net size m x n; the position (kappa_0, nu_0) of the base block; the type ('a'..'d', see
#        RELATION_TEXT) of each row of blocks; the signs (e1, e2, e3, e4) of every block and the sign e0 of the base block.
# The flat angles of the whole net are assembled from the base block by the relations between the flat angles of adjacent
# blocks; the dihedral angles are given by the explicit flexion formulas, with the same signs (e1, e2, e3, e4) for every block,
# (1, -1, 1, -1) or (1, -1, -1, 1). The base block is taken at the parameter t and the sign e0; then the other blocks of its row,
# and then those of every column, are each realized from a realized neighbour with the parameter and the sign for which the two
# agree on their common faces (in the numbering of the pair: the same parameter, and signs related by a factor +-1), and every
# two blocks with common faces are checked to agree. The central faces have their sides closed row by row (the first left side
# of each row large enough for all sides to be positive), the other faces are convex with short edges to the boundary, and the
# net is built in space face by face and drawn with a slider for the dihedral angle theta_1 of the base block
# (t = cot(theta_1/2) is the flexion parameter of the formulas).
#
# Dihedral angles are oriented: -pi (equivalently +pi; angles are compared modulo 2 pi) when adjacent faces are coplanar,
# positive when the face bends towards the side of the normal of the central face (Bricard variables cot(theta/2)).
# Requires numpy and matplotlib only.
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import math, itertools, sys, json, re, textwrap, os, shutil, glob
import numpy as np
import matplotlib
def _version_tuple(v): return tuple(int(x) for x in re.findall(r'\d+', v)[:2])
if _version_tuple(matplotlib.__version__) < (3, 7):
    sys.exit("This program needs matplotlib 3.7 or newer (found %s): pip install -U matplotlib" % matplotlib.__version__)
if _version_tuple(np.__version__) < (1, 20):
    sys.exit("This program needs numpy 1.20 or newer (found %s): pip install -U numpy" % np.__version__)
for _stream in (sys.stdout, sys.stderr):                                    # the report and the messages contain non-ASCII characters
    try: _stream.reconfigure(encoding='utf-8', errors='replace')            # (a Windows console with a legacy code page would otherwise raise)
    except Exception: pass
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'          # LaTeX-like indices and symbols in all labels
from matplotlib.widgets import Slider, TextBox as _TextBox
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

PI = math.pi

# ============================================================================================
# 1. PARAMETERS
# ============================================================================================
BASE_ANGLES_DEG = (60.0, 80.0, 120.0, 60.0)   # (alpha_1, beta_1, gamma_1, delta_1) at the vertex A_1 of the base block
M_FACES, N_FACES = 5, 5                        # m x n: the net has the faces F_ij, 0 <= i < m = M_FACES, 0 <= j < n = N_FACES
MAX_MN = 60                                    # the largest m and n accepted (here and in Params)
MAX_FACES = 400                                # ... and the largest m n: the startup builds ~200 configurations, ~1 min of CPU at 400 faces (10 x 26: ~35 s)
BASE_KN = (2, 2)                               # (kappa_0, nu_0): the base block has the central face F_{kappa_0 nu_0} (1 <= kappa_0 <= m-2, 1 <= nu_0 <= n-2)
ROW_TYPES = {1: 'c', 2: 'a', 3: 'b'}           # type ('a'..'d', see RELATION_TEXT) of the row of 3 x 3 blocks with central faces F_{. nu},
                                               # nu = 1 .. n-2 (rows not listed get the type of the row nu_0)
SIGN_PATTERNS = ((1, -1, 1, -1), (1, -1, -1, 1))   # the two choices of the signs (e1, e2, e3, e4)
SIGNS = (1, -1, 1, -1)                         # the signs of the flexion of every block (Params)
E0 = 1                                         # the sign e0 = +1 or -1 of the base block (Params)
BASE_EDGE = 1.0                                # length of the edge V_{kappa+1,nu} V_{kappa nu} of the base central face
SHOW_SHADOW = True                             # soft ground shadow (button Shadow)
LIGHT_FRAME = 'studio'                         # 'eye': a flashlight in the viewer's hand (shadow behind the net: floor from above, ceiling from below);
                                               # 'top': the sun at the top of the screen (shadow always straight below the net on the screen);
                                               # 'studio': the original setup (shading light attached to the viewer, shadow on the room's floor);
                                               # 'world': the sun at noon fixed in space (vertical footprint on the floor);
                                               # 'room': a photo studio: the lamps are fixed in the room and only the photographer moves; the net stands on
                                               #         its base face F_kn on an invisible stand (see ROOM_*). Buttons Eye | Top | Studio | World | Room
ROOM_LAMP_VIEW = (30.0, -60.0)                 # 'room': the lamps are set up for this view (elev, azim) and stay there
ROOM_CLEARANCE, ROOM_MIN_HEIGHT = 0.06, 0.25   # 'room': the stand keeps the floor this far under the deepest point of the whole flexion, at least this high (net sizes)
FOCAL_LENGTH = 0.4                             # perspective strength (matplotlib focal length; smaller = stronger; None = orthographic)
NSUB = 8                                       # each face is split into NSUB x NSUB pieces for the painter's depth sorting
LIGHT_FACES = 150                              # nets with more faces are drawn in the light mode (plain depth sorting, no hidden lines, no intersection test)
STRIP_CHOICES = 8                              # alternatives per boundary face in the backtracking construction of the boundary strips
STRIP_BUDGET = 400                             # attach3 attempts per strip in that backtracking; when they are used up, the strip is built in 'apex' mode
LIVE_PEN_FACES = 49                            # nets with at most this many faces (7 x 7) get the self-intersection test also while the slider is dragged;
                                               # larger nets (up to LIGHT_FACES) get it when the slider is released
CACHE_SIZE = 1000                              # configurations remembered per half of the slider (one per 1e-6 of t; ~13 KB each for 5 x 5, ~55 KB for 10 x 10)
                                               # besides the sampling grids, which are always kept; beyond it the oldest are dropped (and rebuilt when needed)
SAVE_FULL_WINDOW = False                       # buttons SVG/PNG: False = the picture only (cropped, no panels/controls), True = the whole window
PNG_DPI = 300                                  # resolution of the saved PNG
MOTION_FRAMES = 120                            # frames of the motion saved by the OBJ-sequence button (self-intersecting ones are skipped)
CHECK_TOL = 1e-8                               # a saved Motion frame must pass every check from its vertices within this tolerance (edge lengths: times the
                                               # longest edge; angles: at least 1e-10 x longest/shortest edge, the rounding level of nets with short edges)
BLENDER_PATH = ""                             # the Blender binary for the Render button. Empty: found automatically ('blender' on the PATH, then the usual
                                               # install folders on macOS, Windows and Linux, newest version first). Set it to pin a particular binary (one that is not
                                               # found is reported, no other is used), e.g. "/Applications/Blender 4.2.app/Contents/MacOS/Blender" or r"C:\Tools\blender\blender.exe".
def find_blender():
    """the Blender binary: BLENDER_PATH (None when it is set but not found: no other install is used then), else 'blender' on the PATH,
    else the usual install locations (newest version first: the version in the app's Info.plist on macOS, else in the folder name)"""
    if BLENDER_PATH: return BLENDER_PATH if os.path.isfile(BLENDER_PATH) else shutil.which(BLENDER_PATH)
    for name in ('blender', 'blender.exe'):
        found = shutil.which(name)
        if found: return found
    patterns = [r'C:\Program Files\Blender Foundation\Blender*\blender.exe', os.path.expanduser(r'~\AppData\Local\Programs\Blender Foundation\Blender*\blender.exe'),
                '/Applications/Blender*.app/Contents/MacOS/Blender', os.path.expanduser('~/Applications/Blender*.app/Contents/MacOS/Blender'),
                '/usr/bin/blender', '/usr/local/bin/blender', '/snap/bin/blender', '/opt/blender*/blender', os.path.expanduser('~/blender*/blender'),
                '/var/lib/flatpak/exports/bin/org.blender.Blender', os.path.expanduser('~/.local/share/flatpak/exports/bin/org.blender.Blender')]
    hits = [h for pat in patterns for h in glob.glob(pat) if os.path.isfile(h)]
    def version(h):                                                             # (5, 2, 1); () when unknown (ranks last). Not the digits of the whole path:
        v = ''                                                                  # the usual 'Blender.app' has none, and a user name may have some
        if h.endswith('.app/Contents/MacOS/Blender'):
            try:
                import plistlib
                with open(os.path.join(os.path.dirname(os.path.dirname(h)), 'Info.plist'), 'rb') as fh: v = str(plistlib.load(fh).get('CFBundleShortVersionString', ''))
            except Exception: pass
        if not re.search(r'\d', v):                                            # else the Blender folder's name: 'Blender 4.2', 'blender-4.2.1-linux-x64'
            v = next((mm.group() for c in reversed(re.split(r'[\\/]', h)[:-1]) if 'blender' in c.lower() for mm in [re.search(r'\d+(\.\d+)*', c)] if mm), '')
        return tuple(int(x) for x in re.findall(r'\d+', v)[:3])
    return sorted(hits, key=version)[-1] if hits else None
RENDER_SAMPLES, RENDER_ENGINE, RENDER_SIZE = 256, "cycles", "1920x1080"   # options passed to render_net.py
SHOW_SLIDER_CURVE = False                      # True: draw theta_1..theta_4 of the base block against theta_1 beside the slider (at the height of the track)

# ============================================================================================
# 2. FLAT ANGLES OF THE NET (the four relation types and the assembly of adjacent blocks)
# ============================================================================================
def compl(q): return tuple(PI - x for x in q)
def swap(q): return (q[1], q[0], q[3], q[2])
def compl_gd(q): return (q[0], q[1], PI - q[2], PI - q[3])
RELATION = {'a': compl, 'b': swap, 'c': compl_gd, 'd': lambda q: swap(compl_gd(q))}   # vertex 4 from vertex 1

def block_from_vertex1(q1, reltype):
    """quadruples (alpha_i, beta_i, gamma_i, delta_i), i = 1..4, of a 3 x 3 block with vertex 1 = q1 and the given relation type."""
    q4 = RELATION[reltype](q1)
    return {1: q1, 2: compl(q1), 3: compl(q4), 4: q4}


def wrap_math(text, width):
    """wrap a text with $...$ mathtext spans to 'width' characters per line: spaces inside the spans are removed first (mathtext ignores
    them), so that a span is never split; existing line breaks are kept."""
    def squeeze(mt): return re.sub(r"(\\[A-Za-z]+) +(?=[A-Za-z])", r"\1{}", mt.group(0)).replace(" ", "")   # '\leq M' -> '\leq{}M', not '\leqM'
    out = []
    for para in text.split("\n"):
        para = re.sub(r"\$[^$]*\$", squeeze, para)
        out += textwrap.wrap(para, width=width, break_long_words=False, break_on_hyphens=False) or [""]
    return "\n".join(out)

def angle_problem(angs_deg, tol_deg=1e-6):
    """None if the base angles satisfy the hypotheses; otherwise a sentence naming the violated one:
    (0) alpha, beta, gamma, delta lie in (0, 180) degrees;
    (1) no signed sum alpha +- beta +- gamma +- delta is 0 modulo 360 degrees (elliptic type);
    (2) sigma - alpha, ..., sigma - delta lie in (0, 180) degrees, sigma = (alpha + beta + gamma + delta)/2."""
    try:
        from numbers import Real
        if not all(isinstance(a, Real) for a in angs_deg): raise TypeError        # ('60' would pass float() but not the formulas)
        al, be, ga, de = map(float, angs_deg)
    except (TypeError, ValueError): return "the flat angles must be four numbers, in degrees; here %r" % (angs_deg,)   # (a typo in BASE_ANGLES_DEG)
    bad = [nm + " = %g" % a for nm, a in zip((r"$\alpha_1$", r"$\beta_1$", r"$\gamma_1$", r"$\delta_1$"), (al, be, ga, de)) if not 0 < a < 180]
    if bad: return "the flat angles must lie strictly between 0 and 180 degrees; here " + ", ".join(bad)   # (the same words as the Params dialog)
    for sb, sg, sd in itertools.product((1, -1), repeat=3):
        v = (al + sb*be + sg*ga + sd*de) % 360.0
        if min(v, 360.0 - v) < tol_deg:
            expr = r"$\alpha_1 %s \beta_1 %s \gamma_1 %s \delta_1$" % ("+" if sb > 0 else "-", "+" if sg > 0 else "-", "+" if sd > 0 else "-")
            return expr + " = 0 (mod 360°): the block is not of elliptic type; change one of the angles"
    sig = (al + be + ga + de)/2
    bars = [sig - al, sig - be, sig - ga, sig - de]
    bad = [(nm, b) for nm, b in zip([r"$\bar\alpha_1$", r"$\bar\beta_1$", r"$\bar\gamma_1$", r"$\bar\delta_1$"], bars) if not (tol_deg < b < 180 - tol_deg)]
    if bad:
        return (r"$\bar\alpha_1 = \sigma_1 - \alpha_1$, $\bar\beta_1 = \sigma_1 - \beta_1$, $\bar\gamma_1 = \sigma_1 - \gamma_1$, $\bar\delta_1 = \sigma_1 - \delta_1$ "
                r"with $\sigma_1 = (\alpha_1 + \beta_1 + \gamma_1 + \delta_1)/2$ must lie in (0°, 180°); here $\sigma_1$ = %.1f°, " % sig
                + ", ".join("%s = %g°" % (nm, b) for nm, b in bad))   # the degree sign must not be inside mathtext (it renders as a gamma)
    # (3) M = a_1 b_1 c_1 d_1 differs from 1 for angles of elliptic type; numerically close to 1 (u = 1 - M almost 0) the elliptic
    #     data degenerate, and such angles, close to violating (1), get no flexion
    M = vdata(*[math.radians(a) for a in (al, be, ga, de)])['M']
    if abs(1 - M) < 1e-5:
        return (r"$M = a_1 b_1 c_1 d_1$ = %.7f is numerically too close to 1 (angles of elliptic type give $M \neq 1$, and these are "
                r"close to violating it); change one of the angles" % M)
    return None

def check_parameters(m, n, base_kn, row_types, verbose=True):
    """clear messages for the usual mistakes: sizes that are not whole numbers, net too small or too large, base block out of range or not a
    pair of whole numbers, rows without a relation type."""
    def whole(v): return isinstance(v, (int, np.integer)) and not isinstance(v, bool)
    assert whole(m) and whole(n), "M_FACES and N_FACES must be whole numbers, like 5 (found %r and %r)" % (m, n)
    assert m >= 3 and n >= 3, "M_FACES and N_FACES must be at least 3 (the net must contain a 3 x 3 block)"
    assert m <= MAX_MN and n <= MAX_MN and m*n <= MAX_FACES, ("M_FACES and N_FACES must be at most MAX_MN = %d, and M_FACES x N_FACES at most MAX_FACES = %d "
                                                               "(found %d x %d): a larger net takes minutes to start" % (MAX_MN, MAX_FACES, m, n))
    assert isinstance(base_kn, tuple) and len(base_kn) == 2 and all(map(whole, base_kn)), "BASE_KN must be a pair of whole numbers, like (2, 2) (found %r)" % (base_kn,)
    k0, n0 = base_kn
    assert 1 <= k0 <= m - 2 and 1 <= n0 <= n - 2, "BASE_KN = (kappa, nu) must satisfy 1 <= kappa <= M_FACES-2 and 1 <= nu <= N_FACES-2"
    assert isinstance(row_types, dict), "ROW_TYPES must be a dict {row: type}, like {1: 'c', 2: 'a', 3: 'b'} (found %r)" % (row_types,)
    assert n0 in row_types, "ROW_TYPES must contain the row of the base block"
    rows = dict(row_types)
    missing = [nu for nu in range(1, n - 1) if nu not in rows]
    for nu in missing: rows[nu] = rows[n0]
    if missing and verbose: print("ROW_TYPES: rows %s not given, using the type '%s' of the row of the base block for them" % (missing, rows[n0]))
    for nu, r in rows.items(): assert isinstance(r, str) and r in RELATION, "ROW_TYPES: unknown relation type %r (use 'a', 'b', 'c' or 'd')" % (r,)
    return rows

def assemble_net(base_deg, m, n, base_kn, row_types):
    """quadruples of all 3 x 3 blocks (kappa, nu) of the m x n net, propagated from the base block by the relations between the flat angles of adjacent blocks."""
    q1 = tuple(math.radians(x) for x in base_deg)
    k0, n0 = base_kn
    sub = {base_kn: block_from_vertex1(q1, row_types[n0])}
    rev = lambda q: (q[3], q[2], q[1], q[0])          # (delta, gamma, beta, alpha)
    sw  = lambda q: (q[1], q[0], q[3], q[2])          # (beta, alpha, delta, gamma)
    # vertical propagation along the column kappa = k0
    for nu in range(n0 + 1, n - 1):
        prev = sub[(k0, nu - 1)]
        sub[(k0, nu)] = block_from_vertex1(rev(prev[4]), row_types[nu])
        assert np.allclose(sub[(k0, nu)][2], rev(prev[3]))
    for nu in range(n0 - 1, 0, -1):
        nxt = sub[(k0, nu + 1)]
        q4 = rev(nxt[1]); q1 = RELATION[row_types[nu]](q4)
        sub[(k0, nu)] = {1: q1, 2: compl(q1), 3: compl(q4), 4: q4}
        assert np.allclose(sub[(k0, nu)][3], rev(nxt[2]))
    # horizontal propagation in every row
    for nu in range(1, n - 1):
        for k in range(k0 + 1, m - 1):
            prev = sub[(k - 1, nu)]
            q2, q3 = sw(prev[1]), sw(prev[4])
            sub[(k, nu)] = {1: compl(q2), 2: q2, 3: q3, 4: compl(q3)}
        for k in range(k0 - 1, 0, -1):
            nxt = sub[(k + 1, nu)]
            q1, q4 = sw(nxt[2]), sw(nxt[3])
            sub[(k, nu)] = {1: q1, 2: compl(q1), 3: compl(q4), 4: q4}
    return sub

RELATION_TEXT = {'a': r"$(\alpha_4,\beta_4,\gamma_4,\delta_4) = (\pi-\alpha_1,\ \pi-\beta_1,\ \pi-\gamma_1,\ \pi-\delta_1)$",
                 'b': r"$(\alpha_4,\beta_4,\gamma_4,\delta_4) = (\beta_1,\ \alpha_1,\ \delta_1,\ \gamma_1)$",
                 'c': r"$(\alpha_4,\beta_4,\gamma_4,\delta_4) = (\alpha_1,\ \beta_1,\ \pi-\gamma_1,\ \pi-\delta_1)$",
                 'd': r"$(\alpha_4,\beta_4,\gamma_4,\delta_4) = (\beta_1,\ \alpha_1,\ \pi-\delta_1,\ \pi-\gamma_1)$"}
# in every 3 x 3 block, vertices 2 and 3 are the complements to pi of vertices 1 and 4; the relation between vertices 1 and 4
# is one of the four above (the flat angles are taken at the vertex A_1, ..., A_4 of the block)

def block_labels(k, n):
    A = {1: (k+1, n), 2: (k, n), 3: (k, n+1), 4: (k+1, n+1)}
    B = {1: (k+1, n-1), 2: (k, n-1), 3: (k, n+2), 4: (k+1, n+2)}
    C = {1: (k+2, n), 2: (k-1, n), 3: (k-1, n+1), 4: (k+2, n+1)}
    return A, B, C

def vertex_face_angles(sub):
    """angle[(vertex)][(face)] for all inner vertices, from the block quadruples; checks consistency."""
    ang = {}
    def put(v, f, val):
        ang.setdefault(v, {})
        if f in ang[v]: assert abs(ang[v][f] - val) < 1e-9, ("inconsistent flat angles", v, f)
        ang[v][f] = val
    for (k, n), q in sub.items():
        A, B, C = block_labels(k, n)
        faces = {'alpha': {1: (k, n-1), 2: (k, n-1), 3: (k, n+1), 4: (k, n+1)},
                 'gamma': {1: (k+1, n), 2: (k-1, n), 3: (k-1, n), 4: (k+1, n)},
                 'beta':  {1: (k+1, n-1), 2: (k-1, n-1), 3: (k-1, n+1), 4: (k+1, n+1)},
                 'delta': {i: (k, n) for i in range(1, 5)}}
        for i in range(1, 5):
            for idx, nm in enumerate(('alpha', 'beta', 'gamma', 'delta')):
                put(A[i], faces[nm][i], q[i][idx])
    return ang

# ============================================================================================
# 3. EQUIMODULAR DATA, PHASE SHIFTS, EXPLICIT FLEXIONS
# ============================================================================================
def vdata(al, be, ga, de):
    s = (al + be + ga + de)/2
    a, b, c, d = (math.sin(al)/math.sin(s-al), math.sin(be)/math.sin(s-be), math.sin(ga)/math.sin(s-ga), math.sin(de)/math.sin(s-de))
    return dict(al=al, be=be, ga=ga, de=de, sig=s, bars=(s-al, s-be, s-ga, s-de), a=a, b=b, c=c, d=d,
                r=a*d, s=c*d, f=a*c, M=a*b*c*d, eps=1 if math.sin(s) > 0 else -1)

def ellipK(m):
    """complete elliptic integral K(k), m = k^2, by the arithmetic-geometric mean."""
    a, b = 1.0, math.sqrt(1 - m)
    for _ in range(60):
        if abs(a - b) <= 1e-15*a: break
        a, b = (a + b)/2, math.sqrt(a*b)
    return PI/(2*a)

def jacobi_real(u, m):
    """sn, cn, dn of a real argument u with parameter m = k^2 in (0, 1), by the descending Landen (AGM) recursion."""
    a, c, b = [1.0], [math.sqrt(m)], math.sqrt(1 - m)
    while abs(c[-1]) > 1e-15 and len(a) < 60:
        a.append((a[-1] + b)/2); c.append((a[-2] - b)/2); b = math.sqrt(a[-2]*b)
    n = len(a) - 1
    phi = (2**n)*a[n]*u
    for i in range(n, 0, -1):
        phi = (phi + math.asin(max(-1.0, min(1.0, c[i]*math.sin(phi)/a[i]))))/2
    sn = math.sin(phi); cn = math.cos(phi)
    return sn, cn, math.sqrt(max(1 - m*sn*sn, 0.0))

def _bricard_hom(c): return (0.0, math.copysign(1.0, c)) if math.isinf(c) else (1.0, c)   # cot(theta/2) = c1/c0: +-inf (a folded crease) is (0, +-1)

def bricard_P(v, X, Y):
    """P_i(X, Y) of Bricard's equations for the vertex data v, in homogeneous form P X0^2 Y0^2 (X = X1/X0, Y = Y1/Y0, see
    _bricard_hom): for finite X, Y the same value; an infinite X or Y (a folded crease) gives the limit P/X^2 or P/Y^2, not inf or NaN."""
    ab, bb, gb, db = v['bars']; be, s = v['be'], v['sig']; (X0, X1), (Y0, Y1) = _bricard_hom(X), _bricard_hom(Y)
    return (math.sin(db)*math.sin(db-be)*X1*X1*Y1*Y1 + math.sin(ab)*math.sin(ab-be)*X1*X1*Y0*Y0 + math.sin(gb)*math.sin(gb-be)*X0*X0*Y1*Y1
            - 2*math.sin(v['al'])*math.sin(v['ga'])*X0*X1*Y0*Y1 + math.sin(s)*math.sin(bb)*X0*X0*Y0*Y0)

def sgn(a): return 1.0 if a > 0 else (-1.0 if a < 0 else 0.0)
def rsqrt(a, tol=1e-9):
    """real square root of a radicand that is nonnegative up to rounding (all radicands of the flexion formulas are >= 0 on the admissible set)."""
    if a < -tol*(1 + abs(a)): raise ValueError("negative radicand")
    return math.sqrt(max(a, 0.0))

FLEX_T_INF = 1e50           # |t| above this (t = +-inf included: theta_1 = 0) the flexion formulas are evaluated in s = 1/t
FLEX_REMOVABLE = 1e-5       # a denominator of the formulas below this fraction of its terms: near a removable 0/0 (see _flex_quotient)
FLEX_AGREE = 1e-12          # ... where the conjugate form replaces the formula as written if the two differ by more than this (radians)
FLEX_ULP = sys.float_info.epsilon  # machine epsilon 2^-52 (the ulp of 1.0; the unit roundoff is half of it)
FLEX_NOCANCEL = 1e-3        # ... provided the conjugate (or poly) is at least this fraction of its terms: it does not cancel itself
FLEX_END_ULPS = 8           # a factor of D(t) = (x1 t^2 - 1)(1 - u x1 t^2) within this many ulps of its terms of 0 is 0: t is an end of I (a branch point)

def _cot_gap(c1, c2):
    """the distance of the angles theta = 2 acot(c) of two values c of cot(theta/2) (+-inf: theta = 0)."""
    d = abs(math.atan2(1.0, c1) - math.atan2(1.0, c2))
    return 2*min(d, PI - d)

def _flex_quotient(k, num, conj, c_terms, lead, d, d_terms, poly, p_terms):
    """the value k*num/(lead*d) of one of the flexion formulas, with its limits where d = 0. On the curve num*conj = lead*d*poly
    (conj: num with the other sign of its square root, c_terms: the size of its two terms), so where d = 0 the formula of one
    sign e0 has a pole (num != 0: cot = +-inf, the crease is folded, theta = 0) and that of the other a removable 0/0 (at a branch
    point both). Near such a point (d below FLEX_REMOVABLE of the size d_terms of its terms) rounding in num and d spoils num/d
    unless it is a pole; the same value is also k*poly/conj, and that one is taken when the two differ by more than FLEX_AGREE
    in angle (where the two agree within FLEX_AGREE, num/d is kept to the last bit), provided the conjugate does not
    cancel, or num cancels too while poly does not (p_terms: the size of its terms). On the pole side (num does not cancel, conj
    does) num/d is the accurate one and is kept. The second case covers a double zero: where the first term of the
    numerator vanishes identically (U of a block with H2 H3 = 0, then U = k Q/dU and dU pU = -Q^2, so dU ~ D), num, conj and d
    all vanish at a branch point; there num/d is rounding noise over rounding noise, while tan(theta/2) = conj/(k poly) is
    accurate to rounding (conj -> 0: the crease is folded at the branch point)."""
    den = lead*d
    if abs(d) <= FLEX_REMOVABLE*d_terms and (abs(conj) >= FLEX_NOCANCEL*c_terms or (abs(num) < FLEX_NOCANCEL*c_terms and abs(poly) >= FLEX_NOCANCEL*p_terms)):
        alt = k*poly/conj if conj != 0 else math.copysign(math.inf, k*poly)
        if den == 0 or _cot_gap(k*num/den, alt) > FLEX_AGREE: return alt
    if den == 0 and num != 0: return math.copysign(math.inf, k*num)*math.copysign(1.0, den)
    return k*num/den

def _flex_checked(*vals):
    """the values of thmain_formulas; NaN (no limit, e.g. an inf/inf) is no configuration."""
    if any(v != v for v in vals): raise ValueError("the flexion formulas give no number here")
    return vals

def thmain_formulas(x1, x3, y1, y2, z1, z2, z3, u, eps, e, e0, t, root=None, linear=False):
    """the explicit flexion formulas: (cot theta_1/2, cot theta_2/2, cot theta_3/2, cot theta_4/2) = (Z, W_2, U, W_1) at the parameter
    t = Z, for the signs e = (e1, e2, e3, e4) and the sign e0 = +-1; eps = (eps_1, eps_2, eps_3, eps_4).
    The sign e0 enters only through the signed root r = e0 sqrt(x1 D) (sqrt(u y z D) = sqrt(u y z/x1) sqrt(x1 D)): with root = r
    given it replaces e0 sqrt(x1 D) (near a branch point, D ~ 0, r tells the configuration and t hardly does), and linear=True
    returns, instead of W2, U, W1, their coefficients (n0, n1, d): cot = (n0 + n1 r)/d.
    Where the formulas as written divide by zero they give the limit (see _flex_quotient): a pole is cot = +-inf (theta = 0, a
    folded crease; case (c) at t = 0: theta_3 = 0), a removable 0/0 its value. For |t| > FLEX_T_INF, t = +-inf included (theta_1 = 0),
    numerators and denominators are divided by t^2 and evaluated in s = 1/t (r then stands for e0 sqrt(x1 D)/t^2; no overflow in
    t^4 or D; the terms odd in t vanish in the limit, so t = +inf and t = -inf give the same angles for the same e0).
    At an end of the admissible set (D = 0, a branch point: both signs e0 give the same configuration) rounding can leave t just
    outside, x1 D < 0, where the radicands are negative beyond the tolerance of rsqrt when the other factor of D is large (e.g. t
    = 1/sqrt(u x1) computed in floating point with |u| small): a factor that is 0 up to FLEX_END_ULPS ulps of its terms makes
    D = 0 there (inside I, x1 D >= 0, nothing changes)."""
    e1, e2, e3, e4 = e
    far = not abs(t) <= FLEX_T_INF
    if not far:
        D = (x1*t*t - 1)*(1 - u*x1*t*t)
        if x1*D < 0 and min(abs(x1*t*t - 1)/(abs(x1*t*t) + 1), abs(1 - u*x1*t*t)/(1 + abs(u*x1*t*t))) <= FLEX_END_ULPS*FLEX_ULP: D = 0.0
        T = x1*t*t
        a, dW2, dW1 = t, z2 + x1*t*t, z1 + x1*t*t                                          # W = eps (a A +- B)/(y u dW)
        dW2_terms, dW1_terms, pW2, pW1 = abs(z2) + abs(T), abs(z1) + abs(T), 1 + u*z2*T, 1 + u*z1*T   # (aA + B)(aA - B) = u y dW pW
        pW2_terms, pW1_terms = 1 + abs(u*z2*T), 1 + abs(u*z1*T)
    else:
        s = 1/t; D = (x1 - s*s)*(s*s - u*x1)                                                  # D/t^4
        a, dW2, dW1 = s, z2*s*s + x1, z1*s*s + x1
        dW2_terms, dW1_terms, pW2, pW1 = abs(z2*s*s) + abs(x1), abs(z1*s*s) + abs(x1), s*s + u*z2*x1, s*s + u*z1*x1
        pW2_terms, pW1_terms = s*s + abs(u*z2*x1), s*s + abs(u*z1*x1)
    A2, A1 = a*rsqrt(u*x1*y2*(1 + z2)*(1 + u*z2)), a*rsqrt(u*x1*y1*(1 + z1)*(1 + u*z1))
    if linear: c2, c1 = e2*rsqrt(u*y2*z2/x1), e1*rsqrt(u*y1*z1/x1)                          # B = c r
    elif root is None: B2, B1 = e0*e2*rsqrt(u*y2*z2*D), e0*e1*rsqrt(u*y1*z1*D)
    else: B2, B1 = e2*rsqrt(u*y2*z2/x1)*root, e1*rsqrt(u*y1*z1/x1)*root
    if not linear:
        W2 = _flex_quotient(eps[1], A2 + B2, A2 - B2, abs(A2) + abs(B2), y2*u, dW2, dW2_terms, pW2, pW2_terms)
        W1 = _flex_quotient(eps[0], A1 - B1, A1 + B1, abs(A1) + abs(B1), y1*u, dW1, dW1_terms, pW1, pW1_terms)
    if abs(u*z2*z3 - 1) > 1e-9:                                                                    # case (a)
        den = u*z2*z3 - 1
        H1 = (abs(z2)*rsqrt(z3*(1 + z3)*(1 + u*z3)) + e2*e3*abs(z3)*rsqrt(z2*(1 + z2)*(1 + u*z2)))/den
        H2 = (abs(z2*z3)*rsqrt((1 + u*z2)*(1 + u*z3)) + e2*e3*rsqrt(z2*z3*(1 + z2)*(1 + z3)))/den
        H3 = (u*rsqrt(z2*z3*(1 + z2)*(1 + z3))*sgn(z2*z3) + e2*e3*rsqrt((1 + u*z2)*(1 + u*z3)))/den
        if e2 == -e3 and abs(den) < 1e-3:                                                           # near case (b): each H = (a - b)/den with a - b ~ den, so
            p = z2*z3                                                                               # H = (a^2 - b^2)/((a + b) den), a^2 - b^2 = den*(...) cancelled
            a1, b1 = abs(z2)*rsqrt(z3*(1 + z3)*(1 + u*z3)), abs(z3)*rsqrt(z2*(1 + z2)*(1 + u*z2))
            a2, b2 = abs(p)*rsqrt((1 + u*z2)*(1 + u*z3)), rsqrt(p*(1 + z2)*(1 + z3))
            a3, b3 = u*rsqrt(p*(1 + z2)*(1 + z3))*sgn(p), rsqrt((1 + u*z2)*(1 + u*z3))
            if a1 + b1 > 0: H1 = p*(z3 - z2)/(a1 + b1)
            if a2 + b2 > 0: H2 = p*(z2 + z3 + 1 + u*p)/(a2 + b2)
            if a3 > 0: H3 = (u*(z2 + z3) + u*p + 1)/(a3 + b3)                                       # (a3 < 0: a3 - b3 does not cancel)
    elif e2 == -e3:                                                                                 # case (b)
        rt = 2*rsqrt(z2*z3*(1 + z2)*(1 + z3))
        H1 = z2*z3*(z3 - z2)/rt; H2 = z2*z3*(z2 + z3 + 2)/rt; H3 = (z2 + z3 + 2*z2*z3)*sgn(z2*z3)/rt
    else:                                                                                           # case (c): U = k/t
        k = eps[1]*eps[2]*rsqrt(z2*z3/(x1*x3))*sgn(y2)
        if linear: return t, (eps[1]*A2, eps[1]*c2, y2*u*dW2), ((k*s, 0.0, 1.0) if far else (k, 0.0, t)), (eps[0]*A1, -eps[0]*c1, y1*u*dW1)
        if far: return _flex_checked(t, W2, k*s, W1)                                                 # U -> 0: theta_3 = pi
        U = k/t if t != 0 else math.copysign(math.inf, k)*math.copysign(1.0, t)                    # t = 0: theta_3 = 0
        return _flex_checked(t, W2, U, W1)
    k = eps[1]*eps[2]*sgn(z2)*rsqrt(z2*z3/(x1*x3))                                                  # U = k (P + Q)/dU
    if not far: P, dU, dU_terms, pU = abs(x1)*H2*H3*t, z2*z3 + u*x1*H1*H1*t*t, abs(z2*z3) + abs(u*x1*H1*H1*t*t), x1*(H1*H1 + z2*z3*T)/(z2*z3)
    else: P, dU, dU_terms, pU = abs(x1)*H2*H3*s, z2*z3*s*s + u*x1*H1*H1, abs(z2*z3*s*s) + abs(u*x1*H1*H1), x1*(H1*H1*s*s + z2*z3*x1)/(z2*z3)
    if linear: return t, (eps[1]*A2, eps[1]*c2, y2*u*dW2), (k*P, k*e2*H1, dU), (eps[0]*A1, -eps[0]*c1, y1*u*dW1)
    pU_terms = abs(x1)*(H1*H1 + abs(z2*z3*T))/abs(z2*z3) if not far else abs(x1)*(H1*H1*s*s + abs(z2*z3*x1))/abs(z2*z3)
    Q = e0*e2*H1*rsqrt(x1*D) if root is None else e2*H1*root
    U = _flex_quotient(k, P + Q, P - Q, abs(P) + abs(Q), 1.0, dU, dU_terms, pU, pU_terms)           # (P + Q)(P - Q) = dU pU
    return _flex_checked(t, W2, U, W1)

def admissible_range(x, u):
    """the admissible set I(x) = {t : x D(x, t) > 0} as a list of intervals (t > 0 half; the set is symmetric in t)."""
    if x > 0 and u > 0: return [(1/math.sqrt(x), 1/math.sqrt(u*x))] if u < 1 else [(1/math.sqrt(x), math.inf)]
    if x > 0 and u < 0: return [(1/math.sqrt(x), math.inf)]
    if x < 0 and u < 0: return [(0.0, 1/math.sqrt(u*x))]
    return [(0.0, math.inf)]                    # x < 0, u > 0: x D > 0 for every t

class Block:
    """a 3 x 3 block (Kokotsakis polyhedron): vertex data, elliptic data and the explicit flexions."""
    def __init__(self, k, n, quads):
        self.k, self.n = k, n
        self.A, self.B, self.C = block_labels(k, n)
        self.q = quads
        self.v = {i: vdata(*quads[i]) for i in range(1, 5)}
        M = self.v[1]['M']
        close = lambda a, b: abs(a - b) < 1e-9*max(1.0, abs(a), abs(b))       # relative: M, r, s are large (~ 1/barred angle) when a barred angle is small
        assert all(close(self.v[i]['M'], M) for i in range(1, 5)), "not equimodular"
        assert close(self.v[1]['r'], self.v[2]['r']) and close(self.v[3]['r'], self.v[4]['r']), "amplitudes at common vertices do not match"
        assert close(self.v[1]['s'], self.v[4]['s']) and close(self.v[2]['s'], self.v[3]['s']), "amplitudes at common vertices do not match"
        self.M = M; self.u = 1 - M
        self.x = {i: 1/(self.v[i]['r'] - 1) for i in range(1, 5)}
        self.y = {i: 1/(self.v[i]['s'] - 1) for i in range(1, 5)}
        self.z = {i: 1/(self.v[i]['f'] - 1) for i in range(1, 5)}
        self.eps = tuple(self.v[i]['eps'] for i in range(1, 5))
        self.x1 = self.x[1]
        self.k2 = (1 - M) if M < 1 else (1 - 1/M)            # elliptic modulus k: k^2 = 1 - M (M < 1), 1 - 1/M (M > 1)
        self.kk, self.kp = math.sqrt(self.k2), math.sqrt(1 - self.k2)
        self.K, self.Kp = ellipK(self.k2), ellipK(1 - self.k2)
        self.t = {i: self.phase(i) for i in range(1, 5)}       # phase shifts t_i = mu K + i y, 0 < y < K'

    _phase_cache = {}
    def phase(self, i):
        """phase shift t_i = mu*K + i*y (complex): dn(t_i) = sqrt(f_i) (M < 1) or 1/sqrt(f_i) (M > 1), with the real part mu*K,
        mu in {0, 1, 2, 3}, fixed by the signs of p_i q_i and of sin(sigma_i), and 0 < y < K'.  Only real Jacobi functions are
        needed: dn(i y) = dn(y, k')/cn(y, k') and dn(K + i y) = k' cn(y, k')/dn(y, k')."""
        v = self.v[i]; g = math.sqrt(v['f']) if self.M < 1 else 1/math.sqrt(v['f'])
        key = (round(v['al'], 12), round(v['be'], 12), round(v['ga'], 12), round(v['de'], 12))
        if key in Block._phase_cache: return Block._phase_cache[key]
        p_real, q_real = v['r'] > 1, v['s'] > 1
        cls = 0 if (p_real and q_real) else (2 if (not p_real and not q_real) else 1)   # p q real positive / imaginary / real negative
        mu = [[0, 1, 2], [2, 3, 0]][0 if v['sig'] < PI else 1][cls]
        mp_ = 1 - self.k2
        def dnval(y):
            sn, cn, dn = jacobi_real(y, mp_)
            return dn/cn if mu % 2 == 0 else self.kp*cn/dn
        lo, hi = 1e-9*self.Kp, (1 - 1e-9)*self.Kp
        flo, fhi = dnval(lo) - g, dnval(hi) - g
        assert flo*fhi < 0, "phase shift: no solution in (0, K')"
        for _ in range(70):                                                     # 70 halvings of (0, K') reach double precision
            mid = (lo + hi)/2; fm = dnval(mid) - g
            if fm*flo <= 0: hi, fhi = mid, fm
            else: lo, flo = mid, fm
        Block._phase_cache[key] = complex(mu*self.K, (lo + hi)/2)
        return Block._phase_cache[key]

    def lattice_string(self):
        return r"$\{4K\,m + 2\mathrm{i}K'\,n\}$" if self.M < 1 else r"$\{4K\,m + (2K + 2\mathrm{i}K')\,n\}$"

    def phase_string(self, i):
        t = self.t[i]
        return r"$%.4f\,K + %.4f\,\mathrm{i}K'$" % (t.real/self.K, t.imag/self.Kp)

    def sign_condition_string(self, e):
        """e_1 t_1 + ... + e_4 t_4 written in the basis K, iK' and as an element of the lattice Lambda."""
        z = sum(e[i-1]*self.t[i] for i in range(1, 5))
        if self.M < 1: mm, nn = z.real/(4*self.K), z.imag/(2*self.Kp); basis = r"%d \cdot 4K + %d \cdot 2\mathrm{i}K'"
        else: nn = z.imag/(2*self.Kp); mm = (z.real - 2*self.K*nn)/(4*self.K); basis = r"%d \cdot 4K + %d \cdot (2K + 2\mathrm{i}K')"
        inlat = abs(mm - round(mm)) < 1e-9 and abs(nn - round(nn)) < 1e-9
        val = r"$%.4f\,K + %.4f\,\mathrm{i}K'$" % (z.real/self.K, z.imag/self.Kp)
        return val + (r" $= " + basis % (round(mm), round(nn)) + r" \in \Lambda$" if inlat else r" $\notin \Lambda$")

    # ------------------------------------------------------------------ the flexion formulas with any of the four angles as parameter
    def shifted_data(self, shift):
        """vertex data relabelled cyclically by 'shift' (new vertex j = old vertex j + shift); for odd shifts the roles of
        alpha_i and gamma_i, hence of x_i and y_i, are interchanged."""
        idx = [((j - 1 + shift) % 4) + 1 for j in range(1, 5)]
        x, y = (self.x, self.y) if shift % 2 == 0 else (self.y, self.x)
        return (x[idx[0]], x[idx[2]], y[idx[0]], y[idx[1]], self.z[idx[0]], self.z[idx[1]], self.z[idx[2]],
                tuple(self.eps[i - 1] for i in idx), idx)

    def cots(self, e, e0, t, shift=0, root=None, linear=False):
        """{i: cot(theta_i/2)} from the flexion formulas with the parameter t = cot(theta_{1+shift}/2) and the sign e0 (or the signed
        root e0 sqrt(x1 D) in that numbering, root; linear=True: {i: (n0, n1, d)} for the three other angles, see thmain_formulas)."""
        x1, x3, y1, y2, z1, z2, z3, eps, idx = self.shifted_data(shift)
        es = tuple(e[i - 1] for i in idx)
        vals = thmain_formulas(x1, x3, y1, y2, z1, z2, z3, self.u, eps, es, e0, t, root, linear)
        return {idx[j]: vals[j] for j in range(1 if linear else 0, 4)}

    def angles_from_cots(self, c): return {i: (2*math.atan(1/c[i]) if c[i] != 0 else PI) for i in range(1, 5)}   # cot theta/2 = 0: theta = 180 degrees (the paper's -pi, the same flat edge: kept +pi so that theta_1 -> 180 as t -> 0+ on the slider; angles are compared modulo 2 pi, angle_gap); cot = +-inf (a pole of the formulas, a folded crease): 1/c = +-0, theta = 0

    def admissible(self, shift=0):
        """I(x_1), I(y_2), I(x_3), I(y_4) for shift = 0, 1, 2, 3."""
        x1 = self.shifted_data(shift)[0]
        return admissible_range(x1, self.u)

    def bricard_residual(self, c):
        Z, W2, U, W1 = c[1], c[2], c[3], c[4]
        return max(abs(bricard_P(self.v[1], Z, W1)), abs(bricard_P(self.v[2], Z, W2)), abs(bricard_P(self.v[3], U, W2)), abs(bricard_P(self.v[4], U, W1)))

    def bricard_relative(self, c):
        """the residual of Bricard's equations relative to the size of their terms: near the boundary of the elliptic type
        (M close to 1) the data x_i, y_i, z_i and the cotangents are large and the terms reach 1e4 .. 1e6, so an absolute
        residual of 1e-8 is rounding there, not a violation. An infinite cot (a folded crease) enters through the limit equation
        (see bricard_P)."""
        Z, W2, U, W1 = c[1], c[2], c[3], c[4]; worst = 0.0
        for v, X, Y in ((self.v[1], Z, W1), (self.v[2], Z, W2), (self.v[3], U, W2), (self.v[4], U, W1)):
            ab, bb, gb, db = v['bars']; be, s = v['be'], v['sig']; (X0, X1), (Y0, Y1) = _bricard_hom(X), _bricard_hom(Y)
            terms = (math.sin(db)*math.sin(db-be)*X1*X1*Y1*Y1, math.sin(ab)*math.sin(ab-be)*X1*X1*Y0*Y0, math.sin(gb)*math.sin(gb-be)*X0*X0*Y1*Y1,
                     -2*math.sin(v['al'])*math.sin(v['ga'])*X0*X1*Y0*Y1, math.sin(s)*math.sin(bb)*X0*X0*Y0*Y0)
            worst = max(worst, abs(sum(terms))/max(1.0, max(abs(t_) for t_ in terms)))
        return worst

# ============================================================================================
# 4. GEOMETRY OF ONE 3 x 3 BLOCK (to read off the non-central dihedral angles at its vertices)
# ============================================================================================
def unit(v): return v/np.linalg.norm(v)

def convex_quad(d1, d2, d3, a1=1.0):
    """planar convex quadrilateral A_1 A_2 A_3 A_4 with angles d_i and |A_2 A_1| = a1."""
    d4 = 2*PI - d1 - d2 - d3
    s23, s12 = d2 + d3, d1 + d2
    # the side a2 = |A_2 A_3| of a convex quadrilateral with these angles: a3 > 0 needs a2 > lo_ (when d2 + d3 < pi), a4 > 0 needs
    # a2 < hi_ (when d1 + d2 < pi). At d2 + d3 = pi (or d1 + d2 = pi, a parallelogram-shaped face) lo_ -> 0 (hi_ -> inf): a2 is
    # kept away from 0 (and from infinity), otherwise A_3 falls onto A_2 and the directions of the edges are lost in rounding.
    lo_ = a1*math.sin(s23)/math.sin(d3) if s23 < PI - 1e-12 else 0.0
    hi_ = a1*math.sin(d1)/math.sin(s12) if s12 < PI - 1e-12 else math.inf
    if hi_ == math.inf: a2 = a1 if lo_ == 0.0 else max(1.2*lo_, 1e-3*a1)
    elif lo_ == 0.0: a2 = min(0.8*hi_, 1e3*a1)
    else: a2 = (lo_ + min(hi_, max(1e3*a1, 2*lo_)))/2
    A2 = np.zeros(3); A1 = np.array([a1, 0, 0.]); A3 = a2*np.array([math.cos(d2), math.sin(d2), 0])
    a3 = (a2*math.sin(d3) - a1*math.sin(s23))/math.sin(d4)
    A4 = A1 + a3*np.array([-math.cos(d1), math.sin(d1), 0])
    return {1: A1, 2: A2, 3: A3, 4: A4}

def block_points(S, th):
    """3D points A_i, B_i, C_i of the block with the oriented interior dihedral angles th."""
    q = S.q
    A = convex_quad(q[1][3], q[2][3], q[3][3])
    n0 = unit(np.cross(A[1] - A[2], A[3] - A[2]))
    e = {1: unit(A[1] - A[2]), 2: unit(A[2] - A[3]), 3: unit(A[3] - A[4]), 4: unit(A[4] - A[1])}
    d0 = {1: np.cross(n0, e[1]), 2: np.cross(n0, e[2]), 3: np.cross(n0, e[3]), 4: np.cross(n0, e[4])}   # into the central face
    d = {i: math.cos(th[i])*d0[i] + math.sin(th[i])*n0 for i in range(1, 5)}                              # into the side face
    al = {i: q[i][0] for i in range(1, 5)}; ga = {i: q[i][2] for i in range(1, 5)}
    # the points B_i, C_i only mark the directions of the edges at A_i; their distances from A_i are chosen so that each of the
    # four side faces is a convex quadrilateral: where the two edges of a side face converge (the angles at its base add up to
    # less than pi), half the distance to the point where they meet, else 1. A crossed (bow-tie) side face reverses its normal at
    # the far corners, and a dihedral angle read from these points then comes out as the supplement with the opposite sign
    # (this made the synchronization reject correct neighbours, e.g. for rows of type d).
    L = {1: np.linalg.norm(A[1] - A[2]), 2: np.linalg.norm(A[2] - A[3]), 3: np.linalg.norm(A[3] - A[4]), 4: np.linalg.norm(A[4] - A[1])}
    def ln(base, near, far):
        s = near + far
        return 1.0 if s >= PI - 1e-12 else min(1.0, 0.5*base*math.sin(far)/math.sin(s))
    B = {1: A[1] + ln(L[1], al[1], al[2])*(math.cos(al[1])*(-e[1]) + math.sin(al[1])*d[1]), 2: A[2] + ln(L[1], al[2], al[1])*(math.cos(al[2])*e[1] + math.sin(al[2])*d[1]),
         3: A[3] + ln(L[3], al[3], al[4])*(math.cos(al[3])*(-e[3]) + math.sin(al[3])*d[3]), 4: A[4] + ln(L[3], al[4], al[3])*(math.cos(al[4])*e[3] + math.sin(al[4])*d[3])}
    C = {2: A[2] + ln(L[2], ga[2], ga[3])*(math.cos(ga[2])*(-e[2]) + math.sin(ga[2])*d[2]), 3: A[3] + ln(L[2], ga[3], ga[2])*(math.cos(ga[3])*e[2] + math.sin(ga[3])*d[2]),
         1: A[1] + ln(L[4], ga[1], ga[4])*(math.cos(ga[1])*e[4] + math.sin(ga[1])*d[4]), 4: A[4] + ln(L[4], ga[4], ga[1])*(math.cos(ga[4])*(-e[4]) + math.sin(ga[4])*d[4])}
    pts = {}
    for i in range(1, 5): pts[S.A[i]] = A[i]; pts[S.B[i]] = B[i]; pts[S.C[i]] = C[i]
    closure = max(abs(math.acos(np.clip(np.dot(unit(B[i]-A[i]), unit(C[i]-A[i])), -1, 1)) - q[i][1]) for i in range(1, 5))
    return pts, closure

def _cross(a, b): return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])
def _unit3(v):
    l = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); return (v[0]/l, v[1]/l, v[2]/l)

def face_normal(order, pts):
    """unit normal (prev - P) x (next - P) of a face given by the cyclic order of its corners, at a corner P whose neighbours are known."""
    for j in range(4):
        P, prev, nxt = order[j], order[j-1], order[(j+1) % 4]
        if P in pts and prev in pts and nxt in pts:
            p, q, r = pts[P], pts[prev], pts[nxt]
            return _unit3(_cross((q[0]-p[0], q[1]-p[1], q[2]-p[2]), (r[0]-p[0], r[1]-p[1], r[2]-p[2])))
    return None

DIHEDRAL_ACOS = 0.9999                             # dihedral(): |cos(bend)| above this, the bend is taken by atan2 (acos loses digits there)

def dihedral(S, i, pts):
    """oriented interior dihedral angle theta_i of block S read from 3D points (None if points are missing)."""
    A, B, C = S.A, S.B, S.C
    n0 = face_normal([A[1], A[2], A[3], A[4]], pts)
    orders = {1: [B[2], A[2], A[1], B[1]], 2: [A[3], A[2], C[2], C[3]], 3: [A[4], A[3], B[3], B[4]], 4: [C[1], A[1], A[4], C[4]]}
    ni = face_normal(orders[i], pts)
    if n0 is None or ni is None: return None
    nxt = A[i+1] if i < 4 else A[1]
    d = n0[0]*ni[0] + n0[1]*ni[1] + n0[2]*ni[2]
    if abs(d) < DIHEDRAL_ACOS: bend = math.acos(d)
    else:                                              # nearly flat or folded: acos(1 - 2^-53) is already 1.5e-8, atan2 is exact to rounding
        x_ = _cross(n0, ni); bend = math.atan2(math.sqrt(x_[0]*x_[0] + x_[1]*x_[1] + x_[2]*x_[2]), d)
    e = (pts[A[i]][0] - pts[nxt][0], pts[A[i]][1] - pts[nxt][1], pts[A[i]][2] - pts[nxt][2])
    c = _cross(n0, e)                                  # det(ni, n0, e) = ni . (n0 x e)
    det = ni[0]*c[0] + ni[1]*c[1] + ni[2]*c[2]
    return (1 if det > 0 else -1)*(PI - bend)

# ============================================================================================
# 5. SYNCHRONIZATION OF THE BLOCKS
# ============================================================================================
def angle_gap(a, b):
    """the distance of two oriented angles on the circle, in [0, pi]: a flat edge is both +pi and -pi (the formulas give +pi at
    cot = 0, the vertices -pi, or +-(pi - tiny) with a sign that is rounding noise), so angles are compared modulo 2 pi.
    Written so that it equals abs(a - b) exactly whenever |a - b| <= pi (the remainder is exact for |a - b| < 2 pi, and so is
    2 pi - d for d in (pi, 2 pi))."""
    d = abs(a - b) % (2*PI)
    return min(d, 2*PI - d)

def compatible(Sa, tha, ptsa, Sb, thb, ptsb, tol=1e-6):
    for i in range(1, 5):
        d = dihedral(Sb, i, ptsa)
        if d is not None and angle_gap(d, thb[i]) > tol: return False
        d = dihedral(Sa, i, ptsb)
        if d is not None and angle_gap(d, tha[i]) > tol: return False
    return True

REALIZE_ANGLE_TOL = 1e-6        # realize_from: the old block is reproduced when its angles agree within this (radians)
REALIZE_BRANCH = 1e-10          # ... the pair is at a branch point when a factor of D(t) = (x1 t^2 - 1)(1 - u x1 t^2) is below this relative to its terms
REALIZE_NEAR = 1e-4             # ... and near one within this, where the signed root of the old block is carried to the new one
REALIZE_EXACT = 1e-10           # ... when it reproduces the old block as well as the best sign does, or within this
REALIZE_AGREE = 1e-8            # ... a result differing from that of the first acceptable sign (the rule before) by at most this is replaced by it
REALIZE_KEEP_T = 1e-4           # ... except next to the flat end, |t| below this, where the tool did not go before (synchronize)

def _roots_from_angles(S, signs, t, shift, th):
    """the signed root r = e0 sqrt(x1 D) (numbering 'shift', parameter t) of the configuration th of the block S, read off each
    of its three other angles: cot(theta/2) = (n0 + n1 r)/d, i.e. sin(theta/2) (n0 + n1 r) = cos(theta/2) d (no division by a
    small sine or d: this holds for a folded crease and at a pole too)."""
    out = []
    for i, (n0, n1, d) in S.cots(signs, 1, t, shift=shift, linear=True).items():
        sh, ch = math.sin(th[i]/2), math.cos(th[i]/2)
        if sh*n1 != 0 and math.isfinite(n0) and math.isfinite(d): out.append((ch*d - sh*n0)/(sh*n1))
    return out

def realize_from(S_new, S_old, th_old, direction, signs, keep_old=True):
    """the realization of the block S_new from its realized neighbour S_old (dihedral angles th_old), with the parameter and the
    sign for which two neighbours agree on their common faces. In the numbering of the
    pair (a column: the lower block renumbered cyclically by 2, the upper one kept; a row: the left block by 3, the right one by 1)
    the two polyhedra have the same parameter, and the sign of the lower (left) block is the sign e0 of the upper (right) block
    times -e3 e2' (a column) or -e4 e3' (a row), where e_i are the signs of the lower (left) block and e_i' those of the other; all
    blocks have the same signs here. direction: 'up', 'down', 'right' or 'left', from S_old to S_new. None if the old block
    cannot be reproduced or the formulas of the new one have no value there.
    The sign of the old block is the one whose angles come closest to th_old (the largest angle_gap of the four), among those
    that reproduce it: within 1e-6 relative in cot(theta/2), or within REALIZE_ANGLE_TOL in angle (a crease close to folded,
    theta ~ 0, has a huge cotangent that rounding moves by more than 1e-6 relative). The closest, not the first: where the two
    signs almost coincide (near a branch point of the pair; near the flat state t = 0 they can differ by O(t) only) the first
    one could be the mirror branch.
    Near a branch point of the pair (D(tp) ~ 0) the parameter hardly determines the configuration: at a relative distance rho
    sqrt(x1 D) turns an error in tp (rounding; the old block's own error) into one 1/sqrt(rho) times larger, and near t = 0
    the two branches differ by less than that. Nor do the data: the two blocks of a pair have the same D in exact arithmetic,
    but each block computes it from its own x1 and u = 1 - M (M from its own vertex 1), which differ from those of the other
    block by a few ulps; where |u| is small (M close to 1) that is ~1e-10 relative, and at rho ~ 1e-10 from a zero of the factor
    1 - u x1 t^2 the two values of D differ by O(1) (the new block's own sign then misses the old block by ~1e-6).
    So near a branch point (rho <= REALIZE_NEAR) the signed root r = e0 sqrt(x1 D) of the old block, read off its angles (or its
    own sign times the root of its own D), is carried to the new one (times the factor of the signs), provided it reproduces the
    old block (within REALIZE_ANGLE_TOL) as well as the best sign does (or to REALIZE_EXACT); at the branch point itself
    (rho <= REALIZE_BRANCH), and when no sign reproduces the old block at all, whenever it reproduces it.
    With keep_old, wherever the result differs from that of the rule before (the first sign within 1e-6 in cot(theta/2)) by at
    most REALIZE_AGREE, the latter is returned, to the last bit: the new choices only act where the old one is off. synchronize
    turns this off next to the flat end (|t| < REALIZE_KEEP_T), which the tool did not reach before: there the rule before can be
    off by up to ~1e-7 per step, and the better result is kept.
    A common crease at theta ~ 0 (folded) has the parameter cot(theta/2) = +-inf or huge: the formulas take the limit."""
    f_col, f_row = -signs[2]*signs[1], -signs[3]*signs[2]
    shift_old, shift_new, factor = {'up': (2, 0, f_col), 'down': (0, 2, f_col), 'right': (3, 1, f_row), 'left': (1, 3, f_row)}[direction]
    def cot_half(a_): s_ = math.sin(a_/2); return math.cos(a_/2)/s_ if abs(s_) > 1e-12 else None
    def gap_of(c): a = S_old.angles_from_cots(c); return max(angle_gap(a[i], th_old[i]) for i in range(1, 5))
    def new_block(sign, root=None):
        try: return S_new.angles_from_cots(S_new.cots(signs, sign, tp, shift=shift_new, root=root))
        except (ValueError, ZeroDivisionError): return None
    tp = cot_half(th_old[1 + shift_old])                                            # the parameter, in the numbering of the pair
    if tp is None:                                                                   # the common crease is (almost) folded
        s_ = math.sin(th_old[1 + shift_old]/2); tp = math.cos(th_old[1 + shift_old]/2)/s_ if s_ != 0 else math.inf
    target = {i: cot_half(th_old[i]) for i in range(1, 5)}
    s_old = None; best = None; first = None                                          # the sign of the old block in that numbering
    for e0c in (1, -1):
        try: c = S_old.cots(signs, e0c, tp, shift=shift_old)
        except (ValueError, ZeroDivisionError): continue
        gap = gap_of(c); in_cot = all(target[i] is None or abs(c[i] - target[i]) < 1e-6*(1 + abs(c[i])) for i in range(1, 5))
        if in_cot and first is None: first = e0c                                     # the choice of the first acceptable sign (the rule before)
        if (gap <= REALIZE_ANGLE_TOL or in_cot) and (best is None or gap < best): s_old, best = e0c, gap
    th_first = new_block(factor*first) if first is not None else None               # the factor is +-1: its own inverse
    th_new = th_first if s_old == first else (new_block(factor*s_old) if s_old is not None else None)
    x1p, u = S_old.shifted_data(shift_old)[0], S_old.u                               # rho: relative distance from a branch point
    rho = min(abs(x1p*tp*tp - 1)/(abs(x1p*tp*tp) + 1), abs(1 - u*x1p*tp*tp)/(1 + abs(u*x1p*tp*tp))) if abs(tp) <= FLEX_T_INF else math.inf
    at_branch = rho <= REALIZE_BRANCH
    if best is None or rho <= REALIZE_NEAR:                                          # carry the signed root
        r_old = None; r_best = None
        try: roots = _roots_from_angles(S_old, signs, tp, shift_old, th_old)
        except (ValueError, ZeroDivisionError): roots = []
        if s_old is not None:                                                        # the old block's own: its sign, the root of its D
            try: roots.append(s_old*rsqrt(x1p*(x1p*tp*tp - 1)*(1 - u*x1p*tp*tp)))
            except ValueError: pass
        for r in roots:
            try: gap = gap_of(S_old.cots(signs, 1, tp, shift=shift_old, root=r))
            except (ValueError, ZeroDivisionError): continue
            if r_best is None or gap < r_best: r_old, r_best = r, gap
        if r_old is not None and r_best <= REALIZE_ANGLE_TOL and (best is None or at_branch or r_best <= max(best, REALIZE_EXACT)):
            th_root = new_block(1, root=factor*r_old)
            if th_root is not None: th_new = th_root
    if th_new is None: return None
    if keep_old and th_first is not None and max(angle_gap(th_first[i], th_new[i]) for i in range(1, 5)) <= REALIZE_AGREE: return th_first
    return th_new                                                                    # (where the rule before is as good, its result is kept to the last bit)

def synchronize(subs, base_kn, signs, e0, t):
    """the dihedral angles {block: {i: theta_i}} of all blocks at the parameter t, or None. The base block (kappa_0, nu_0) is
    realized at the parameter t and the sign e0 by the flexion formulas; then the blocks of its row, and then those of every
    column, each from its realized neighbour by realize_from (the parameter and the sign for which neighbours agree).
    Finally every two blocks that share faces are checked to agree on them (their dihedral angles along the common edges)."""
    ks = sorted({k for k, _ in subs}); ns = sorted({nu for _, nu in subs}); k0, n0 = base_kn
    th = {}
    try: th[base_kn] = subs[base_kn].angles_from_cots(subs[base_kn].cots(signs, e0, t))
    except (ValueError, ZeroDivisionError): return None
    steps = ([((k, n0), (k - 1, n0), 'right') for k in ks if k > k0] + [((k, n0), (k + 1, n0), 'left') for k in reversed(ks) if k < k0])
    for k in ks:
        steps += [((k, nu), (k, nu - 1), 'up') for nu in ns if nu > n0] + [((k, nu), (k, nu + 1), 'down') for nu in reversed(ns) if nu < n0]
    for new, old, direction in steps:
        th_new = realize_from(subs[new], subs[old], th[old], direction, signs, keep_old=abs(t) >= REALIZE_KEEP_T)
        if th_new is None: return None
        th[new] = th_new
    pts = {}
    for kn in th:
        pts[kn], clo = block_points(subs[kn], th[kn])
        if clo > 1e-7: return None
    for kn in th:                                                                    # every pair of blocks with common faces agrees
        for other in th:
            if other <= kn or abs(other[0] - kn[0]) > 2 or abs(other[1] - kn[1]) > 2: continue
            if not compatible(subs[kn], th[kn], pts[kn], subs[other], th[other], pts[other]): return None
    return th

# ============================================================================================
# 6. THE NET IN SPACE WITH CONVEX FACES (built face by face)
# ============================================================================================

def inner_edge_lengths(ang, m, n, base_kn, a1=1.0):
    """strictly positive edge lengths for the edges of the central faces F_ij, 1 <= i <= m-2, 1 <= j <= n-2, that make every central
    face a closed planar quadrilateral with the prescribed angles (two linear closure equations per face). The solution space is a
    linear subspace; a positive point of it is found by alternating projections onto the subspace and onto {lengths >= 1};
    the result is scaled so that the edge V_{k+1,n} V_{kn} of the base central face has length a1. None if no positive solution."""
    faces = [(i, j) for i in range(1, m-1) for j in range(1, n-1)]
    def corners(f): i, j = f; return [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
    edges = sorted({frozenset((c[k], c[(k+1) % 4])) for f in faces for c in [corners(f)] for k in range(4)}, key=lambda e: sorted(e))
    idx = {e: k for k, e in enumerate(edges)}
    A = np.zeros((2*len(faces), len(edges)))
    for r, f in enumerate(faces):
        c = corners(f); th = 0.0
        for k in range(4):
            e = frozenset((c[k], c[(k+1) % 4]))
            A[2*r, idx[e]] += math.cos(th); A[2*r + 1, idx[e]] += math.sin(th)
            th += PI - ang[c[(k+1) % 4]][f]                                  # exterior angle at the next corner
    # orthonormal basis of the null space
    U, S, Vt = np.linalg.svd(A)
    rank = int((S > 1e-10*S[0]).sum()); N = Vt[rank:].T                       # (E, d) basis of the solution space
    if N.shape[1] == 0: return None
    x = np.ones(len(edges))
    for _ in range(3000):                                                        # alternating projections: subspace / {x >= 1}
        x = N @ (N.T @ x)
        if x.min() >= 1 - 1e-9: break
        x = np.maximum(x, 1.0)
    x = N @ (N.T @ x)
    if x.min() <= 1e-9*x.max(): return None
    k0, n0 = base_kn; e0 = frozenset(((k0 + 1, n0), (k0, n0)))
    x = x*(a1/x[idx[e0]])
    return {e: float(x[idx[e]]) for e in edges}

def side_spread(ang, m, n, base_kn):
    """longest / shortest side of the central faces (inf when they cannot be closed): above ~1e6 a failed construction is due to
    the angles, which force sides too different for double precision, not to rounding near a condition"""
    try: L = central_face_lengths(ang, m, n, base_kn)
    except (ArithmeticError, ValueError): return math.inf
    return max(L.values())/min(L.values()) if L and min(L.values()) > 0 else math.inf

def central_face_lengths(ang, m, n, base_kn, a1=1.0):
    """the sides of the central faces F_{kappa nu}, closed row by row. In the block (kappa, nu), the
    vertices A_1, ..., A_4 are V_{kappa+1,nu}, V_{kappa nu}, V_{kappa,nu+1}, V_{kappa+1,nu+1} with the angles delta_1, ..., delta_4, and
        |A_3A_4| = (|A_1A_2| sin delta_1 - |A_2A_3| sin(delta_1 + delta_2))/sin delta_4       (upper side),
        |A_4A_1| = (|A_2A_3| sin delta_3 - |A_1A_2| sin(delta_2 + delta_3))/sin delta_4       (right side).
    The lower sides of the first row of blocks are 1; then, row by row, the left side of the first block is taken large enough
    (the bound that makes all right sides of the row positive, plus the mean lower side of the row, so that the shortest side of
    the row is not much shorter than the others), and the formulas give the others, from left to right; the upper sides are
    positive because delta_1 + delta_2 >= pi. Finally all sides are scaled so that the edge
    V_{kappa_0+1,nu_0} V_{kappa_0 nu_0} has length a1. None if some side is not positive."""
    E = lambda P, Q: frozenset((P, Q)); L = {}
    def delta(k, nu, i):
        A = {1: (k+1, nu), 2: (k, nu), 3: (k, nu+1), 4: (k+1, nu+1)}
        return ang[A[i]][(k, nu)]
    for k in range(1, m-1): L[E((k, 1), (k+1, 1))] = 1.0                                   # the lower sides of the first row
    for nu in range(1, n-1):
        # the right side of the block (k, nu) is a_k X - b_k, X = the left side of the first block of the row
        a_, b_, bound = 1.0, 0.0, 0.0
        for k in range(1, m-1):
            d2, d3, d4 = delta(k, nu, 2), delta(k, nu, 3), delta(k, nu, 4); low = L[E((k, nu), (k+1, nu))]
            a_, b_ = a_*math.sin(d3)/math.sin(d4), (b_*math.sin(d3) + low*math.sin(d2 + d3))/math.sin(d4)
            bound = max(bound, b_/a_)
        mean_low = float(np.mean([L[E((k, nu), (k+1, nu))] for k in range(1, m-1)]))
        L[E((1, nu), (1, nu+1))] = bound + mean_low                                     # above the bound by a typical side
        for k in range(1, m-1):
            d1, d2, d3, d4 = (delta(k, nu, i) for i in (1, 2, 3, 4))
            low, lft = L[E((k, nu), (k+1, nu))], L[E((k, nu), (k, nu+1))]
            L[E((k, nu+1), (k+1, nu+1))] = (low*math.sin(d1) - lft*math.sin(d1 + d2))/math.sin(d4)    # upper side
            L[E((k+1, nu), (k+1, nu+1))] = (lft*math.sin(d3) - low*math.sin(d2 + d3))/math.sin(d4)    # right side
    if min(L.values()) <= 0: return None
    k0, n0 = base_kn; sc = a1/L[E((k0 + 1, n0), (k0, n0))]
    return {e: v*sc for e, v in L.items()}

def build_net(ang, thetas, m, n, base_kn, a1=1.0, free=None):
    """positions of all vertices; 'free' records the free lengths chosen for the boundary faces (reused at every t)."""
    pos = {}; Lref = a1
    record = free is None
    if record: free = {}
    def inner(v): return 1 <= v[0] <= m-1 and 1 <= v[1] <= n-1
    def corners(f): i, j = f; return [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
    def normal(f): c = [pos[v] for v in corners(f)]; return unit(np.cross(c[1]-c[0], c[3]-c[0]))
    def dih_index(edge, f):
        k, nn = f; A = {1: (k+1, nn), 2: (k, nn), 3: (k, nn+1), 4: (k+1, nn+1)}
        for i in range(1, 5):
            if edge == frozenset((A[i], A[i+1 if i < 4 else 1])): return i
    def convex2d(pts):
        e = [(pts[(k+1)%4]-pts[k]).tolist() for k in range(4)]
        cr = [(e[k][0]*e[(k+1)%4][1] - e[k][1]*e[(k+1)%4][0])/(math.hypot(*e[k])*math.hypot(*e[(k+1)%4]) or 1.0) for k in range(4)]   # sines of the turning angles (scale-free: faces with very short sides)
        return all(c > 1e-9 for c in cr) or all(c < -1e-9 for c in cr)
    # The construction is done as originally (the base face explicitly, heuristic lengths for the faces attached
    # with two known corners, boundary strips with a list of free lengths). Only if that fails, the inner region is rebuilt with edge
    # lengths solved from the closure equations of all central faces, and the strips with balanced completions and backtracking.
    inner_mode = free.get('inner_mode', 'legacy') if not record else 'legacy'
    Lin = free.get('inner_lengths') if (not record and inner_mode == 'solved') else None
    k0, n0 = base_kn
    def place_base_face():
        d1, d2, d3 = ang[(k0+1, n0)][(k0, n0)], ang[(k0, n0)][(k0, n0)], ang[(k0, n0+1)][(k0, n0)]
        if Lin is None:
            A = convex_quad(d1, d2, d3, a1)
            pos[(k0, n0)], pos[(k0+1, n0)], pos[(k0, n0+1)], pos[(k0+1, n0+1)] = A[2], A[1], A[3], A[4]
        else:
            la1, la2 = Lin[frozenset(((k0 + 1, n0), (k0, n0)))], Lin[frozenset(((k0, n0), (k0, n0 + 1)))]
            A2 = np.zeros(3); A1 = np.array([la1, 0, 0.]); A3 = la2*np.array([math.cos(d2), math.sin(d2), 0])
            d4 = 2*PI - d1 - d2 - d3; a3 = (la2*math.sin(d3) - la1*math.sin(d2 + d3))/math.sin(d4)
            A4 = A1 + a3*np.array([-math.cos(d1), math.sin(d1), 0])
            pos[(k0, n0)], pos[(k0+1, n0)], pos[(k0, n0+1)], pos[(k0+1, n0+1)] = A2, A1, A3, A4
    place_base_face()
    def attach2(f, P, Q, oldf, booster=1.0):
        """new face f sharing the edge PQ with the built central face oldf (plane from the dihedral angle of oldf's block)."""
        cs = corners(f); ip, iq = cs.index(P), cs.index(Q)
        R, S = cs[(ip - (iq - ip)) % 4], cs[(iq + (iq - ip)) % 4]
        th = thetas[oldf][dih_index(frozenset((P, Q)), oldf)]
        u = unit(pos[Q] - pos[P]); Mo = sum(pos[v] for v in corners(oldf))/4
        dold = Mo - pos[P]; dold = unit(dold - np.dot(dold, u)*u); N = normal(oldf)
        dnew = math.cos(th)*dold + math.sin(th)*N
        phP, phQ = ang[P][f], ang[Q][f]
        if not inner(R) and not inner(S) and strip_mode[0] == 'apex':
            # a boundary face with two free corners: both at lengths that keep this face and its neighbours convex
            lP = free.setdefault(('len', f, P, R), safe_len(P, R)) if record else free[('len', f, P, R)]
            lQ = free.setdefault(('len', f, Q, S), safe_len(Q, S)) if record else free[('len', f, Q, S)]
            pos[R] = pos[P] + lP*(math.cos(phP)*u + math.sin(phP)*dnew)
            pos[S] = pos[Q] + lQ*(-math.cos(phQ)*u + math.sin(phQ)*dnew)
            return
        phR = ang[R][f] if inner(R) else (2*PI - phP - phQ)/2
        phS = 2*PI - phP - phQ - phR
        assert 0 < phR < PI and 0 < phS < PI
        L = np.linalg.norm(pos[Q] - pos[P])
        lP = (free.setdefault(('len', f, P, R), booster*Lref) if record else free[('len', f, P, R)]) if not inner(R) else (Lin[frozenset((P, R))] if Lin is not None else inner_boost[0]*(lambda d1, d2, d3: (L if d2+d3 >= PI and d1+d2 >= PI else 0.8*L*math.sin(d1)/math.sin(d1+d2) if d2+d3 >= PI else 1.2*L*math.sin(d2+d3)/math.sin(d3) if d1+d2 >= PI else L*(math.sin(d2+d3)/math.sin(d3) + math.sin(d1)/math.sin(d1+d2))/2))(phQ, phP, phR))
        lQ = (lP*math.sin(phR) - L*math.sin(phP + phR))/math.sin(phS)
        assert lP > 0 and lQ > 0
        pos[R] = pos[P] + lP*(math.cos(phP)*u + math.sin(phP)*dnew)
        pos[S] = pos[Q] + lQ*(-math.cos(phQ)*u + math.sin(phQ)*dnew)
        # positive lengths lP, lQ do not make the face convex: its fourth side RS can come out reversed (a crossed face)
        quad = [pos[P], pos[Q], pos[S], pos[R]]; nq = np.cross(quad[1] - quad[0], quad[3] - quad[0])
        crs = [np.dot(np.cross(quad[(k+1) % 4] - quad[k], quad[(k+2) % 4] - quad[(k+1) % 4]), nq) for k in range(4)]
        assert all(c_ > 1e-12*L**4 for c_ in crs) or all(c_ < -1e-12*L**4 for c_ in crs), ("nonconvex face", f)
    def attach3(f, choice=0):
        """face with three known corners; the fourth is determined (inner corner) or placed convexly (boundary corner;
        'choice' selects among the best-balanced convex completions, for the backtracking along the boundary strips)."""
        cs = corners(f); known = [v in pos for v in cs]
        assert sum(known) == 3
        ix = known.index(False); X, K1, K2, K3 = cs[ix], cs[(ix+1) % 4], cs[(ix+2) % 4], cs[(ix+3) % 4]
        u = unit(pos[K1] - pos[K2]); w = pos[K3] - pos[K2]; w = unit(w - np.dot(w, u)*u)
        L1, L3 = np.linalg.norm(pos[K1] - pos[K2]), np.linalg.norm(pos[K3] - pos[K2])
        ph2 = math.acos(np.clip(np.dot(unit(pos[K1]-pos[K2]), unit(pos[K3]-pos[K2])), -1, 1))
        if inner(K2): assert abs(ph2 - ang[K2][f]) < 1e-6, ("inconsistent configuration at", K2)
        K1p, K3p = np.array([L1, 0.]), L3*np.array([math.cos(ph2), math.sin(ph2)])
        def ok(Xp): return convex2d([K1p, np.zeros(2), K3p, Xp])
        if inner(K1) and inner(K3):
            ph1, ph3 = ang[K1][f], ang[K3][f]
            dir1 = np.array([-math.cos(ph1), math.sin(ph1)]); dir3 = -np.array([math.cos(ph2+ph3), math.sin(ph2+ph3)])
            s, r = np.linalg.solve(np.array([[dir1[0], -dir3[0]], [dir1[1], -dir3[1]]]), K3p - K1p)
            assert s > 0 and r > 0; Xp = K1p + s*dir1
        elif inner(K1) or inner(K3):
            if inner(K1): base, dr, K = K1p, np.array([-math.cos(ang[K1][f]), math.sin(ang[K1][f])]), K1
            else: ph3 = ang[K3][f]; base, dr, K = K3p, -np.array([math.cos(ph2+ph3), math.sin(ph2+ph3)]), K3
            key = ('len', f, K, X)
            if not record:
                Xp = base + free[key]*dr
            else:
                # the free length: among the convex completions, the one whose two free angles (at X and at the other boundary
                # corner) are most balanced, which keeps the boundary faces well shaped and leaves room for the next face
                def free_angles(Xq):
                    P4 = [K1p, np.zeros(2), K3p, Xq]; out_ = []
                    for k in (0, 3):                                                   # corners K1 (index 0) and X (index 3)
                        a_, b_ = P4[(k-1) % 4] - P4[k], P4[(k+1) % 4] - P4[k]
                        out_.append(math.acos(max(-1.0, min(1.0, float(np.dot(a_, b_))/(np.linalg.norm(a_)*np.linalg.norm(b_))))))
                    return out_
                if strip_mode[0] == 'apex':                                            # the length that keeps both faces at this edge convex
                    Xp = base + safe_len(K, X)*dr
                    assert choice == 0 and ok(Xp), ("no convex completion", f)
                elif strip_mode[0] == 'legacy':                                        # as originally: the first convex length of the list
                    Xp = next((base + fac*Lref*dr for fac in (0.8, 0.6, 1.0, 0.45, 1.3, 0.3, 1.7, 0.2, 2.5, 0.12) if ok(base + fac*Lref*dr)), None)
                    assert Xp is not None and choice == 0, ("no convex completion", f)
                else:
                    cand = []
                    for fac in np.geomspace(0.01, 10.0, 120):
                        Xq = base + fac*Lref*dr
                        if not ok(Xq): continue
                        a1_, a2_ = free_angles(Xq)
                        score = abs(a1_ - a2_) + (0.0 if min(a1_, a2_) > 0.35 else 10.0)   # avoid very sharp free corners
                        cand.append((score, fac, Xq))
                    cand.sort(key=lambda c: c[0])
                    picks = [cand[0]] if cand else []                                       # the best, then spread-out alternatives
                    for c in cand[1:]:
                        if len(picks) >= STRIP_CHOICES: break
                        if all(abs(math.log(c[1]/p[1])) > 0.45 for p in picks): picks.append(c)
                    assert choice < len(picks), ("no convex completion", f)
                    Xp = picks[choice][2]
                free[key] = float(np.linalg.norm(Xp - base))
        else:
            Xp = K1p + K3p
        assert ok(Xp), ("nonconvex face", f)
        pos[X] = pos[K2] + Xp[0]*u + Xp[1]*w
    # inner block: breadth-first over the central faces. The faces attached with two known corners get their shape from a
    # heuristic length (scaled by inner_boost); if the forced lengths then make some face nonconvex, the block is rebuilt with
    # another scale (the scale that worked is recorded in 'free' and reused at every t, so that the faces stay congruent).
    central = [(k, nn) for k in range(1, m-1) for nn in range(1, n-1)]
    inner_boost = [free.get('inner_boost', 1.0)] if not record else [1.0]
    def build_inner():
        built_ = {base_kn}; pending = [f for f in central if f != base_kn]
        while pending:
            progress = False
            for f in list(pending):
                kn_known = [v in pos for v in corners(f)]
                if sum(kn_known) >= 3:
                    attach3(f); built_.add(f); pending.remove(f); progress = True
                elif sum(kn_known) == 2:
                    cs = corners(f)
                    for j in range(4):
                        P, Q = cs[j], cs[(j+1) % 4]
                        if P in pos and Q in pos:
                            oldf = next(g for g in built_ if P in corners(g) and Q in corners(g))
                            attach2(f, P, Q, oldf); built_.add(f); pending.remove(f); progress = True; break
            assert progress, "cannot build the inner block"
        return built_
    built = None
    if record:
        Lin = central_face_lengths(ang, m, n, base_kn, a1)                              # the central faces, closed row by row
        if Lin is not None:
            inner_mode = 'solved'; pos.clear(); place_base_face()
            try: built = build_inner()
            except AssertionError: built = None; Lin = None; inner_mode = 'legacy'
        if built is None:                                                             # (a safeguard: other lengths)
            for sc in [1.0, 0.8, 1.25, 0.65, 1.55, 0.5, 1.9, 0.4, 2.4, 0.3]:            # the original construction, several scales
                inner_boost[0] = sc; pos.clear(); place_base_face()
                try: built = build_inner(); break
                except AssertionError: pass
        if built is None:                                                             # fallback: the solved edge lengths
            Lin = inner_edge_lengths(ang, m, n, base_kn, a1)
            assert Lin is not None, "no positive edge lengths make all central faces planar quadrilaterals with the prescribed angles"
            inner_mode = 'solved'; pos.clear(); place_base_face()
            built = build_inner()
        free['inner_mode'], free['inner_boost'], free['inner_lengths'] = inner_mode, inner_boost[0], Lin
    else:
        built = build_inner()
    Lref = float(np.mean([np.linalg.norm(pos[(i,j)] - pos[(i+1,j)]) for i in range(1, m-1) for j in range(1, n)] +
                         [np.linalg.norm(pos[(i,j)] - pos[(i,j+1)]) for i in range(1, m) for j in range(1, n-1)]))
    # boundary strips: bottom (j = 0), top (j = n-1), left (i = 0), right (i = m-1), then corners
    strip_mode = [free.get('strip_mode', 'legacy') if not record else 'legacy']
    faces_net = [(a, b) for a in range(m) for b in range(n)]
    def safe_len(v, w):
        """length of the boundary edge from the inner vertex v to the boundary vertex w: in each face at this edge whose other
        corner next to v is inner too, the two edges from that base converge to a point unless the base angles add up to pi or
        more; the length stays below half the distance to that point (and below 0.8 of the mean inner edge), so that every
        boundary face is convex whatever the angles"""
        lim = 0.8*Lref
        for g in faces_net:
            cg = corners(g)
            if v not in cg or w not in cg: continue
            kv = cg.index(v); vv = cg[(kv + 1) % 4] if cg[(kv - 1) % 4] == w else cg[(kv - 1) % 4]
            if not inner(vv) or vv not in pos: continue
            s_ = ang[v][g] + ang[vv][g]
            if s_ < PI - 1e-9: lim = min(lim, 0.45*np.linalg.norm(pos[vv] - pos[v])*math.sin(ang[vv][g])/math.sin(s_))
        return lim
    def strip(faces):
        f0 = faces[0]; cs = corners(f0)
        P, Q = next((cs[j], cs[(j+1) % 4]) for j in range(4) if cs[j] in pos and cs[(j+1) % 4] in pos)
        oldf = next(g for g in built if P in corners(g) and Q in corners(g))
        def undo(g):
            for v in corners(g):
                if not inner(v) and v in pos and not any(v in corners(h) for h in built_faces): del pos[v]
            if record:
                for key in [k for k in free if k[1] == g]: del free[key]
        built_faces = set(); budget = [0]
        def place(k):                                                            # depth-first over the faces of the strip
            if k == len(faces): return True
            g = faces[k]
            for choice in range(STRIP_CHOICES if (record and strip_mode[0] == 'balanced') else 1):
                budget[0] -= 1
                if budget[0] < 0: return False                                   # the search used up its budget: the next mode takes over
                try: attach3(g, choice)
                except AssertionError:
                    undo(g); return False if not record else False
                built_faces.add(g)
                if place(k + 1): return True
                built_faces.discard(g); undo(g)
            return False
        modes = ['legacy', 'balanced', 'apex'] if record else [free.get(('strip_mode', f0), 'legacy')]
        for mode in modes:                                                       # 'apex': the last resort (it can still fail on nearly degenerate nets)
            strip_mode[0] = mode; budget[0] = STRIP_BUDGET if (record and mode == 'balanced') else math.inf
            for booster in ((0.8, 0.6, 1.0, 0.45, 1.3, 0.3, 1.7, 0.2, 2.5, 0.12) if (record and mode != 'apex') else (None,)):
                try:
                    if booster is not None: attach2(f0, P, Q, oldf, booster)
                    else: attach2(f0, P, Q, oldf)
                except AssertionError:
                    undo(f0); continue
                built_faces.add(f0)
                if place(1):
                    if record: free[('strip_mode', f0)] = mode
                    return
                built_faces.discard(f0); undo(f0)
        raise AssertionError(("strip failed", faces))
    strip([(i, 0) for i in range(1, m-1)]); strip([(i, n-1) for i in range(1, m-1)])
    strip([(0, j) for j in range(1, n-1)]); strip([(m-1, j) for j in range(1, n-1)])
    for f in [(0, 0), (m-1, 0), (0, n-1), (m-1, n-1)]: attach3(f)
    # checks: planarity of all faces and the dihedral angles of all inner edges against the blocks' values
    worst = 0.0
    for f in central:
        k, nn = f; A = {1: (k+1, nn), 2: (k, nn), 3: (k, nn+1), 4: (k+1, nn+1)}
        for i in range(1, 5):
            P, Q = A[i], A[i+1 if i < 4 else 1]
            g = next(g for g in [(a, b) for a in range(m) for b in range(n)] if g != f and P in corners(g) and Q in corners(g))
            worst = max(worst, abs((PI - math.acos(np.clip(np.dot(normal(f), normal(g)), -1, 1))) - abs(thetas[f][i])))
    plan = max(abs(np.dot(pos[c[2]] - pos[c[0]], normal(f)))/np.linalg.norm(pos[c[2]] - pos[c[0]]) for f in [(a, b) for a in range(m) for b in range(n)] for c in [corners(f)])
    return pos, worst, plan, free


# ============================================================================================
# 7. RENDERING (matplotlib has no z-buffer: painter's algorithm on subdivided faces, exact hidden-line test for the edges)
# ============================================================================================
def project(ax, pts):
    """screen coordinates (x, y, depth) of 3D points with the current projection of the axes
    (smaller depth = closer: matplotlib's projected z grows away from the viewer)."""
    from mpl_toolkits.mplot3d import proj3d
    M = ax.get_proj(); pts = np.asarray(pts, dtype=float)
    tx, ty, tz = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2], M)
    return np.stack([tx, ty, tz], axis=1)

def screen_faces(ax, pos, faces_all, corners):
    """for every face: its projected polygon (4 x 2), the affine depth function (a, b, c) with depth = a x + b y + c, and its corners."""
    out = {}
    for f in faces_all:
        c = corners(f); S = project(ax, [pos[v] for v in c])
        A = np.column_stack([S[:3, 0], S[:3, 1], np.ones(3)])
        try: abc = np.linalg.solve(A, S[:3, 2])
        except np.linalg.LinAlgError: abc = np.array([0, 0, S[:, 2].mean()])
        out[f] = (S[:, :2], abc, set(c))
    return out

def inside_convex(poly, X, tol=1e-12):
    """boolean array: which of the points X (n x 2) lie strictly inside the convex polygon poly (4 x 2)."""
    inside = np.ones(len(X), dtype=bool)
    area = sum((poly[k][0]*poly[(k+1) % 4][1] - poly[(k+1) % 4][0]*poly[k][1]) for k in range(4))
    sgn = 1.0 if area > 0 else -1.0
    for k in range(4):
        a, b = poly[k], poly[(k+1) % 4]
        cr = (b[0]-a[0])*(X[:, 1]-a[1]) - (b[1]-a[1])*(X[:, 0]-a[0])
        inside &= (sgn*cr > tol)
    return inside

def occlusion_counts(M, sf, pts3, skip=()):
    """for each 3D point: the number of faces (other than skip) that are in front of it on the screen, with the projection M."""
    from mpl_toolkits.mplot3d import proj3d
    others = [f for f in sf if f not in skip]
    A = np.array([sf[f][0] for f in others]); Bp = np.roll(A, -1, axis=1)          # (F, 4, 2): the edges a -> b of the projected faces
    abc = np.array([sf[f][1] for f in others])                                       # (F, 3): affine depth functions
    sgn = np.where((A[:, :, 0]*Bp[:, :, 1] - Bp[:, :, 0]*A[:, :, 1]).sum(axis=1) > 0, 1.0, -1.0)
    pts3 = np.asarray(pts3, dtype=float)
    tx, ty, tz = proj3d.proj_transform(pts3[:, 0], pts3[:, 1], pts3[:, 2], M)
    cr = (Bp[:, :, 0, None] - A[:, :, 0, None])*(ty[None, None, :] - A[:, :, 1, None]) - (Bp[:, :, 1, None] - A[:, :, 1, None])*(tx[None, None, :] - A[:, :, 0, None])
    inside = np.all(sgn[:, None, None]*cr > 1e-12, axis=1)                                        # (F, n): strictly inside the face
    depth = abc[:, 0, None]*tx[None, :] + abc[:, 1, None]*ty[None, :] + abc[:, 2, None]          # (F, n): depth of the face there
    return (inside & (depth < tz[None, :] - 1e-9)).sum(axis=0)                                   # a point on its own face is never counted

def edge_segments(ax, P, Q, PQ_faces, sf, nsamp=32):
    """sub-segments of the edge PQ grouped by the number of faces in front of them on the screen
    ({0: visible, 1, 2, 3: three or more}); the transitions are located by bisection."""
    M = ax.get_proj()
    def level(ts): return np.minimum(occlusion_counts(M, sf, P[None, :] + ts[:, None]*(Q - P)[None, :], PQ_faces), 3)
    ts = (np.arange(nsamp) + 0.5)/nsamp
    lvl = level(ts)
    segs = {0: [], 1: [], 2: [], 3: []}
    start, cur = 0.0, lvl[0]
    for k in range(1, nsamp):
        if lvl[k] != cur:
            a, b = ts[k-1], ts[k]
            for _ in range(10):
                c = (a + b)/2
                if level(np.array([c]))[0] == cur: a = c
                else: b = c
            tm = (a + b)/2
            segs[cur].append((P + start*(Q - P), P + tm*(Q - P))); start, cur = tm, lvl[k]
    segs[cur].append((P + start*(Q - P), Q))
    return segs

def faces_intersect(A, B, tol=1e-9):
    """True if the convex planar quadrilaterals A, B (4 x 3 arrays, no common vertex) intersect (penetrate each other)."""
    nB = np.cross(B[1] - B[0], B[3] - B[0]); nB /= np.linalg.norm(nB)
    dist = np.dot(A - B[0], nB)
    scale = max(np.linalg.norm(A[2] - A[0]), np.linalg.norm(B[2] - B[0]))
    if np.all(dist > tol*scale) or np.all(dist < -tol*scale): return False
    if np.all(np.abs(dist) < tol*scale): return False          # coplanar: not counted as penetration
    pts = []                                                     # segment A ∩ plane(B)
    for k in range(4):
        d1, d2 = dist[k], dist[(k+1) % 4]
        if d1*d2 < 0: pts.append(A[k] + (d1/(d1 - d2))*(A[(k+1) % 4] - A[k]))
        elif abs(d1) <= tol*scale: pts.append(A[k])
    if len(pts) < 2: return False
    pts = np.array(pts); P, Q = pts[0], pts[np.argmax(np.linalg.norm(pts - pts[0], axis=1))]
    if np.linalg.norm(Q - P) < tol*scale: return False
    e1 = B[1] - B[0]; e1 /= np.linalg.norm(e1); e2 = np.cross(nB, e1)
    B2 = np.stack([np.dot(B - B[0], e1), np.dot(B - B[0], e2)], axis=1)
    P2 = np.array([np.dot(P - B[0], e1), np.dot(P - B[0], e2)]); Q2 = np.array([np.dot(Q - B[0], e1), np.dot(Q - B[0], e2)])
    t0, t1 = 0.0, 1.0                                            # clip the segment against the convex polygon B (2D)
    for k in range(4):
        a, b = B2[k], B2[(k+1) % 4]; c = B2[(k+2) % 4]
        nrm = np.array([-(b[1]-a[1]), b[0]-a[0]])
        if np.dot(c - a, nrm) < 0: nrm = -nrm                    # inward normal of the edge
        fP, fQ = np.dot(P2 - a, nrm), np.dot(Q2 - a, nrm)
        if fP < 0 and fQ < 0: return False
        if fP < 0: t0 = max(t0, fP/(fP - fQ))
        elif fQ < 0: t1 = min(t1, fP/(fP - fQ))
        if t0 >= t1 - 1e-9: return False
    return (t1 - t0)*np.linalg.norm(Q - P) > tol*scale

def corners_intersect(A, B, ka, kb, tol=1e-9):
    """True if the convex planar quadrilaterals A, B (4 x 3 arrays) with the single common vertex A[ka] = B[kb] penetrate each other.
    Each face lies in its corner at that vertex, so the two meet only on the line where their planes cross, along a segment from the
    vertex: they penetrate exactly when the corner of A crosses the plane of B (its two edges strictly on opposite sides) along a
    direction strictly inside the corner of B. tol is an angle (radians): faces that only touch (at the vertex, or along an edge lying
    in the other's plane) or are coplanar are not counted, as in faces_intersect."""
    V = A[ka]
    a1, a2 = unit(A[ka-1] - V), unit(A[(ka+1) % 4] - V); b1, b2 = unit(B[kb-1] - V), unit(B[(kb+1) % 4] - V)
    nB = unit(np.cross(b1, b2))
    s1, s2 = np.dot(a1, nB), np.dot(a2, nB)                         # sines of the angles of the edges of A with the plane of B
    if not (min(s1, s2) < -tol and max(s1, s2) > tol): return False
    q = unit(abs(s2)*a1 + abs(s1)*a2)                                # the corner of A meets the plane of B along q
    return np.dot(np.cross(b1, q), nB) > tol and np.dot(np.cross(q, b2), nB) > tol

def self_intersections(pos, faces_all, corners):
    """pairs of faces that penetrate each other. Faces with a common edge cannot (each lies on one side of the edge in its own
    plane); faces with a single common vertex (diagonal neighbours) can cut through each other along a segment from that vertex,
    so their corners at that vertex are tested (corners_intersect; the common vertex itself does not count)."""
    pairs = []
    P = {f: np.array([pos[x] for x in corners(f)]) for f in faces_all}
    lo = {f: P[f].min(axis=0) for f in faces_all}; hi = {f: P[f].max(axis=0) for f in faces_all}
    for i, f in enumerate(faces_all):
        for g in faces_all[i+1:]:
            if (lo[f] > hi[g]).any() or (lo[g] > hi[f]).any(): continue           # bounding boxes do not meet
            shared = set(corners(f)) & set(corners(g))
            if len(shared) >= 2: continue
            if shared:
                v = shared.pop()
                if corners_intersect(P[f], P[g], corners(f).index(v), corners(g).index(v)): pairs.append((f, g))
            elif faces_intersect(P[f], P[g]) or faces_intersect(P[g], P[f]): pairs.append((f, g))
    return pairs

def wheel_horizontal(ev):
    """a wheel event with a modifier key means horizontal scrolling: Shift (Linux / Windows; on macOS the system turns Shift+wheel into a
    horizontal wheel that matplotlib's MacOSX backend drops), so also Option/Alt, Control and Command."""
    key = (getattr(ev, 'key', None) or '').lower()
    return any(k in key for k in ('shift', 'alt', 'ctrl', 'control', 'cmd', 'super', 'meta'))

def view_axes(elev, azim, ax=None):
    """unit vectors of the screen in data coordinates: u to the right, up to the top, w toward the viewer (as in matplotlib).
    With the axes given, the vectors of the actual projection are used (they include the roll and the flips of the camera)."""
    if ax is not None:
        try:
            ax.get_proj()
            u, up, w = (np.array(getattr(ax, n), dtype=float) for n in ('_view_u', '_view_v', '_view_w'))
            return unit(u), unit(up), unit(w)
        except Exception: pass
    e, a = math.radians(elev), math.radians(azim)
    w = np.array([math.cos(e)*math.cos(a), math.cos(e)*math.sin(a), math.sin(e)])
    V = np.array([0.0, 0.0, -1.0 if abs(elev) > 90 else 1.0])
    u = unit(np.cross(V, w))
    return u, np.cross(w, u), w

def layered_pieces(M, sf, polys, cols, max_split=2, straddle=False):
    """occlusion layer of every piece (the number of faces in front of it), with the pieces that straddle an occlusion boundary
    (9 sample points hidden by different numbers of faces) split into 4 until the counts agree or max_split is reached.
    A piece still straddling then gets its highest count, or with straddle the fractional layer (lo + hi)/2 - 0.25 between its
    lowest and highest count: drawn after the pieces hidden by hi faces and before those hidden by lo (sorted by depth in LayerPolys).
    Returns the final pieces, their colours and their layers (pieces of a higher layer are drawn earlier)."""
    out_p, out_c, out_l = [], [], []
    Pp, cc, depth = np.array(polys), np.array(cols), 0
    while len(Pp):
        c = Pp.mean(axis=1, keepdims=True)
        smp = np.concatenate([c, (Pp + c)/2, 0.9*Pp + 0.1*c], axis=1).reshape(-1, 3)
        cnt = occlusion_counts(M, sf, smp).reshape(len(Pp), 9)
        uniform = (cnt == cnt[:, :1]).all(axis=1)
        done = uniform | (depth >= max_split)
        hi, lo = np.minimum(cnt[done].max(axis=1), 3), np.minimum(cnt[done].min(axis=1), 3)
        out_p += list(Pp[done]); out_c += list(cc[done]); out_l += list(np.where(hi > lo, (hi + lo)/2 - 0.25, hi) if straddle else hi)
        Pp, cc = Pp[~done], cc[~done]
        if len(Pp):
            m01, m12, m23, m30, ce = (Pp[:, 0] + Pp[:, 1])/2, (Pp[:, 1] + Pp[:, 2])/2, (Pp[:, 2] + Pp[:, 3])/2, (Pp[:, 3] + Pp[:, 0])/2, Pp.mean(axis=1)
            Pp = np.concatenate([np.stack([Pp[:, 0], m01, ce, m30], 1), np.stack([m01, Pp[:, 1], m12, ce], 1),
                                 np.stack([ce, m12, Pp[:, 2], m23], 1), np.stack([m30, ce, m23, Pp[:, 3]], 1)])
            cc = np.concatenate([cc]*4); depth += 1
    return out_p, out_c, np.array(out_l)

class LayerPolys(PolyCollection):
    """the face pieces of one occlusion layer. Pieces of the same layer never overlap on the screen (an overlap would make the
    rear piece a member of a higher layer), so no depth sorting is needed inside a layer: the collection just projects its
    3D pieces at draw time. Pieces hidden by more faces (higher layer) are drawn earlier. A fractional layer holds pieces left
    straddling an occlusion boundary (layered_pieces with straddle): these can overlap, so they are drawn far to near."""
    def __init__(self, verts, layer=0, **kwargs):
        self.layer, self.verts3d = layer, np.asarray(verts, dtype=float)
        super().__init__(np.zeros((0, 4, 2)), **kwargs)
        self.cols3d = np.array(self.get_facecolor())                                  # the colours in the order of verts3d
    def do_3d_projection(self, *args, **kwargs):
        from mpl_toolkits.mplot3d import proj3d
        V = self.verts3d.reshape(-1, 3)
        tx, ty, tz = proj3d.proj_transform(V[:, 0], V[:, 1], V[:, 2], self.axes.M)
        P = np.stack([tx, ty], axis=1).reshape(self.verts3d.shape[0], 4, 2)
        if self.layer % 1:                                                             # straddling pieces: far to near (larger depth first)
            o = np.argsort(-tz.reshape(-1, 4).mean(axis=1), kind='stable'); P = P[o]; self.set_facecolor(self.cols3d[o])
        self.set_verts(P, True)
        return 1e8 + self.layer

class FrontLines(Line3DCollection):
    """lines always drawn last (on top of the faces); a smaller rank is drawn later."""
    def __init__(self, segments, rank=0, **kwargs):
        self.rank = rank; super().__init__(segments, **kwargs)
    def do_3d_projection(self, *args, **kwargs):
        super().do_3d_projection(*args, **kwargs); return -1e9 + self.rank

# material: near-white matte plastic with a cool grey shadow side (the look of the QS-nets figures)
# ---- material measured from the reference renders (k-means over the face pixels of nine renders)
SURFACE_RAMP = [np.array([0.980, 0.988, 0.988]),   # #FAFCFC  peak highlight        (light -> shade)
                np.array([0.969, 0.973, 0.976]),   # #F7F8F9
                np.array([0.941, 0.941, 0.929]),   # #F0F0ED
                np.array([0.906, 0.910, 0.898]),   # #E7E8E5
                np.array([0.867, 0.878, 0.875]),   # #DDE0DF
                np.array([0.831, 0.839, 0.835]),   # #D4D6D5
                np.array([0.780, 0.784, 0.776])]   # #C7C8C6  deepest face tone
LIGHT_COL   = SURFACE_RAMP[0]
SHADOW_COL  = SURFACE_RAMP[-1]
WARM_STRONG = np.array([0.886, 0.886, 0.749])    # #E2E2BF  strongest warm found (the most lit faces)
COOL_DEEP   = np.array([0.651, 0.671, 0.694])    # #A6ABB1  deep cool, rim only (grazing faces)
WARM_BOUNCE = np.array([0.965, 0.961, 0.914])    # #F6F5E9  hue of the lit end
COOL_BOUNCE = np.array([0.847, 0.890, 0.918])    # #D8E3EA  hue of the shaded end
TINT_MIX    = 0.35                               # how much of that hue is mixed in
SPEC_COL    = np.array([0.992, 1.000, 1.000])    # cool-white specular
SPEC_K      = 0.12                               # strength of the specular highlight
EDGE_COL    = (0.173, 0.267, 0.439)              # #2C4470  silhouette and front creases
EDGE_SOFT   = (0.451, 0.459, 0.463)              # #737576  interior / back creases
HIDDEN_COL  = (0.318, 0.412, 0.545)              # #51698B  hidden-line segments
GROUND_COL  = (0.400, 0.420, 0.450)              # ground shadow ink, neutral
GROUND_A    = 0.14                               # its opacity at the centre
PEN_TINT    = (np.array([0.980, 0.898, 0.882]),  # penetrating faces, same value range
               np.array([0.847, 0.694, 0.678]))
# the measured palette as defaults for the Colours dialog; the ramp keeps its shape when the two end tones are changed
_RAMP_T = [float(np.mean((c - SURFACE_RAMP[-1])/(SURFACE_RAMP[0] - SURFACE_RAMP[-1]))) for c in SURFACE_RAMP]
COLOR_DEFAULTS = dict(LIGHT_COL=LIGHT_COL.copy(), SHADOW_COL=SHADOW_COL.copy(), COOL_DEEP=COOL_DEEP.copy(), WARM_BOUNCE=WARM_BOUNCE.copy(),
                      COOL_BOUNCE=COOL_BOUNCE.copy(), TINT_MIX=TINT_MIX, SPEC_K=SPEC_K, EDGE_COL=EDGE_COL, EDGE_SOFT=EDGE_SOFT, HIDDEN_COL=HIDDEN_COL,
                      GROUND_COL=GROUND_COL, GROUND_A=GROUND_A, PEN_LIGHT=PEN_TINT[0].copy(), PEN_DARK=PEN_TINT[1].copy(),
                      FACE_COLOR='#FFFFFF', INK_COLOR='#2c4a74')

def theme(light, shadow, rim, warm, cool, mix, spec, edge, soft, hidden, ground, ground_a, face, ink='#2c4a74'):
    return dict(COLOR_DEFAULTS, LIGHT_COL=np.array(light), SHADOW_COL=np.array(shadow), COOL_DEEP=np.array(rim), WARM_BOUNCE=np.array(warm),
                COOL_BOUNCE=np.array(cool), TINT_MIX=mix, SPEC_K=spec, EDGE_COL=tuple(edge), EDGE_SOFT=tuple(soft), HIDDEN_COL=tuple(hidden),
                GROUND_COL=tuple(ground), GROUND_A=ground_a, FACE_COLOR=face, INK_COLOR=ink)

THEMES = {                                                  # picture themes (LIGHT/SHADOW/RIM/WARM/COOL, MIX, SPEC, EDGE/SOFT/HIDDEN, GROUND @ alpha, canvas)
    # The reference palette: measured tones with distinct cobalt silhouette ink
    'Paper':     dict(COLOR_DEFAULTS),
    # Clean architectural plaster (the tool's first look; its bounce hues equal its end tones, so the mix changes nothing)
    'Classic':   theme([0.985, 0.985, 0.975], [0.760, 0.790, 0.850], [0.760, 0.790, 0.850],
                       [0.985, 0.985, 0.975], [0.760, 0.790, 0.850], 0.22, 0.12,
                       (0.160, 0.270, 0.460), (0.420, 0.500, 0.620), (0.450, 0.550, 0.720),
                       (0.28, 0.31, 0.40), 0.17, '#FFFFFF'),
    # Fine white ceramic: deep navy silhouette with soft slate-blue interior creases
    'Porcelain': theme([0.980, 0.988, 0.988], [0.745, 0.776, 0.800], [0.651, 0.671, 0.694],
                       [0.957, 0.949, 0.890], [0.839, 0.886, 0.918], 0.35, 0.12,
                       (0.118, 0.208, 0.349), (0.404, 0.427, 0.463), (0.275, 0.376, 0.541),
                       (0.40, 0.42, 0.45), 0.10, '#FFFFFF'),
    # Heavy drafting vellum: ivory-white facets with crisp structural ink
    'Vellum':    theme([0.988, 0.992, 0.992], [0.880, 0.895, 0.905], [0.820, 0.835, 0.845],
                       [0.975, 0.965, 0.920], [0.890, 0.925, 0.945], 0.38, 0.12,
                       (0.141, 0.251, 0.420), (0.480, 0.520, 0.580), (0.431, 0.486, 0.576),
                       (0.42, 0.45, 0.50), 0.08, '#FFFFFF'),
    # Pure matte gypsum board: zero specular gloss with soft charcoal pencil lines
    'Chalk':     theme([0.972, 0.969, 0.961], [0.800, 0.792, 0.776], [0.714, 0.706, 0.686],
                       [0.965, 0.949, 0.910], [0.878, 0.886, 0.886], 0.25, 0.00,
                       (0.227, 0.243, 0.271), (0.520, 0.530, 0.540), (0.416, 0.431, 0.463),
                       (0.38, 0.38, 0.38), 0.09, '#FCFCFB'),
    # Mineral amethyst: gentle lavender/cool tint with deep plum-slate creases
    'Quartz':    theme([0.976, 0.973, 0.988], [0.745, 0.737, 0.784], [0.655, 0.647, 0.706],
                       [0.957, 0.941, 0.918], [0.855, 0.859, 0.918], 0.36, 0.12,
                       (0.200, 0.188, 0.369), (0.460, 0.440, 0.540), (0.322, 0.306, 0.478),
                       (0.40, 0.39, 0.47), 0.11, '#FDFCFF'),
    # Cyan/cobalt architectural schematic with vibrant structural edge ink
    'Blueprint': theme([0.969, 0.980, 0.992], [0.749, 0.788, 0.831], [0.659, 0.706, 0.765],
                       [0.949, 0.957, 0.949], [0.796, 0.863, 0.918], 0.40, 0.12,
                       (0.114, 0.243, 0.561), (0.380, 0.460, 0.560), (0.290, 0.435, 0.647),
                       (0.34, 0.40, 0.50), 0.12, '#FBFCFE'),
    # Pale jade ceramic: subtle seafoam fill with deep pine-teal edges
    'Celadon':   theme([0.965, 0.980, 0.969], [0.753, 0.800, 0.765], [0.667, 0.722, 0.686],
                       [0.945, 0.953, 0.894], [0.827, 0.886, 0.878], 0.38, 0.12,
                       (0.122, 0.310, 0.290), (0.380, 0.480, 0.460), (0.263, 0.439, 0.416),
                       (0.35, 0.45, 0.43), 0.12, '#FCFDFB'),
    # Warm aged parchment: rich cream facets with warm umber fold lines
    'Ivory':     theme([0.992, 0.984, 0.957], [0.812, 0.780, 0.722], [0.729, 0.694, 0.627],
                       [0.957, 0.922, 0.839], [0.875, 0.878, 0.863], 0.40, 0.12,
                       (0.353, 0.275, 0.196), (0.580, 0.510, 0.430), (0.541, 0.471, 0.384),
                       (0.45, 0.41, 0.35), 0.14, '#FFFDF8'),
    # Terracotta/bronze: higher specular highlights with deep burnt-umber ink
    'Copper':    theme([0.871, 0.729, 0.612], [0.565, 0.396, 0.318], [0.478, 0.325, 0.259],
                       [0.906, 0.780, 0.639], [0.686, 0.588, 0.561], 0.35, 0.26,
                       (0.243, 0.137, 0.090), (0.480, 0.340, 0.280), (0.420, 0.267, 0.200),
                       (0.40, 0.28, 0.22), 0.18, '#FAF6F3'),
    # Industrial galvanized sheet: cool metallic silver with dark slate creases
    'Zinc':      theme([0.808, 0.827, 0.847], [0.478, 0.506, 0.541], [0.404, 0.431, 0.471],
                       [0.820, 0.808, 0.769], [0.690, 0.745, 0.796], 0.32, 0.28,
                       (0.122, 0.157, 0.200), (0.340, 0.380, 0.430), (0.243, 0.282, 0.333),
                       (0.28, 0.31, 0.35), 0.20, '#F4F6F7'),
    # Dark studio presentation: luminous gray facets with inverted bright cobalt strokes
    'Graphite':  theme([0.541, 0.580, 0.620], [0.227, 0.251, 0.282], [0.184, 0.204, 0.231],
                       [0.604, 0.576, 0.518], [0.494, 0.549, 0.612], 0.35, 0.12,
                       (0.784, 0.831, 0.894), (0.460, 0.500, 0.550), (0.306, 0.357, 0.420),
                       (0.02, 0.03, 0.04), 0.45, '#14171C', ink='#C8D4E4'),
}

def hx(c):
    """'#RRGGBB' -> RGB triple in 0..1."""
    c = c.lstrip('#'); return tuple(int(c[i:i+2], 16)/255.0 for i in (0, 2, 4))

def theme_hex(light, shadow, rim, warm, cool, edge, soft, hidden, ground, pen_light, pen_dark, mix, spec, ground_a, face='#FFFFFF', ink='#2c4a74'):
    d = theme(hx(light), hx(shadow), hx(rim), hx(warm), hx(cool), mix, spec, hx(edge), hx(soft), hx(hidden), hx(ground), ground_a, face, ink)
    d['PEN_LIGHT'], d['PEN_DARK'] = np.array(hx(pen_light)), np.array(hx(pen_dark)); return d

THEMES.update({
    # Calibrated clay render matching the reference figures
    'Kaolin':   theme_hex('#FFFFFF', '#BAC4CE', '#9AA7B4', '#F8F6E8', '#D6E4F0',
                          '#2C4470', '#5A6B7C', '#8E9CA8', '#48525D', '#F9DCDC', '#C99696',
                          0.38, 0.12, 0.14),
    # Warm architectural print with charcoal pencil linework
    'Drafting': theme_hex('#FAFAFA', '#C8CCD0', '#A2A8B0', '#F5F2EB', '#E2E7EC',
                          '#1E252D', '#4A535E', '#9AA3AD', '#3D434A', '#FADBD8', '#B85D56',
                          0.25, 0.08, 0.18),
    # Warm gallery lighting with dark sepia creases
    'Amber':    theme_hex('#FFFDF9', '#CCC5B8', '#ABA394', '#FFF2D6', '#EAEBE8',
                          '#5C3D28', '#7A5B45', '#B09B8A', '#4A3C33', '#FDE2D2', '#C66B44',
                          0.42, 0.14, 0.16),
    # High-contrast frosted arctic look with deep navy silhouettes
    'Glacier':  theme_hex('#F8FCFD', '#B5C8D4', '#8FA8B9', '#F7F8F0', '#C7E2F0',
                          '#1C3B5E', '#3B638A', '#83A7C7', '#334657', '#FCE3EA', '#BF6E82',
                          0.42, 0.18, 0.12),
    # High-contrast technical black-and-white drafting
    'Basalt':   theme_hex('#FFFFFF', '#A8ADB5', '#7C838F', '#F9F7EE', '#D0D7DE',
                          '#11161E', '#333C48', '#788290', '#20252D', '#FCE4D6', '#C45A38',
                          0.25, 0.10, 0.20),
})
DEFAULT_THEME = 'Paper'                                     # the theme applied at startup

def apply_colors(vals):
    """set the material globals from a dict like COLOR_DEFAULTS (colours as RGB triples in 0..1)."""
    g = globals()
    g['LIGHT_COL'], g['SHADOW_COL'] = np.array(vals['LIGHT_COL']), np.array(vals['SHADOW_COL'])
    g['SURFACE_RAMP'] = [g['SHADOW_COL'] + t*(g['LIGHT_COL'] - g['SHADOW_COL']) for t in _RAMP_T]
    g['WARM_BOUNCE'], g['COOL_BOUNCE'], g['TINT_MIX'] = np.array(vals['WARM_BOUNCE']), np.array(vals['COOL_BOUNCE']), float(vals['TINT_MIX'])
    g['COOL_DEEP'], g['SPEC_K'] = np.array(vals.get('COOL_DEEP', COOL_DEEP)), float(vals.get('SPEC_K', 0.12))
    g['EDGE_COL'], g['EDGE_SOFT'], g['HIDDEN_COL'] = tuple(vals['EDGE_COL']), tuple(vals['EDGE_SOFT']), tuple(vals['HIDDEN_COL'])
    g['GROUND_COL'], g['GROUND_A'] = tuple(vals['GROUND_COL']), float(vals['GROUND_A'])
    g['PEN_TINT'] = (np.array(vals['PEN_LIGHT']), np.array(vals['PEN_DARK']))
INK, PANEL_FC, PANEL_EC = '#2c4a74', '#f7f9fc', '#b9c7da'                          # text and box colours of buttons and panels
THETA_COLS = ['#1f3a63', '#c0392b', '#2e8b57', '#8e44ad']                            # theta_1 .. theta_4 in the plots


THEMES.update({
    # The photo-studio look: shade floor lifted to the reference figures' 0.80, one cobalt hue shared by
    # ink, shade and ground shadow, a cream key; measured on the fig-5a net: faces 0.78-0.95, cool -0.067 / warm +0.024
    'Harmony':  theme_hex('#FCFCFB', '#D4D5D4', '#B2B8BF', '#F7F4E7', '#D5E0EC',
                          '#2E4D7E', '#7D828B', '#6680A3', '#616B7A', '#FAE5E1', '#D8B1AD',
                          0.36, 0.10, 0.13),
})
apply_colors(THEMES[DEFAULT_THEME])                        # startup material = the default theme

def view_eye(M):
    """the eye point of the projection M in data coordinates (the point that rows 0, 1 and 3 of M send to zero);
    None for an orthographic projection (the eye at infinity: the view direction decides)."""
    C = np.linalg.svd(np.vstack([M[0], M[1], M[3]]))[2][-1]
    return C[:3]/C[3] if abs(C[3]) > 1e-12 else None

def edge_ink(e, pos, w, faces_all, corners, eye=None):
    """navy on the silhouette and the creases facing the camera, soft grey on the others. The side of a face that is seen
    is decided from the eye point when given (the perspective picture), else from the view direction w."""
    def front(f):
        c = [pos[v] for v in corners(f)]
        return np.dot(unit(np.cross(c[1] - c[0], c[3] - c[0])), w if eye is None else eye - c[0]) > 0
    fs = [f for f in faces_all if set(e) <= set(corners(f))]
    return EDGE_COL if len(fs) < 2 or front(fs[0]) != front(fs[1]) else EDGE_SOFT

def shaded_pieces(pos, faces_all, corners, up, w, L, cen, size, tints, nsub, eye=None):
    """the faces split into nsub x nsub pieces (for the painter's depth sorting), shaded like a studio render:
    half-Lambert key light L on the visible side, a fill light at the camera, and the soft highlight of the key light
    treated as an area light at finite distance (which gives the gentle gradient across large faces)."""
    Lpos = cen + 8.0*size*L
    g = np.arange(nsub + 1)/nsub
    U, W = np.meshgrid(g, g, indexing='ij')
    wts = np.stack([(1-U)*(1-W), U*(1-W), U*W, (1-U)*W], axis=-1)                   # bilinear weights of the grid points
    polys, cols = [], []
    for f in faces_all:
        c = np.array([pos[x] for x in corners(f)])
        n = unit(np.cross(c[1]-c[0], c[3]-c[0]))
        if np.dot(n, w if eye is None else eye - c[0]) < 0: n = -n                   # normal of the side we see (from the eye, if given)
        d = 0.5*(1 + np.dot(n, L))
        I = np.clip((0.40*d + 0.05*abs(np.dot(n, w)))/0.45, 0, 1)
        if f in tints:                                                                # tinted faces (penetrations): linear ramp
            light, shadow = tints[f]; base = shadow + (light - shadow)*I
        else:                                                                         # the measured surface ramp, shade -> light
            knots = np.linspace(0, 1, len(SURFACE_RAMP)); ramp = np.array(SURFACE_RAMP[::-1])
            base = np.array([np.interp(I, knots, ramp[:, k]) for k in range(3)])
        warm = WARM_BOUNCE if I < 0.8 else WARM_BOUNCE + (WARM_STRONG - WARM_BOUNCE)*0.35*(I - 0.8)/0.2
        tint = COOL_BOUNCE + (warm - COOL_BOUNCE)*I                                   # cool in shade, warm in light
        base = base*(1 - TINT_MIX) + base.mean()*(tint/tint.mean())*TINT_MIX          # borrow the hue, keep the value
        rim = (1 - abs(np.dot(n, w)))**4                                              # faces seen at grazing angle: deep cool rim
        base = base*(1 - 0.30*rim) + COOL_DEEP*0.30*rim
        G = wts @ c                                                                   # (nsub+1, nsub+1, 3) grid points
        C = (G[:-1, :-1] + G[1:, :-1] + G[1:, 1:] + G[:-1, 1:])/4                     # piece centres
        toL = Lpos - C; toL /= np.linalg.norm(toL, axis=-1, keepdims=True)
        h = toL + w; h /= np.linalg.norm(h, axis=-1, keepdims=True)
        spec = np.clip(h @ n, 0, 1)**14
        sky = 0.03*((C - cen) @ up)/size
        pc = np.clip(base*(1 + sky[..., None]) + SPEC_K*spec[..., None]*(SPEC_COL - base), 0, 1)
        quads = np.stack([G[:-1, :-1], G[1:, :-1], G[1:, 1:], G[:-1, 1:]], axis=2)     # (nsub, nsub, 4, 3)
        polys += list(quads.reshape(-1, 4, 3)); cols += list(pc.reshape(-1, 3))
    return polys, cols

class ShadowPolys(Poly3DCollection):
    """the shadow bands: always drawn first (behind everything)."""
    def do_3d_projection(self, *args, **kwargs):
        super().do_3d_projection(*args, **kwargs); return 2e8

def draw_shadow_frame(ax, pos, faces_all, corners, e1, e2, nrm, size, ncell=80, nlevels=12):
    """soft shadow of the net on a floor perpendicular to the unit vector nrm (which points from the floor towards the net), the
    light falling along -nrm: the net's footprint on that floor, blurred, as nested translucent bands. Used for the floor of the
    viewer's frame (nrm = the screen's up direction), so that the shadow always lies under the net on the screen.
    Returns the 3D corner points of the drawn area (for the crop of saved pictures) or None."""
    try: from contourpy import contour_generator
    except ImportError: return None
    allP = np.array(list(pos.values())); h0 = (allP @ nrm).min() - 0.06*size
    P = np.array([[pos[x] for x in corners(f)] for f in faces_all])            # (F, 4, 3)
    Q = P - ((P @ nrm) - h0)[..., None]*nrm                                     # footprint on the floor
    X = Q @ e1; Y = Q @ e2
    blur = 0.07*size
    x0, x1 = X.min() - 3*blur, X.max() + 3*blur; y0, y1 = Y.min() - 3*blur, Y.max() + 3*blur
    gx, gy = np.meshgrid(np.linspace(x0, x1, ncell), np.linspace(y0, y1, ncell), indexing='ij')
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    mask = np.zeros(len(pts), dtype=bool)
    for k in range(len(faces_all)): mask |= Path(np.column_stack([X[k], Y[k]])).contains_points(pts)
    img = mask.reshape(ncell, ncell).astype(float)
    def kernel(step):
        r = int(math.ceil(2.5*blur/step)); k = np.exp(-0.5*((np.arange(-r, r+1)*step)/blur)**2); return k/k.sum()
    img = np.apply_along_axis(lambda v: np.convolve(v, kernel((y1 - y0)/(ncell - 1)), mode='same'), 1, img)
    img = np.apply_along_axis(lambda v: np.convolve(v, kernel((x1 - x0)/(ncell - 1)), mode='same'), 0, img)
    if img.max() < 0.05: return None
    levels = np.linspace(0.05, 1.0, nlevels)[:-1]
    a = 1 - (1 - GROUND_A)**(1/len(levels))
    gen = contour_generator(x=gx, y=gy, z=img, fill_type="OuterOffset")
    origin = h0*nrm
    polys = []
    for lv in levels:
        for outer, offsets in zip(*gen.filled(lv, 1.0001)):
            for i in range(len(offsets) - 1):
                ring = outer[offsets[i]:offsets[i+1]]
                if len(ring) < 3: continue
                polys.append([origin + x*e1 + y*e2 for x, y in ring])
        # one collection per level: bands of growing opacity stack up towards the core
        if polys:
            ax.add_collection3d(ShadowPolys(polys, facecolors=[GROUND_COL + (a,)], edgecolors='none', antialiased=True)); polys = []
    return [origin + x*e1 + y*e2 for x in (x0, x1) for y in (y0, y1)]


def draw_shadow(ax, pos, faces_all, corners, L, size, ncell=80, z0=None, nlevels=12):
    """returns the extents (x0, x1, y0, y1, z0) of the drawn shadow, or None."""
    """soft shadow of the net cast along the light L onto the horizontal plane z = z0 on the far side of the net from the light:
    the floor below the net when the light comes from above, the ceiling above it when the light comes from below (a flashlight
    held by a viewer under the net). Filled contour bands of a blurred mask (nested bands of growing opacity)."""
    if abs(L[2]) < 0.2: return None
    allP = np.array(list(pos.values()))
    if z0 is None: z0 = allP[:, 2].min() - 0.015*size if L[2] > 0 else allP[:, 2].max() + 0.015*size
    S = np.array([[p - (p[2] - z0)/L[2]*L for p in [pos[x] for x in corners(f)]] for f in faces_all])
    blur = 0.07*size
    x0, x1 = S[..., 0].min() - 3*blur, S[..., 0].max() + 3*blur
    y0, y1 = S[..., 1].min() - 3*blur, S[..., 1].max() + 3*blur
    X, Y = np.meshgrid(np.linspace(x0, x1, ncell), np.linspace(y0, y1, ncell), indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel()])
    mask = np.zeros(len(pts), dtype=bool)
    for sh in S: mask |= Path(sh[:, :2]).contains_points(pts)
    img = mask.reshape(ncell, ncell).astype(float)
    def kernel(step):
        r = int(math.ceil(2.5*blur/step)); k = np.exp(-0.5*((np.arange(-r, r+1)*step)/blur)**2); return k/k.sum()
    img = np.apply_along_axis(lambda v: np.convolve(v, kernel((y1 - y0)/(ncell - 1)), mode='same'), 1, img)
    img = np.apply_along_axis(lambda v: np.convolve(v, kernel((x1 - x0)/(ncell - 1)), mode='same'), 0, img)
    if img.max() < 0.05: return None
    levels = np.linspace(0.05, 1.0, nlevels)[:-1]                 # nested regions {img >= level}, each a little darker
    a = 1 - (1 - GROUND_A)**(1/len(levels))                      # GROUND_A = opacity at the centre of the shadow
    try:                                                          # all nested regions in ONE collection (much faster than a
        from contourpy import contour_generator                   # filled contour set per level; the overlaps stack up alike)
        gen = contour_generator(x=X, y=Y, z=img, fill_type="OuterOffset"); polys = []
        for lv in levels:
            for outer, offsets in zip(*gen.filled(lv, 1.0001)):
                for i in range(len(offsets) - 1):
                    ring = outer[offsets[i]:offsets[i+1]]
                    if len(ring) >= 3: polys.append([(x_, y_, z0) for x_, y_ in ring])
        if polys: ax.add_collection3d(ShadowPolys(polys, facecolors=[GROUND_COL + (a,)], edgecolors='none', antialiased=False))
    except ImportError:
        for lv in levels:
            cs = ax.contourf(X, Y, img, levels=[lv, 1.0001], colors=[GROUND_COL + (a,)], zdir='z', offset=z0, antialiased=False)
            cols = [cs] if hasattr(cs, 'do_3d_projection') else cs.collections      # >= 3.8: the set itself is the 3D collection
            for col in cols:                                                          # (its deprecated .collections must not be touched)
                col.do_3d_projection = (lambda c: (lambda *a_, **k: (type(c).do_3d_projection(c, *a_, **k), 1e9)[1]))(col)
    return (x0, x1, y0, y1, z0)

class FancyButton:
    """rounded button on its own small axes: hover highlight, optional on/off toggle, Button-like on_clicked."""
    COLORS = {'off': ('#f7f9fc', '#b9c7da', '#2c4a74'), 'on': ('#cfe0f5', '#3f6fae', '#173d6e'),
              'hover_off': ('#e6eef8', '#8fa9c9', '#2c4a74'), 'hover_on': ('#bcd3f0', '#2f5f9e', '#173d6e')}
    def __init__(self, fig, rect, label, toggle=False, on=False, fontsize=9, rounding=0.16, italic=False, light=False):
        self.fig, self.toggle, self.on, self.hover, self.callbacks, self.light = fig, toggle, on, False, [], light   # light: no hover, no redraw
        self.ax = fig.add_axes(rect); self.ax.set_axis_off(); self.ax.set_navigate(False)
        w_in, h_in = rect[2]*fig.get_figwidth(), rect[3]*fig.get_figheight()
        rounding = min(rounding, 0.45*h_in/w_in)                   # the corner radius must stay below half the height (wide buttons)
        self.patch = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=%.3f" % rounding, mutation_aspect=w_in/h_in,
                                    transform=self.ax.transAxes, clip_on=False, linewidth=0.9)
        self.ax.add_patch(self.patch)
        self.text = self.ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=fontsize, transform=self.ax.transAxes,
                                 fontstyle='italic' if italic else 'normal')
        self._style()
        self._cids = [fig.canvas.mpl_connect('button_release_event', self._release), fig.canvas.mpl_connect('motion_notify_event', self._motion)]
    def remove(self):
        """remove the button and disconnect its event handlers (a rebuilt button must not leave them behind)"""
        for cid in self._cids: self.fig.canvas.mpl_disconnect(cid)
        self._cids = []; self.callbacks = []
        try: self.ax.remove()
        except Exception: pass
    def _style(self):
        fc, ec, tc = self.COLORS[('hover_' if self.hover else '') + ('on' if self.on else 'off')]
        self.patch.set_facecolor(fc); self.patch.set_edgecolor(ec); self.text.set_color(tc)
        self.text.set_fontweight('semibold' if self.on else 'normal')
    def _motion(self, ev):
        if self.light or not self.ax.get_visible(): return
        h = ev.inaxes is self.ax
        if h != self.hover: self.hover = h; self._style(); self.fig.canvas.draw_idle()
    def _release(self, ev):
        if ev.inaxes is not self.ax or not self.ax.get_visible(): return
        if self.toggle: self.on = not self.on
        self._style()
        for cb in self.callbacks: cb(ev)
        if not self.light: self.fig.canvas.draw_idle()
    def on_clicked(self, cb): self.callbacks.append(cb)
    def set_on(self, on): self.on = on; self._style()

class TextBox(_TextBox):
    """matplotlib's TextBox, except that a click outside a box that is not being typed into draws nothing: matplotlib redraws the
    whole window there (stop_typing), once per box and click - 10 full draws for a click on Cancel of the Params dialog"""
    def stop_typing(self):
        if self.capturekeystrokes: super().stop_typing()

class PanelBox(TextBox):
    """a value box on a panel over the picture: no mouse or keys while it is hidden with its panel or blocked (a dialog over it,
    the view being turned); matplotlib's boxes react wherever they lie, also invisible or under a dialog"""
    def __init__(self, ax, blocked, **kw): super().__init__(ax, '', **kw); self.blocked = blocked
    def ignore(self, ev): return super().ignore(ev) or not self.ax.get_visible() or self.blocked()

def make_slider(ax_sl, lo, hi, t0):
    """the (hidden) slider holding the signed flexion parameter t = cot(theta_1/2): the positive half is the family of the net,
    the negative half its mirror image; the two halves meet at t = 0 (theta_1 = +-180, +0 and -0) where t = 0 is admissible; next to
    an end where D = 0 instead, (-lo_flat, lo_flat) is a margin the tool keeps out of. It is driven by the theta_1 slider and the
    value boxes; all updates of the picture go through its on_changed callbacks."""
    kw = dict(orientation='vertical', valinit=t0, valfmt='%.3f', color='none')
    try: sl = Slider(ax_sl, 'flexion parameter $t$', -hi, hi, initcolor='none', track_color='none',
                     handle_style={'facecolor': 'white', 'edgecolor': '#3f6fae', 'size': 11}, **kw)
    except TypeError: sl = Slider(ax_sl, 'flexion parameter $t$', -hi, hi, **kw)
    sl.label.set_fontsize(8.5); sl.label.set_color('#2c4a74'); sl.valtext.set_fontsize(9); sl.valtext.set_color('#2c4a74')
    sl.label.set_position((0.5, 1.04)); sl.valtext.set_position((0.5, -0.03))
    for art in (getattr(sl, 'track', None), getattr(sl, 'poly', None)):
        if art is not None: art.set_visible(False)
    handle = getattr(sl, '_handle', None)                                     # the white circle: above the track and never clipped
    if handle is not None: handle.set_zorder(5); handle.set_clip_on(False); handle.set_markersize(12); handle.set_markeredgewidth(1.4)
    ax_sl.set_xlim(0, 1); ax_sl.set_ylim(-hi, hi)
    return sl

def set_projection(ax):
    if FOCAL_LENGTH is None: ax.set_proj_type('ortho'); return
    try: ax.set_proj_type('persp', focal_length=FOCAL_LENGTH)
    except TypeError: ax.set_proj_type('persp')

# ============================================================================================
# 8. MAIN: assemble, synchronize, draw with a slider
# ============================================================================================
def admissible_t_range(B):
    """a part [lo, hi] of I(x_1) for the first configuration, the trial parameters (startup, Params, a switch of the flexion) and the
    tolerances of the value boxes: the positive half, kept 0.1 % inside the ends and 3 % of the range away from a flat end, an
    unbounded I(x_1) cut at 4 times its lower end. The slider and all sampling of the flexion use the working range of theta_1."""
    lo, hi = B.admissible()[0]
    if lo == 0.0: lo = 0.03*hi if hi < math.inf else 0.05
    if hi == math.inf: hi = 4*lo
    return lo + 0.001*(hi - lo), hi - 0.001*(hi - lo)

DEFAULTS = dict(BASE_ANGLES_DEG=(60.0, 80.0, 120.0, 60.0), M_FACES=5, N_FACES=5, BASE_KN=(2, 2), ROW_TYPES={1: 'c', 2: 'a', 3: 'b'}, SIGNS=(1, -1, 1, -1), E0=1)   # the built-in example
STARTUP_ERROR = [None]
APPLIED_THEME = [None]                                                         # (name, colours) applied in the Colours dialog: kept when Params restarts the tool
USAGE = ("usage: python %s [--frames [--view ELEV,AZIM]]\n"
         "  without options     the interactive window\n"
         "  --frames            headless: saves frame_00.png .. frame_05.png of the flexion in the current folder and exits\n"
         "  --view ELEV,AZIM    with --frames: the view in degrees, e.g. --view 30,-60 or --view=30,-60") % os.path.basename(__file__)

def command_view():
    """the view (elev, azim) of '--view ELEV,AZIM' or '--view=ELEV,AZIM', or None. -h/--help prints the usage; a malformed value ends the
    program with the usage and exit code 2, before anything is built. Other arguments are ignored."""
    args = sys.argv[1:]
    if '-h' in args or '--help' in args: print(USAGE); raise SystemExit(0)
    view = None
    for k, a in enumerate(args):
        if a != '--view' and not a.startswith('--view='): continue
        val = a[7:] if a.startswith('--view=') else (args[k + 1] if k + 1 < len(args) else '')
        try:
            view = tuple(float(x) for x in val.split(','))
            if len(view) != 2 or not all(map(math.isfinite, view)): raise ValueError
        except ValueError:
            print("%s: --view needs two numbers ELEV,AZIM in degrees, e.g. --view 30,-60 (found %r)\n%s" % (os.path.basename(__file__), val, USAGE), file=sys.stderr)
            raise SystemExit(2)
    return view

def main():
    global SIGNS, E0, BASE_ANGLES_DEG, M_FACES, N_FACES, BASE_KN, ROW_TYPES
    command_view()                                                               # a malformed command line: the usage, before the startup
    try:
        return _main()
    except SystemExit as ex:                                                     # the parameters at the top of the file give no net (only SystemExit: other errors are bugs):
        msg = str(ex).strip() or ex.__class__.__name__                                                   # start with the built-in defaults and show the message in the dialog
        names = ('BASE_ANGLES_DEG', 'M_FACES', 'N_FACES', 'BASE_KN', 'ROW_TYPES', 'SIGNS', 'E0')
        if (not isinstance(ex.code, str) or STARTUP_ERROR[0] is not None or '--frames' in sys.argv           # --frames: the message and exit code 1, no frames of the dialog
                or repr(tuple(globals()[k] for k in names)) == repr(tuple(DEFAULTS[k] for k in names))):    # the defaults themselves failed (compared exactly: 5.0 is not 5)
            raise
        print(msg + "\nStarting with the default net instead; the message is shown in the Params dialog.")
        STARTUP_ERROR[0] = msg
        BASE_ANGLES_DEG, M_FACES, N_FACES, BASE_KN = DEFAULTS['BASE_ANGLES_DEG'], DEFAULTS['M_FACES'], DEFAULTS['N_FACES'], DEFAULTS['BASE_KN']
        ROW_TYPES, SIGNS, E0 = dict(DEFAULTS['ROW_TYPES']), DEFAULTS['SIGNS'], DEFAULTS['E0']
        return _main()

def _main():
    m, n = M_FACES, N_FACES
    try: signs_ok = tuple(SIGNS) in SIGN_PATTERNS and E0 in (1, -1)
    except TypeError: signs_ok = False
    if not signs_ok:
        raise SystemExit("\nSIGNS must be (1, -1, 1, -1) or (1, -1, -1, 1) and E0 must be 1 or -1 (found %s and %s)." % (SIGNS, E0))
    problem = angle_problem(BASE_ANGLES_DEG)                                 # impossible base angles: say so instead of dividing by zero
    if problem: raise SystemExit("\nThe base angles %r give no net: %s\nChange BASE_ANGLES_DEG and start again." % (BASE_ANGLES_DEG, problem))
    try: rows = check_parameters(m, n, BASE_KN, ROW_TYPES)
    except AssertionError as ex: raise SystemExit("\n%s." % ex)                    # a parameter problem (main() starts the default net with the message)
    ROW_TYPES.update(rows)
    sub = assemble_net(BASE_ANGLES_DEG, m, n, BASE_KN, rows)
    ang = vertex_face_angles(sub)
    try: subs = {kn: Block(kn[0], kn[1], q) for kn, q in sub.items()}
    except (ValueError, AssertionError, ZeroDivisionError) as ex:                  # angles that pass the checks but are numerically degenerate (e.g. 1e-300)
        raise SystemExit("\nIn floating point, the flexion of the blocks for the angles %s could not be computed (%s): the angles are very close to "
                         "violating one of the conditions.\nChange BASE_ANGLES_DEG slightly and start again." % (tuple(BASE_ANGLES_DEG), ex))
    B = subs[BASE_KN]
    lo, hi = admissible_t_range(B)
    I_lo, I_hi = B.admissible()[0]                                                 # the admissible set itself (the slider stops a little short of an end where D = 0)
    def theta_of_t(t_): return math.degrees(2*math.atan(1/t_)) if t_ != 0 else math.copysign(180.0, t_)   # t = cot(theta_1/2) <-> theta_1 in degrees, signs alike (t = -0: theta_1 = -180, the flat end seen from the mirrored half)
    def t_of_theta(th_deg):
        if abs(th_deg) == 180.0: return math.copysign(0.0, th_deg)                   # the flat end exactly (1/tan(pi/2) is 6e-17 in floating point)
        return 1/math.tan(math.radians(th_deg)/2) if th_deg != 0 else math.inf
    TH_EPS = 0.25                                                                      # degrees: theta_1 = 0 (t infinite) is excluded
    # the working range of the slider, defined on theta_1: 0.1 % of its span inside an end where the discriminant vanishes,
    # the flat end itself (theta_1 = +-180, t = 0) where t = 0 is admissible, and TH_EPS short of theta_1 = 0 (t infinite)
    th_min_adm = theta_of_t(I_hi) if I_hi != math.inf else 0.0; th_max_adm = theta_of_t(I_lo) if I_lo > 0 else 180.0
    span_th = th_max_adm - th_min_adm
    th_lo_w = max(th_min_adm + 0.001*span_th, TH_EPS) if th_min_adm > 0 else TH_EPS
    FLAT_END = I_lo == 0.0                                                             # t = 0 (the base crease flat) is admissible: the slider reaches it
    th_hi_g = th_max_adm - (0.01*span_th if th_max_adm == 180.0 else 0.001*span_th)   # the end of the regular sampling grid of theta_1 (1 % short of a flat end)
    th_hi_w = 180.0 if FLAT_END else th_hi_g                                           # the upper end of the working range of theta_1
    lo_flat = t_of_theta(th_hi_w)                                                      # the smallest |t| the slider takes: 0 at a flat end, else a little above I_lo
    t_grid = t_of_theta(th_hi_g)                                                       # t at the end of the grid: below it (at a flat end) only a few extra samples
    t_hi_w = t_of_theta(th_lo_w)                                                       # the largest |t| the slider takes (theta_1 = th_lo_w; t = infinity is excluded)
    t0 = lo + 0.15*(hi - lo)
    thetas = None
    for f_ in (0.15, 0.05, 0.25, 0.35, 0.5, 0.65, 0.8, 0.92):                   # the blocks agree for every admissible t; numerically,
        thetas = synchronize(subs, BASE_KN, SIGNS, E0, lo + f_*(hi - lo))           # a parameter close to a pole may still fail
        if thetas is not None: t0 = lo + f_*(hi - lo); break
    if thetas is None:
        raise SystemExit("\nWith the row types %s (bottom row first) the blocks of the %d x %d net for the angles %s could not be made to agree "
                         "on their common faces numerically, for the signs %s and e0 = %+d.\nThe flexion exists for all angles that pass the checks, "
                         "so this points to a numerical problem (for instance angles close to the boundary of the elliptic type).\n"
                         "Change the angles, ROW_TYPES or BASE_KN and start again."
                         % ("".join(rows[nu] for nu in range(1, n - 1)), m, n, BASE_ANGLES_DEG, SIGNS, E0))
    cache = {}; free_lengths = {}; kept = set()                                     # kept: the keys of the sampling grids (theta_regions, room_height, motion_samples), never trimmed
    def cache_key(t): return round(t, 6) if abs(t) >= 1e-6 else t                   # the configurations are cached per 1e-6 of t (there round(t, 6) >= 1e-6 is never 0); near the flat end t = 0 per exact value
    def trim(c_):                                                                    # room for one more: at most CACHE_SIZE configurations besides the kept ones, the oldest go first
        if len(c_) >= CACHE_SIZE:
            old_ = [k_ for k_ in c_ if k_ not in kept]
            for k_ in old_[:max(0, len(old_) - CACHE_SIZE + 1)]: del c_[k_]
    def configuration(t, keep=False):
        key = cache_key(t)
        if keep: kept.add(key)
        if key not in cache:
            trim(cache); th = synchronize(subs, BASE_KN, SIGNS, E0, t)
            if th is None: cache[key] = None
            else:
                try:
                    pos, worst, plan, fr = build_net(ang, th, m, n, BASE_KN, BASE_EDGE, free_lengths if free_lengths else None)
                    if not free_lengths: free_lengths.update(fr)
                    cache[key] = (pos, worst, plan, th)
                except AssertionError as ex: cache[key] = None; state_fail['reason'] = str(ex)   # construction failed at this t: skipped
        return cache[key]
    state_fail = {}
    cache_other = {}                                                                # nets with the other sign -E0 at t > 0
    def configuration_other(t, keep=False):
        """the net at t > 0 with the other sign -E0 (its mirror image is the net at -t with the sign E0)"""
        key = cache_key(t)
        if keep: kept.add(key)
        if key not in cache_other:
            trim(cache_other); th = synchronize(subs, BASE_KN, SIGNS, -E0, t)
            cache_other[key] = None
            if th is not None:
                try:
                    pos, worst, plan, fr = build_net(ang, th, m, n, BASE_KN, BASE_EDGE, free_lengths if free_lengths else None)
                    cache_other[key] = (pos, worst, plan, th)
                except AssertionError: pass
        return cache_other[key]
    def configuration_at(v, keep=False):
        """the net of the flexion with the signs SIGNS and the sign E0 at the signed parameter v = cot(theta_1/2) of the base
        block. For v < 0 it is the mirror image (z -> -z, every dihedral angle negated) of the net at -v with the other sign
        -E0: the flexion formulas give cot(theta_i/2) at (v, e0) equal to minus their value at (-v, -e0). At the flat end the sign of
        zero tells the halves apart: v = +0 is the net at theta_1 = 180, v = -0 the mirror image of the net with -E0 at t = 0
        (theta_1 = -180), the same configuration reached from the mirrored half of the slider."""
        if math.copysign(1.0, v) > 0: return configuration(v, keep)
        c_ = configuration_other(-v, keep)
        if c_ is None: return None
        pos_, worst_, plan_, th_ = c_
        return ({k_: np.array([p_[0], p_[1], -p_[2]]) for k_, p_ in pos_.items()}, worst_, plan_,
                {f_: {i_: -a_ for i_, a_ in d_.items()} for f_, d_ in th_.items()})
    bugs_seen = set()
    def report_bug(where):
        """an unexpected exception in a guarded call (one that expects only the failures of the construction at extreme parameters,
        or none): its traceback on stderr, once per place and kind, so that a bug does not pass for a dead part, a skipped frame or a stale picture"""
        key_ = (where, type(sys.exc_info()[1]).__name__)
        if key_ not in bugs_seen: bugs_seen.add(key_); import traceback; print("unexpected error in %s (shown once):" % where, file=sys.stderr); traceback.print_exc()
    def block_sign(kn):
        """the sign e0 of the block kn in the net shown: the one for which the flexion formulas, with the signs SIGNS and the
        block's own parameter cot(theta_1/2), give its four dihedral angles (E0 for the base block; the others follow from the
        agreement with their neighbours). None where both signs give the same angles or the angles are at a pole."""
        th = state['th'][kn]
        def cot_half(a_): s_ = math.sin(a_/2); return math.cos(a_/2)/s_ if abs(s_) > 1e-12 else None
        target = {i: cot_half(th[i]) for i in range(1, 5)}
        if target[1] is None: return None
        found = []
        for e0c in (1, -1):
            try: c = subs[kn].cots(SIGNS, e0c, target[1])
            except (ValueError, ZeroDivisionError): continue
            if all(target[i] is None or abs(c[i] - target[i]) < 1e-6*(1 + abs(c[i])) for i in range(2, 5)): found.append(e0c)
        return found[0] if len(found) == 1 else None
    def sign_str(kn, tex=True):
        b_ = block_sign(kn)
        return "?" if b_ is None else ("%+d" % b_ if not tex else ("$+1$" if b_ > 0 else "$-1$"))
    def theta_regions(other=False):
        """the parts of the circle of theta_1 (degrees, t > 0 half) computed once per net: 'live' = the slider's range where the
        whole net has a configuration, 'pen' = with penetrating faces, 'tail' = admissible beyond the slider's range (the
        margins it keeps at an end where D = 0 and at theta_1 = 0; empty at a flat end, which the slider reaches), 'zeros' /
        'poles' = where a cotangent of the base block vanishes / is infinite, 'flat_ok' = the net exists at the flat end t = 0."""
        key_, conf_, e0_ = ('theta_regions_other', configuration_other, -E0) if other else ('theta_regions', configuration, E0)
        if state.get(key_) is None:
            B = subs[BASE_KN]; I0, I1 = B.admissible()[0]
            th_a, th_b = th_lo_w, th_hi_g                                              # the regular sampling grid of theta_1
            th_e = th_hi_w                                                             # the end of the working range: th_b, or 180 at a flat end
            th_min = theta_of_t(I1) if I1 != math.inf else 0.0; th_max = theta_of_t(I0) if I0 > 0 else 180.0
            th_s = th_a                                                              # the sampling starts at the working end (already inside a closed end)
            N = 96; h_ = 0.5*(th_b - th_s)/N                                          # half a sample: isolated samples stay visible
            # at a flat end the band (th_b, 180] beyond the grid gets a few samples (at most half a grid step apart), the last
            # one exactly at theta_1 = 180 (t = 0)
            NB = max(4, int(math.ceil((th_e - th_b)/h_))) if th_e > th_b else 0; hb_ = 0.5*(th_e - th_b)/NB if NB else 0.0
            samples = [th_s + (th_b - th_s)*k_/N for k_ in range(N + 1)] + [(th_e if j_ == NB else th_b + (th_e - th_b)*j_/NB) for j_ in range(1, NB + 1)]
            runs, pruns = [], []; run = prun = None; last_ok = False
            for k_ in range(len(samples) + 1):
                if k_ < len(samples):
                    th_ = samples[k_]
                    try: cfg = conf_(float(t_of_theta(th_)), True)
                    except (ArithmeticError, ValueError, AssertionError): cfg = None       # an extreme parameter the construction cannot take: dead
                    except Exception: report_bug("theta_regions"); cfg = None             # anything else is a bug: reported (and dead, as before)
                    ok = cfg is not None; last_ok = ok
                    pn = ok and len(faces_all) <= LIGHT_FACES and self_intersections(cfg[0], faces_all, corners)
                else: ok = pn = False
                run = [k_, k_] if (ok and run is None) else ([run[0], k_] if ok else run)
                if not ok and run is not None: runs.append(tuple(run)); run = None
                prun = [k_, k_] if (pn and prun is None) else ([prun[0], k_] if pn else prun)
                if not pn and prun is not None: pruns.append(tuple(prun)); prun = None
            flat_ok = bool(NB) and last_ok                                            # the net exists at the flat end itself
            # the isolated parameters of the base block where a cotangent has a pole (theta_i = 0: a denominator of the formulas
            # vanishes) or a zero (theta_i = 180): located by sign changes on a fine grid and refined by bisection
            def cot_i(t_, i):
                try: return B.cots(SIGNS, e0_, float(t_))[i]
                except (ValueError, ZeroDivisionError): return float('nan')
            def cots4(t_):                                                             # cot_i(t_, 1..4) from one evaluation of the formulas
                try: c_ = B.cots(SIGNS, e0_, float(t_)); return [c_[i] for i in range(1, 5)]
                except (ValueError, ZeroDivisionError): return [float('nan')]*4
            def bisect(f_, a_, b_):
                fa = f_(a_)
                for _ in range(60):
                    m_ = (a_ + b_)/2; fm = f_(m_)
                    if fa*fm <= 0: b_ = m_
                    else: a_, fa = m_, fm
                return (a_ + b_)/2
            zeros, poles, excl = [], [], []
            def scan(ths_):
                ts_ = np.array([t_of_theta(th__) for th__ in ths_])
                cs_ = np.array([cots4(t_) for t_ in ts_])
                for i in range(4):
                    v = cs_[:, i]
                    for j in np.where(np.sign(v[:-1])*np.sign(v[1:]) < 0)[0]:
                        pole = min(abs(v[j]), abs(v[j+1])) > 10
                        t_x = bisect((lambda t_: 1/cot_i(t_, i + 1)) if pole else (lambda t_: cot_i(t_, i + 1)), float(ts_[j]), float(ts_[j+1]))
                        (poles if pole else zeros).append(theta_of_t(t_x)); excl.append((t_x, theta_of_t(t_x), i + 1, 'pole' if pole else 'zero'))
            scan(np.linspace(th_b, th_a, 2001))                                       # the grid's range, evenly in the angle
            if NB:
                # the band, as finely, without t = 0 itself: a cotangent that vanishes there (another crease flat with the base
                # edge, e.g. theta_3 when e2 = -e3 and z2 = z3) is part of the flat state, not an isolated parameter (its sign at
                # t = 0 is rounding noise)
                scan(np.linspace(th_e, th_b, max(3, int(math.ceil(2000*(th_e - th_b)/(th_b - th_a))) + 1))[1:])
                try: c0_ = B.cots(SIGNS, e0_, 0.0)
                except (ValueError, ZeroDivisionError): c0_ = {}
                for i_, c_ in sorted(c0_.items()):                                     # a cotangent infinite at t = 0 itself (theta_i = 0; case (c), U ~ 1/t): the
                    if math.isinf(c_): poles.append(180.0); excl.append((0.0, 180.0, i_, 'pole'))   # formulas give the limit; a red tick at the end, t = 0 excluded
            excl.sort(key=lambda e_: (round(e_[0], 9), e_[2], e_[3]))                # by t; coincident parameters (within rounding) by the index of the angle
            def ext(ia, ib):                                                           # a run of samples as an interval, half a sample wider on each side
                return (max(th_s, samples[ia] - (h_ if ia <= N else hb_)), min(th_e, samples[ib] + (h_ if ib < N or not NB else hb_)))
            live = [ext(*r_) for r_ in runs]; pen = [ext(*r_) for r_ in pruns]
            t_live = [(t_of_theta(b_), t_of_theta(a_)) for a_, b_ in live]              # the same parts in t (t decreases with theta)
            # the real sets (for the report): the parts that touch the slider's ends run on to the admissible ends of the base
            # block, open at an end where D = 0 and at t = infinity (theta_1 = 0); at a flat end (t = 0, theta_1 = 180) closed where
            # the net exists at t = 0 itself, open where it exists only next to it; interior gaps as sampled
            real_th = []
            open_flat = bool(NB) and bool(runs) and runs[-1][1] == len(samples) - 2     # the last run reaches the sample next to a dead flat end
            for k_, (a_, b_) in enumerate(live):
                lb, rb = '[', ']'
                if k_ == 0 and abs(a_ - th_s) < 1e-9: a_, lb = th_min, '('                # the ends of I(x_1) are not in it (open set)
                if k_ == len(live) - 1 and (abs(b_ - th_e) < 1e-9 or open_flat): b_, rb = th_max, (']' if flat_ok else ')')
                real_th.append((a_, b_, lb, rb))
            real_t = [((0.0 if b_ == 180.0 else t_of_theta(b_)), (math.inf if a_ == 0.0 else t_of_theta(a_)), rb.replace(']', '[').replace(')', '('), lb.replace('[', ']').replace('(', ')'))
                      for a_, b_, lb, rb in real_th]
            state[key_] = dict(live=live, pen=pen, tail=[(th_min, th_s), (th_e, th_max)], zeros=zeros, poles=poles, excl=excl, t_live=t_live,
                                          real_th=real_th, real_t=real_t, flat_ok=flat_ok)
        return state[key_]
    first = configuration(t0)
    if first is None:                                                           # try other admissible values of t before giving up
        for f_ in (0.05, 0.25, 0.35, 0.5, 0.65, 0.8, 0.92):
            first = configuration(lo + f_*(hi - lo))
            if first is not None: t0 = lo + f_*(hi - lo); break
    if first is None:
        spread = side_spread(vertex_face_angles(sub), m, n, BASE_KN)
        raise SystemExit("\nThe net could not be built in space for any value of the flexion parameter (%s).\n%s" % (state_fail.get('reason', 'unknown reason'),
                         "These angles force sides of the central faces that differ by a factor of %.0e, too much for double precision: make the smallest angle larger." % spread
                         if spread > 1e6 else "Change the base angles or the row types."))
    pos0, worst, plan, th0 = first
    def corners(f): i, j = f; return [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
    faces_all = [(a, b) for a in range(m) for b in range(n)]
    edges_all = sorted({frozenset((c[k], c[(k+1) % 4])) for f in faces_all for c in [corners(f)] for k in range(4)}, key=lambda e: sorted(e))
    def edge_length(pos, e): P, Q = sorted(e); return float(np.linalg.norm(pos[P] - pos[Q]))
    def face_angle(pos, v, f):
        c = corners(f); k = c.index(v); a, b = c[(k-1) % 4], c[(k+1) % 4]
        return math.acos(np.clip(np.dot(unit(pos[a]-pos[v]), unit(pos[b]-pos[v])), -1, 1))
    ref_lengths = {e: edge_length(pos0, e) for e in edges_all}
    ref_face_angles = {(v, f): face_angle(pos0, v, f) for f in faces_all for v in corners(f)}
    def Fn(f, tex=True): return (r"$F_{%d%d}$" if tex else "F%d%d") % f
    def Vn(v, tex=True): return (r"$V_{%d%d}$" if tex else "V%d%d") % v
    def admissible_sets_text(tex=True):
        """the lines 't in ...', '-t in ...', 'theta_1 in ...', '-theta_1 in ...' for the explanations and the report file (t > 0 with
        the sign e0; t < 0 from the nets with the other sign, mirrored)"""
        # the sets of the flexion parameter and of theta_1 of the base block where the net has a configuration (the parts the
        # slider reaches, as a union of intervals), minus the isolated parameters where a cotangent of the base block has a
        # pole (a denominator of the formulas vanishes, theta_i = 0) or a zero (theta_i = 180)
        R, Ro = theta_regions(), theta_regions(other=True); out_ = []
        def set_lines(label, ivals, excl, fmt, unit):
            def num(v_): return "0" + unit if v_ == 0 else ("\u221e" if v_ == math.inf else ("180" + unit if v_ == 180.0 and unit else fmt % v_ + unit))
            body = " \u222a ".join("%s%s, %s%s" % (lb, num(a_), num(b_), rb) for a_, b_, lb, rb in ivals) or "\u2205"
            if excl:
                body += " \\ {" + ", ".join("%s%s (%s)" % (fmt % v_, unit, (r"$\theta_%d = %s$" % (i_, "0" if k_ == 'pole' else "180^\\circ")) if tex
                                              else "theta_%d = %s" % (i_, "0" if k_ == 'pole' else "180\u00b0")) for v_, i_, k_ in excl) + "}"
            words, lines_, cur = re.findall(r"(?:[^\s$]*\$[^$]*\$)+[^\s$]*|\S+", body), [], "  " + label + " "   # a $...$ span is never split (an odd '$' shows a line as source)
            for w_ in words:
                if len(re.sub(r"\$[^$]*\$", "x", cur + w_)) > 58: lines_.append(cur.rstrip()); cur = "      " + w_ + " "
                else: cur += w_ + " "
            return "\n".join(lines_ + [cur.rstrip()])
        for R_, sg_t, sg_th in ((R, "$t$" if tex else "t", r"$\theta_1$" if tex else "theta_1"), (Ro, "$-t$" if tex else "-t", r"$-\theta_1$" if tex else "-theta_1")):
            out_.append(set_lines(sg_t + " ∈", R_['real_t'], [(t_, i_, k_) for t_, th_, i_, k_ in R_['excl']], "%.4f", ""))
        for R_, sg_t, sg_th in ((R, "", r"$\theta_1$" if tex else "theta_1"), (Ro, "", r"$-\theta_1$" if tex else "-theta_1")):
            out_.append(set_lines(sg_th + " ∈", R_['real_th'], [(th_, i_, k_) for t_, th_, i_, k_ in sorted(R_['excl'], key=lambda e_: e_[1])], "%.1f", "\u00b0"))
        return "\n".join(out_)
    def check_values(pos, th):
        """the checks recomputed from the vertices of a net: the largest deviations of the edge lengths and the face angles from
        those of the first net, of the flat angles at the inner vertices from the prescribed ones, of the faces from planarity, and
        of the dihedral angles along the edges of the central faces from the formula values of every block; and the convexity of
        all faces."""
        d_len = max(abs(edge_length(pos, e) - ref_lengths[e]) for e in edges_all)
        d_fa = max(abs(face_angle(pos, v, f) - ref_face_angles[(v, f)]) for f in faces_all for v in corners(f))
        d_flat = max(abs(face_angle(pos, v, f) - ang[v][f]) for v in ang for f in ang[v])
        def normal(f): c = [pos[v] for v in corners(f)]; return unit(np.cross(c[1]-c[0], c[3]-c[0]))
        d_plan = max(abs(np.dot(pos[c[2]] - pos[c[0]], normal(f)))/np.linalg.norm(pos[c[2]] - pos[c[0]]) for f in faces_all for c in [corners(f)])
        d_dih = max(angle_gap(dihedral(subs[f], i, pos), th[f][i]) for f in th for i in range(1, 5))
        convex = True
        for f in faces_all:
            c = [pos[v] for v in corners(f)]; N = normal(f)
            if not all(np.dot(np.cross(c[(k+1)%4]-c[k], c[(k+2)%4]-c[(k+1)%4]), N) > 0 for k in range(4)): convex = False
        return dict(len=d_len, fa=d_fa, flat=d_flat, plan=d_plan, dih=d_dih, convex=convex)
    def verify(pos, th, t, tex=True):
        """everything recomputed from the drawn vertices: congruence of all faces, prescribed flat angles, planarity, dihedral angles
        (tex: face names as mathtext for the panels; plain names for the text file)."""
        cv = check_values(pos, th)
        d_len, d_fa, d_flat, d_plan, d_dih, d_conv = cv['len'], cv['fa'], cv['flat'], cv['plan'], cv['dih'], (0.0 if cv['convex'] else 1.0)
        light = len(faces_all) > LIGHT_FACES
        live_pen = len(faces_all) <= LIVE_PEN_FACES                                 # small nets: the intersection test also while dragging
        pairs = [] if (light or (state.get('dragging', False) and not live_pen)) else self_intersections(pos, faces_all, corners)
        state_pen = ("(not tested for nets with more than %d faces)" % LIGHT_FACES if light else ("(checked when the slider is released)" if (state.get('dragging', False) and not live_pen) else "none")) if not pairs else ", ".join(Fn(f, tex) + " & " + Fn(g, tex) for f, g in pairs[:3]) + (" (+%d more)" % (len(pairs) - 3) if len(pairs) > 3 else "")
        rep = ("checks\n\n\n  edge lengths vs. initial:      %.1e (all %d edges)\n  face angles vs. initial:       %.1e rad (all %d faces)\n"
               "  flat angles vs. prescribed:    %.1e rad (%d inner vertices)\n  planarity of faces:            %.1e\n"
               "  dihedral angles vs. formulas:  %.1e rad (%d in %d blocks)\n  convex faces:                  %s\n"
               "  self-intersection:             %s"
               % (d_len, len(edges_all), d_fa, len(faces_all), d_flat, len(ang), d_plan, d_dih, 4*len(th), len(th), "all" if d_conv == 0 else "NO", state_pen))
        state['penetrating'] = set(sum([[f, g] for f, g in pairs], []))
        return rep
    P0 = np.array(list(pos0.values())); mid = (P0.max(0) + P0.min(0))/2; rng = (P0.max(0) - P0.min(0)).max()*0.56

    # ---------------------------------------------------------------- figure and panels
    fig = plt.figure(figsize=(12.5, 9.5), facecolor='white')
    try: fig.canvas.manager.set_window_title("%d \u00d7 %d net" % (m, n))       # the window title (instead of 'Figure 1')
    except Exception: pass
    # matplotlib searches all artists for pickable ones on every mouse press and every wheel notch (recomputing axis ticks on
    # the way); nothing here uses that picking (faces are found by pick_face), so it is switched off: clicks and scrolling are faster
    for pick_id in ('_button_pick_id', '_scroll_pick_id'):
        try: fig.canvas.mpl_disconnect(getattr(fig, pick_id))
        except Exception: pass
    # matplotlib's own keys (rcParams keymap.*) in this window: only full screen ('f'), save ('s'), close (ctrl/cmd+w) and the grids ('g', 'G':
    # only the face panel's inset shows one) are kept. The others break the tool: 'l', 'k', 'L' make the axes under the mouse logarithmic (the
    # theta_1 track disappears), 'p', 'o' (pan/zoom) lock every button, slider and box, 'h', 'r', 'c', 'v', the arrows and backspace (the
    # toolbar's view history) turn the 3D view under a picture drawn for another view, and 'q' would close the tool at one stray key
    def default_keys(ev):
        if ev.key != 'q' and any(ev.key in matplotlib.rcParams[k_] for k_ in ('keymap.fullscreen', 'keymap.save', 'keymap.quit', 'keymap.grid', 'keymap.grid_minor')):
            matplotlib.backend_bases.key_press_handler(ev)                         # (while a value box is typed in, matplotlib empties all keymaps)
    try:
        if fig.canvas.manager.key_press_handler_id is not None:
            fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id); fig.canvas.mpl_connect('key_press_event', default_keys)
    except Exception: pass
    DESIGN_W, DESIGN_H = 12.5, 9.5                                               # the layout below is designed for this window size (inches)
    # Physical layout: the right column and the parameters dialog keep their size in inches whatever the window size; when the window
    # is too small for them, or for the report/explanation panel, the mouse wheel over the respective region scrolls it smoothly.
    groups = {'col': [], 'dlg': []}; offs = {'col': 0.0, 'dlg': 0.0, 'panel': 0.0}; offs_x = {'dlg': 0.0, 'panel': 0.0}
    def reg(group, *artists):
        for a in artists:
            if not hasattr(a, '_design'):
                a._design = tuple(a.get_position()) if isinstance(a, matplotlib.text.Text) else tuple(a.get_position().bounds)
            if a not in groups[group]: groups[group].append(a)
    def mapped(group, d):
        W, H = fig.get_figwidth(), fig.get_figheight(); sx, sy = DESIGN_W/W, DESIGN_H/H
        if group == 'col': x = 1 - (1 - d[0])*sx                                 # anchored to the right edge
        else: x = 0.5 + (d[0] - 0.5)*sx - offs_x['dlg']                          # centred, minus the horizontal scroll
        y = 1 - (1 - d[1])*sy + offs[group]                                      # anchored to the top edge, plus the scroll offset
        return (x, y) if len(d) == 2 else (x, y, d[2]*sx, d[3]*sy)
    def overflow(group):
        offs_save = offs[group]; offs[group] = 0.0
        ys = [mapped(group, a._design)[1] for a in groups[group] if a.get_visible() or group == 'dlg']
        offs[group] = offs_save
        return max(0.0, -min(ys) + 0.03) if ys else 0.0
    def overflow_x(group):
        """how far the group extends beyond the right window edge (figure fractions), with no horizontal scroll applied."""
        if group == 'dlg':
            save = offs_x['dlg']; offs_x['dlg'] = 0.0
            xs = [mapped('dlg', a._design)[0] + (a._design[2]*DESIGN_W/fig.get_figwidth() if len(a._design) == 4 else 0) for a in groups['dlg']]
            offs_x['dlg'] = save
            return max(0.0, max(xs) - 1 + 0.02) if xs else 0.0
        xs = []
        for t_ in (report_text, face_text):
            if t_.get_visible(): xs.append(extent(t_)[2] + offs_x['panel'])
        return max(0.0, max(xs) - 1 + 0.02) if xs else 0.0
    scroll_hints = {}
    def relayout():
        for group in groups:
            for a in groups[group]:
                p = mapped(group, a._design)
                if isinstance(a, matplotlib.text.Text): a.set_position(p[:2])
                else: a.set_position(list(p))
        # a small hint when the column or the open dialog does not fit the window (the wheel scrolls them)
        if not scroll_hints:
            hint_box = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#b9c7da', alpha=0.92)
            scroll_hints['col'] = fig.text(0.9245, 0.006, "\u21c5 wheel scrolls the column", ha='center', va='bottom', fontsize=7.5, color='#5a6b7c', zorder=40, bbox=hint_box)
            scroll_hints['dlg'] = fig.text(0.5, 0.006, "\u21c5 wheel scrolls the dialog; alt/shift + wheel sideways", ha='center', va='bottom', fontsize=7.5, color='#5a6b7c', zorder=40, bbox=hint_box)
        scroll_hints['col'].set_visible(overflow('col') > 0.035 and offs['col'] < overflow('col') - 0.01)
        scroll_hints['dlg'].set_visible(bool(state.get('dlg_open')) and overflow('dlg') > 0.035 and offs['dlg'] < overflow('dlg') - 0.01)
        fig.canvas.draw_idle()
    state_scroll = {'relayout': relayout}
    ax = fig.add_axes([0.13, 0.02, 0.72, 0.93], projection='3d')
    title_text = fig.text(0.9245, 0.985, "%d \u00d7 %d net" % (m, n), ha="center", va="top", fontsize=13.5, color=INK, fontweight="bold"); reg('col', title_text)
    state = {'t': t0, 'pos': pos0, 'th': th0, 'labels': {'vertices': False, 'faces': False, 'edges': False, 'angles': False},
             'elev': 30, 'azim': -60, 'report': True, 'info': False, 'unrolled': 0,
             'hidden': True, 'shadow': SHOW_SHADOW, 'selected': None, 'dfig': None}
    ax.view_init(elev=state['elev'], azim=state['azim'])                           # the startup view (Reset returns to it; ROOM_LAMP_VIEW assumes it)
    def panel(x, y, size, **kw):
        return fig.text(x, y, "", fontsize=size, family='monospace', va='top', color=INK, linespacing=1.25,
                        bbox=dict(boxstyle='round,pad=0.6,rounding_size=0.9', facecolor=PANEL_FC, edgecolor=PANEL_EC, alpha=0.88), **kw)
    report_text = panel(0.012, 0.968, 7.9)
    n_edges, n_faces, n_inner, n_blocks = m*(n + 1) + n*(m + 1), m*n, (m - 1)*(n - 1), (m - 2)*(n - 2)
    I0_, I1_ = subs[BASE_KN].admissible()[0]                                        # the admissible set of the base block, for the explanations
    def adm_num(v_): return "\u221e" if v_ == math.inf else "%.4f" % v_
    adm_txt = ("\u211d" if (I0_ == 0 and I1_ == math.inf) else "(\u2212%s, %s)" % (adm_num(I1_), adm_num(I1_)) if I0_ == 0 else
               "(\u2212%s, \u2212%s) \u222a (%s, %s)" % (adm_num(I1_), adm_num(I0_), adm_num(I0_), adm_num(I1_)))
    INFO = ("HOW THE CHECKS ARE COMPUTED (all from the current vertex\n"
            "coordinates, recomputed at every position of the slider;\n"
            "a value box (under a slider, in this report, or in the Params\n"
            "dialog) can be edited: click it, type, press Enter; Tab moves\n"
            "to the next box)\n\n"
            "edge lengths vs. initial\n"
            r"  $|V_{ij} - V_{kl}|$ for each of the %d edges, minus the same" % n_edges + "\n"
            "  length in the initial net; maximum over edges.\n\n"
            "face angles vs. initial   (rigidity: nothing changes)\n"
            "  at every corner of every face, including the corners at\n"
            "  the boundary of the net, the angle between the two edges\n"
            "  (arccos of the dot product of the unit edge vectors) minus\n"
            "  the value it had in the initial net; maximum over\n"
            "  the %d angles. With the edge lengths: every face stays\n" % (4*n_faces) +
            "  congruent to itself, so the net moves as a mechanism.\n"
            "  This line does not say WHICH angles the faces have.\n\n"
            "flat angles vs. prescribed   (identity: the right values)\n"
            "  only at the %d inner vertices (%d angles, four faces\n" % (n_inner, 4*n_inner) +
            "  meet there): the measured angle minus the value the flat\n"
            r"  angle must have, namely $\alpha_1, \beta_1, \gamma_1, \delta_1$ at the vertex" "\n"
            "  $A_1$ of the base block, carried to all blocks by the relations\n"
            "  between the flat angles of adjacent 3 x 3 blocks.\n"
            "  At the boundary vertices only two faces meet and nothing\n"
            "  is prescribed, so they appear in the previous line only.\n\n"
            "planarity of faces\n"
            "  for each face, the distance of the fourth vertex from\n"
            "  the plane of the other three, divided by the diagonal.\n\n"
            "dihedral angles vs. formulas\n"
            r"  at each edge of each central face $F_{\kappa\nu}$, the oriented" "\n"
            "  dihedral angle read from the drawn vertices (with the\n"
            r"  normals of the two faces and its sign) minus $\theta_i$ from" "\n"
            "  the explicit flexion formulas of the 3 x 3 block.\n\n"
            "convex faces\n"
            "  for every face the four cross products of consecutive\n"
            "  edge vectors point to the same side of the face normal.\n\n"
            "self-intersection\n"
            "  for every pair of faces without a common edge, the\n"
            "  segment in which one face meets the plane of the other\n"
            "  is clipped by that other face; a nonempty result means\n"
            "  the faces penetrate each other (drawn in red tones).\n"
            "  Diagonal neighbours (one common vertex) are tested by\n"
            "  their corners at that vertex: they penetrate when the\n"
            "  corners overlap beyond the vertex itself.\n\n"
            "view, zoom\n"
            "  the direction of view (elevation and azimuth in degrees; the\n"
            "  left mouse button in the picture turns it, with Freeze on it\n"
            "  pans) and the size of\n"
            "  the picture against the startup frame (1 = start and Reset,\n"
            "  2 = twice as large; the right mouse button zooms). Typed\n"
            "  values apply on Enter (a zoom about the current centre).\n\n"
            "slider\n"
            + ("  the slider is the dihedral angle $\\theta_1$ of the base block $F_{%d%d}$\n"
               "  (the still face) along its edge $A_1A_2$, on the full circle;\n"
               "  the boxes under it show $\\theta_1$ and the flexion parameter of\n"
               "  the formulas, $t = \\cot(\\theta_1/2)$ (both can be edited). The\n"
               "  admissible set of this block is $I(x_1)$ = %s;\n"
               "  the parts of it (and of $\\theta_1$) where the net can be built in\n"
               "  space, minus the isolated parameters listed after a backslash,\n"
               "  are\n"
               "{SETS}\n"
               "  For $t < 0$ ($\\theta_1 < 0$) the net is the mirror image of the net\n"
               "  at $-t$ with the other sign $-e_0$ (reflected in the plane of the\n"
               "  base face, every dihedral angle negated). The slider stops\n"
               "  short of the ends by 0.1%% of the span of $\\theta_1$ where\n"
               "  $D(x_1, t) = 0$ and 0.25$^\\circ$ before $\\theta_1 = 0$ ($t$ infinite). Where\n"
               "  $t = 0$ is admissible it reaches the flat end $\\theta_1 = \\pm 180^\\circ$\n"
               "  ($t = \\pm 0$, the edge $A_1A_2$ flat), where its two halves meet\n"
               "  in the same net. The +/- buttons step $\\theta_1$ by one degree.\n") % (BASE_KN[0], BASE_KN[1], adm_txt) +
            "  Colours: blue = the net can be built, rose = faces penetrate,\n"
            "  pale = admissible, in a margin the slider keeps out of,\n"
            "  grey = the net cannot be built (the picture stays),\n"
            "  ticks = a cotangent of the base block vanishes (grey, some\n"
            "  $\\theta_i = 180^\\circ$) or has a pole (red, $\\theta_i = 0$, a denominator\n"
            "  of the formulas).\n\n"
            "picture\n"
            "  visible edges: navy on the silhouette and on creases facing\n"
            "  the camera, soft grey on the other creases; parts behind\n"
            "  faces are faint thin lines (button Hidden). The shadow\n"
            "  is cast by the key light onto a plane below the net\n"
            "  (button Shadow). Light: Eye = a flashlight in your hand, it\n"
            "  turns with you and casts the shadow behind the net (floor from\n"
            "  above, ceiling from below); Top = the sun at the top of the\n"
            "  screen, the shadow lies straight below the net on the screen;\n"
            "  Studio = the shading light follows you while the shadow is cast\n"
            "  by a near-vertical light onto the room's floor; World = the sun\n"
            "  fixed in space above the net, the shadow is its vertical\n"
            "  footprint and stays with the net when you move; Room = a\n"
            "  photo studio: the lamps are fixed in the room and only you\n"
            "  move, the net stands on its base face on an invisible stand\n"
            "  high enough for the whole flexion; from under the floor there\n"
            "  is no shadow to see (the Blender render adds the studio's ink\n"
            "  rule: blue creases on the top side, grey underneath).\n\n"
            "signs\n"
            r"  all 3 x 3 blocks flex with the same signs $(e_1, e_2, e_3, e_4)$," "\n"
            "  (1, -1, 1, -1) or (1, -1, -1, 1), chosen in Params with the\n"
            r"  sign $e_0$ of the base block; the sign $e_0$ of every other block" "\n"
            "  is the one with which it agrees with its neighbours on their\n"
            "  common faces (shown with the data of its central face and in\n"
            "  Dihedrals).\n\n"
            "buttons\n"
            "  Freeze locks the rotation so that the\n"
            "  left mouse button drags the picture instead.\n"
            "  Colours: the picture theme. Paper is the palette measured\n"
            "  from the reference renders; the other themes change the face\n"
            "  tones, the bounce hues, the specular strength, the inks of\n"
            "  the edges and hidden lines, the shadow and the canvas.\n"
            "  Advanced: pick each entry's colour on the wheel (with the\n"
            "  lightness slider) or type its code; - / + for tint mix,\n"
            "  specular and shadow opacity. Apply repaints the net.\n"
            "  save: SVG / PNG of the picture, OBJ (+ report) of the net\n"
            "  shown, Motion = an OBJ sequence spaced evenly in\n"
            "  theta_1 over the slider's range for t >= 0 (from the flat\n"
            "  end where t = 0 is admissible), without the penetrating\n"
            "  ones (index.txt lists theta_1 and t of each frame). Render in\n"
            "  Blender exports the net shown and renders it in the\n"
            "  background with the built-in recipe (Blender is found\n"
            "  automatically, or set BLENDER_PATH; a render_net.py next\n"
            "  to this script replaces the built-in recipe). Seen from below,\n"
            "  a render shows the shadow in the Top light, and in the Eye\n"
            "  light from more than about 20 degrees below (Studio and World\n"
            "  draw it on a glass floor in front of the net, which the\n"
            "  render cannot show).\n"
            "  Params: the net itself (angles, size, base block, row types,\n"
            r"  signs, sign $e_0$); Apply builds the net before it replaces the" "\n"
            "  window (a change of the signs alone keeps it).\n"
            "  Click on a face for its angles and\n"
            "  edges; for a central face also the data of its 3 x 3\n"
            r"  block ($M$, $r_i$, $s_i$, $f_i$, $K$, $K'$, phase shifts, $e_i$, $e_0$)" "\n"
            r"  and a plot of its dihedral angles against $\theta_1$ ($t \geq 0$)." "\n"
            r"  Button Dihedrals: a window with $\theta_1, \dots, \theta_4$ of" "\n"
            + ("  all %d 3 x 3 blocks\n" % n_blocks if n_blocks <= 100 else "  the 3 x 3 blocks within 3 rows and columns of the base block\n") +
            r"  against $\theta_1$ of the base block, $t \geq 0$ (marker = now)." "\n\n"
            "Values up to 1e-11 (some larger nets: 1e-8) are double-precision\n"
            "rounding; a geometric inconsistency would appear as 1e-3 or more.")
    INFO_LINES = INFO.split("\n")
    def with_glyphs(text, glyphs):
        """reserves room at the right end of the first line of a (monospace) panel for the small round (i) / (x) buttons."""
        def vis_len(line):                                                      # rendered length in monospace cells: mathtext source is much longer than its rendering
            def m(mt):
                t_ = re.sub(r"\\[A-Za-z]+", "x", mt.group(0)[1:-1])           # a command renders as roughly one symbol
                return re.sub(r"[{}_^ \\,]", "", t_)
            return len(re.sub(r"\$[^$]*\$", m, line))
        lines = text.split("\n"); width = max(vis_len(l) for l in lines)
        first = lines[0]; pad = max(2, width - vis_len(first) - len(glyphs))
        lines[0] = first + " "*pad + glyphs
        return "\n".join(lines)
    VIEW_W, VIEW_A, VIEW_B = 9, "  view:   elev ", "   azim "                      # the view boxes: VIEW_W columns of the report text, after these labels
    VIEW_LINES = ["", VIEW_A + " "*VIEW_W + VIEW_B + " "*VIEW_W, "", "  zoom:" + " "*(len(VIEW_A) - 7 + VIEW_W) + " \u00d7"]   # (the zoom box under the elev box)
    def report_string():                             # the report, continued by the unrolled part of the explanations
        body = state['report_str']
        if state.get('blender_line') and 'Blender' not in body: body = body.rstrip() + "\n  " + state['blender_line']
        lines = body.split("\n"); k_ = next((i + 1 for i, l in enumerate(lines) if l.startswith("  self-intersection:")), len(lines))
        body = "\n".join(lines[:k_] + VIEW_LINES + [""]*(k_ < len(lines)) + lines[k_:])   # the view lines right after the checks, before the notes: the boxes stay put
        if state['unrolled'] > 0:                                                    # the explanations, with the current sets of t and theta_1
            body = body + "\n\n" + "\n".join(INFO_LINES[:state['unrolled']]).replace("{SETS}", admissible_sets_text(True))
        return with_glyphs(body, " "*8)
    face_text = panel(0.012, 0.62, 7.9, visible=False)
    inset = fig.add_axes([0.05, 0.16, 0.14, 0.20]); inset.set_visible(False)
    set_projection(ax)
    def room_height():
        """height of the invisible stand of the photo studio (light 'room'): the base face F_kn stays at z = 0 and the rest of the
        net flexes about it, so the floor must stay clear of the deepest point of EVERY admissible configuration. Set once per net
        from a sample of the flexion range (the slider's whole range, evenly in theta_1, both halves):
        ROOM_CLEARANCE under that point, at least ROOM_MIN_HEIGHT (in net sizes)."""
        if state.get('room_floor') is None:
            zmin, sz = 0.0, 0.0
            for th_ in np.linspace(th_lo_w, th_hi_w, 41):
                t_ = float(t_of_theta(float(th_)))
                for cfg in (configuration(t_, True), configuration_at(-t_, True)):           # both halves of the slider (-0.0 at a flat end)
                    if cfg is None: continue
                    P = np.array(list(cfg[0].values())); zmin = min(zmin, float(P[:, 2].min())); sz = max(sz, float((P.max(0) - P.min(0)).max()))
            P = np.array(list(state['pos'].values())); zmin = min(zmin, float(P[:, 2].min())); sz = max(sz, float((P.max(0) - P.min(0)).max()))
            state['room_floor'] = max(ROOM_MIN_HEIGHT*sz, -zmin + ROOM_CLEARANCE*sz)
        return state['room_floor']
    def redraw():
        state['elev'], state['azim'] = ax.elev, ax.azim
        lims = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()) if state.get('drawn') else None   # keep the user's zoom
        ax.cla(); pos = state['pos']
        if state.get('canvas_color'): ax.set_facecolor(state['canvas_color'])   # cla() resets the patch: keep a dark canvas dark
        rotating = bool(state.get('rotating')) and state.get('report_str') is not None
        report = state['report_str'] if rotating else verify(state['pos'], state['th'], state['t'])   # also updates state['penetrating']
        if lims is None:
            ax.set_xlim(mid[0]-rng, mid[0]+rng); ax.set_ylim(mid[1]-rng, mid[1]+rng); ax.set_zlim(mid[2]-rng, mid[2]+rng)
        else:
            ax.set_xlim3d(lims[0]); ax.set_ylim3d(lims[1]); ax.set_zlim3d(lims[2])
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=state['elev'], azim=state['azim']); set_projection(ax)
        u, up, w = view_axes(state['elev'], state['azim'], ax)
        mode = state.get('light_frame', LIGHT_FRAME)
        if mode == 'eye':
            # a flashlight held by the viewer, a little above and to the left of the eye: it turns with the view, its shadow falls
            # behind the net, on the floor when you look from above and on the ceiling when you look from below
            L = unit(w + 0.35*up - 0.2*u); Lsh = L
        elif mode == 'top':
            # the sun at the top of the screen (fixed to the screen, above the picture): the light comes from above the viewer's
            # eye, and the shadow falls on a floor of the viewer's frame straight below the net on the screen
            L = unit(up + 0.35*w - 0.2*u); Lsh = None
        elif mode == 'studio':
            # the original setup: the shading light attached to the viewer (high, to the upper left), the shadow cast by a nearly
            # vertical light in space tilted a little towards the viewer, onto the floor of the room
            L = unit(-0.5*u + 0.85*up + 0.3*w); Lsh = unit(np.array([0, 0, 1.0]) - 0.18*u - 0.10*w)
        elif mode == 'room':
            # the photo studio: the lamps are fixed in the room (set up once for the view ROOM_LAMP_VIEW) and only the photographer
            # moves; the net stands on its base face F_kn on an invisible stand, the shadow falls on the studio's floor
            u0, up0, w0 = view_axes(*ROOM_LAMP_VIEW)
            L = unit(-0.5*u0 + 0.85*up0 + 0.3*w0); Lsh = unit(np.array([0, 0, 1.0]) - 0.45*u0 - 0.25*w0)
        else:
            # 'world': the sun at noon, fixed in space: the shadow is the vertical footprint of the net on the floor; the shading
            # light is anchored high above the first view (so that vertical faces still show their modelling)
            if 'L_world' not in state:
                e_ = state['elev'] if -180 <= state['elev'] < 180 else (state['elev'] + 180.0) % 360.0 - 180.0   # the elevation in [-180, 180),
                el0 = e_ if abs(e_) <= 90 else (180 - e_ if e_ > 0 else -180 - e_)                                   # folded into [-90, 90]
                u0, up0, w0 = view_axes(max(el0, 15.0), state['azim'])
                state['L_world'] = unit(np.array([0, 0, 1.0]) + 0.35*w0 - 0.25*u0)
            L, Lsh = state['L_world'], np.array([0, 0, 1.0])
        allP = np.array(list(pos.values())); size = (allP.max(0) - allP.min(0)).max(); cen = (allP.max(0) + allP.min(0))/2
        state['shadow_box'] = None; state['shadow_pts'] = None
        dragging = bool(state.get('dragging')) or rotating                          # theta_1 dragged or the view rotated: coarser sampling
        n_before_shadow = len(ax.collections)
        if state['shadow']:
            lims_ = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
            if Lsh is None:                                                       # 'top': floor of the viewer's frame, seen from 30 degrees above
                nrm = unit(math.cos(math.radians(30))*up + math.sin(math.radians(30))*w)
                state['shadow_pts'] = draw_shadow_frame(ax, pos, faces_all, corners, u, unit(np.cross(nrm, u)), nrm, size,
                                                         ncell=40 if dragging else 80, nlevels=12)
            elif mode == 'room' and (cen[2] + (18.2*FOCAL_LENGTH*0.55*size*w[2] if FOCAL_LENGTH else (1e9 if w[2] > 0 else -1e9))) < -room_height():
                pass                                                              # the eye is under the studio's floor: no shadow to be seen from there
            else:
                n_before = len(ax.collections)
                state['shadow_box'] = draw_shadow(ax, pos, faces_all, corners, Lsh, size, z0=(-room_height() if mode == 'room' else None),   # None when the light is too flat
                                                  ncell=40 if dragging else 80, nlevels=12)                               # lighter while dragging
                if state['shadow_box'] is not None and (w[2] > 0) != (Lsh[2] > 0):
                    # the viewer is on the other side of the shadow plane than the light (under the floor, or above the ceiling): the
                    # shadow lies on a glass plane BETWEEN the viewer and the net, so it is drawn in front of the net, translucent
                    for c in ax.collections[n_before:]:
                        c.do_3d_projection = (lambda orig: (lambda *a, **k: (orig(*a, **k), -2e9)[1]))(c.do_3d_projection)
            ax.set_xlim3d(lims_[0]); ax.set_ylim3d(lims_[1]); ax.set_zlim3d(lims_[2])   # contourf autoscales; undo
        state['shadow_cols'] = ax.collections[n_before_shadow:]                    # kept by the preview while the view is rotated
        sf = screen_faces(ax, pos, faces_all, corners)                 # projected faces with the current projection
        state['screen_faces'] = sf
        tints = {f: PEN_TINT for f in state.get('penetrating', set())}          # only penetrating faces are tinted
        quick = len(faces_all) > LIGHT_FACES                                        # the light style: large nets only
        nsub = max(2, min(NSUB, int(round(NSUB*math.sqrt(25.0/len(faces_all))))))    # fewer pieces per face for large nets
        if dragging: nsub = max(2, nsub//2)                                         # while dragging: the same style, coarser sampling
        eye = view_eye(ax.get_proj())                                               # the perspective eye (None: orthographic): decides the side seen
        polys, cols = shaded_pieces(pos, faces_all, corners, up, w, L, cen, size, tints, nsub, eye=eye)
        # the edges as thin quads in the view plane: used by the light mode and by the preview shown while the view is rotated
        Lm = float(np.mean([edge_length(pos, e) for e in edges_all])); wq = 0.012*Lm; lift = 0.004*size*w; eq, ec = [], []
        for e in edges_all:
            P, Q = sorted(e); d = pos[Q] - pos[P]; nq = np.cross(w, d); ln = np.linalg.norm(nq)
            if ln < 1e-12: continue
            nq = nq/ln*wq
            eq.append([pos[P] - nq + lift, pos[Q] - nq + lift, pos[Q] + nq + lift, pos[P] + nq + lift]); ec.append(EDGE_COL + (1.0,))
        state['preview'] = (list(polys) + eq, list(cols) + ec)
        if quick:                                                                  # light mode: plain depth sorting, no occlusion layers
            ax.add_collection3d(Poly3DCollection(polys + eq, facecolors=list(cols) + ec, edgecolors='none', antialiased=False))
        else:
            polys, cols, layer = layered_pieces(ax.get_proj(), sf, polys, cols, max_split=2 if len(faces_all) <= 36 else 1, straddle=len(faces_all) > 36)
            for lv in np.unique(layer):
                idx = np.nonzero(layer == lv)[0]
                if len(idx): ax.add_collection(LayerPolys([polys[i] for i in idx], layer=lv, facecolors=[cols[i] for i in idx], edgecolors='none', antialiased=False), autolim=False)
        # edges: visible parts in dark blue, parts behind faces as faint thin lines
        segs_all = {0: [], 1: [], 2: [], 3: []}; segs_edge = {0: []}
        for e in edges_all:
            P, Q = sorted(e)
            if quick: continue                                                     # quick / light mode: edges already drawn with the faces
            PQ_faces = {f for f in faces_all if P in corners(f) and Q in corners(f)}
            for lv, sg in edge_segments(ax, pos[P], pos[Q], PQ_faces, sf, nsamp=12 if dragging else 32).items():
                segs_all[lv] += sg
                if lv == 0: segs_edge[0] += [e]*len(sg)
        style = {1: (HIDDEN_COL + (0.32,), 0.5), 2: (HIDDEN_COL + (0.20,), 0.45), 3: (HIDDEN_COL + (0.14,), 0.4)}
        for lv in (3, 2, 1):
            if not state['hidden']: continue
            if segs_all[lv]:
                col, lw = style[lv]
                ax.add_collection3d(FrontLines(segs_all[lv], rank=lv, colors=[col], linewidths=lw))
        if segs_all[0]:                                                              # visible parts: ink by the kind of crease
            inks = [edge_ink(sg_edge, pos, w, faces_all, corners, eye=eye) for sg_edge in segs_edge[0]]
            ax.add_collection3d(FrontLines(segs_all[0], rank=0, colors=[ink + (1.0,) for ink in inks], linewidths=[0.95 if ink == EDGE_COL else 0.75 for ink in inks], capstyle='round', joinstyle='round'))
        LB = dict(facecolor='white', alpha=0.75, edgecolor='none', pad=0.5); Z = 1000       # labels above all collections
        if state['labels']['vertices']:
            for vv, p in pos.items(): ax.text(*p, Vn(vv), fontsize=8.5, color='#0b2a6f', bbox=LB, zorder=Z)
        if state['labels']['faces']:
            for f in faces_all:
                c = np.array([pos[x] for x in corners(f)]).mean(axis=0)
                ax.text(*c, Fn(f), fontsize=9, color='#6a1b9a', ha='center', va='center', bbox=LB, zorder=Z)
        if state['labels']['edges']:
            for e in edges_all:
                P, Q = sorted(e); mid_e = (pos[P] + pos[Q])/2
                ax.text(*mid_e, "%.3f" % edge_length(pos, e), fontsize=6.5, color='#b00020', fontweight='bold', ha='center', va='center', bbox=LB, zorder=Z)
        if state['labels']['angles']:          # all 4 angles of all faces, re-measured from the current vertices
            for f in faces_all:
                c = corners(f)
                for k, vv in enumerate(c):
                    a_, b_ = c[(k-1) % 4], c[(k+1) % 4]
                    p = pos[vv] + 0.22*(unit(pos[a_]-pos[vv]) + unit(pos[b_]-pos[vv]))*min(edge_length(pos, frozenset((vv, a_))), edge_length(pos, frozenset((vv, b_))))
                    ax.text(*p, "%.1f°" % math.degrees(face_angle(pos, vv, f)), fontsize=6.5, fontweight='bold', ha='center', va='center',
                            color='#1b7f2a' if vv in ang else '#c2571a', bbox=LB, zorder=Z)
        ax.set_axis_off(); state['drawn'] = True
        state['report_str'] = report; report_text.set_text(report_string())
        if 'sync_view' in state: state['sync_view']()                              # the view boxes follow the view and the zoom
        report_text.set_visible(state['report'])
        if 'layout_panels' in state: state['layout_panels']()
        fig.canvas.draw_idle()
    # ---- the motion sampled along t (cached configurations), for the plots of the dihedral angles
    def motion_samples(nmin):
        ts_ = sorted(k for k in cache if cache[k] is not None)
        if sum(1 for k in ts_ if not FLAT_END or k >= t_grid - 1e-6) < nmin:        # (the few samples between the grid and a flat end do not count)
            for th_ in np.linspace(th_lo_w, th_hi_g, nmin + 1): configuration(float(t_of_theta(float(th_))), True)   # the slider's range, evenly in theta_1
            ts_ = sorted(k for k in cache if cache[k] is not None)
        return ts_
    def block_curves(kn, ts_):
        """theta_1..theta_4 (deg) of the block kn at the samples ts_; a NaN breaks the line where the oriented angle wraps at +-180."""
        out = []
        for i in range(1, 5):
            v = np.array([math.degrees(cache[k][3][kn][i]) for k in ts_])
            v[1:][np.abs(np.diff(v)) > 180] = np.nan
            out.append(v)
        return out
    def style_axes(a_):
        a_.grid(alpha=0.3); a_.tick_params(labelsize=7, colors=INK)
        for sp in a_.spines.values(): sp.set_color(PANEL_EC)
    # ---- face picking: click on a face to see the data of its block
    state['press'] = None
    def on_press(ev):
        if ev.inaxes is ax:
            state['press'] = (ev.x, ev.y)
            if getattr(fig.canvas, 'supports_blit', False) and fast['kind'] is None and 'draw_idle' not in vars(fig.canvas):
                fig.canvas.draw_idle = view_request                                  # a rotation, zoom or pan will ask for a redraw
    def view_request(*a, **k):
        """the first redraw request of a rotation, zoom or pan (matplotlib's own handler has changed the view): the fast repaint
        starts, with the picture rebuilt for the current view at every step; any other request is an ordinary redraw"""
        if fast['kind'] == 'rotate': fast_request(); return
        if state.get('press') is not None and fast['kind'] is None:
            state['rotating'] = True
            if fast_begin('rotate', drag_arts(), rebuild=redraw):
                fig.canvas.draw_idle = fast_request; return
            state['rotating'] = False
        if vars(fig.canvas).get('draw_idle') is view_request: del fig.canvas.draw_idle
        fig.canvas.draw_idle()
    def on_motion(ev):
        """the occlusion layers and the hidden-line segments are valid only for the view they were computed for: as soon as a drag
        in the 3D axes starts, the picture is replaced by a plain depth-sorted preview (faces and edges in one collection), which
        stays correct under rotation; the full picture is redrawn when the mouse is released."""
        if state.get('rotating') and fast['kind'] is None and 'sync_view' in state: state['sync_view']()   # (the preview: the view boxes follow too)
        if state.get('press') is None or state.get('rotating') or state.get('preview') is None or ev.inaxes is not ax: return
        if abs(ev.x - state['press'][0]) < 3 and abs(ev.y - state['press'][1]) < 3: return
        if fast['kind'] == 'rotate': return                                          # the same style as at rest, repainted fast
        state['rotating'] = True                                                      # (a backend that cannot blit: the preview)
        keep = set(map(id, state.get('shadow_cols', [])))                            # the shadow stays (no blink)
        for c in list(ax.collections):
            if id(c) not in keep: c.remove()
        for t_ in list(ax.texts): t_.remove()
        polys_p, cols_p = state['preview']
        ax.add_collection3d(Poly3DCollection(polys_p, facecolors=cols_p, edgecolors='none', antialiased=False))
        if state.get('shadow_box'): pass
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    def pick_face(ev):
        """the closest face under the mouse (screen-space test with the current projection)."""
        sf = state.get('screen_faces')
        if sf is None: return None
        best = None
        for f, (poly, abc, cset) in sf.items():
            disp = ax.transData.transform(poly)                     # projected polygon in display pixels
            X = np.array([[ev.x, ev.y]])
            if inside_convex(disp, X)[0]:
                p = ax.transData.inverted().transform(X)[0]
                depth = abc[0]*p[0] + abc[1]*p[1] + abc[2]
                if best is None or depth < best[0]: best = (depth, f)
        return None if best is None else best[1]
    def show_face(f):
        state['selected'] = f
        cs = corners(f); pos = state['pos']
        lines = ["Face " + Fn(f), "", "  angles (deg): " + ", ".join("%.2f" % math.degrees(face_angle(pos, vv, f)) for vv in cs),
                 "  edges: " + ", ".join("%.4f" % edge_length(pos, frozenset((cs[k], cs[(k+1) % 4]))) for k in range(4))]
        if f in subs:
            S = subs[f]; e = tuple(SIGNS); th = state['th'][f]
            lines += [r"  central face of a $3 \times 3$ block with " + RELATION_TEXT[ROW_TYPES[f[1]]],
                      r"  signs $(e_1, e_2, e_3, e_4)$ = %s (all blocks),  sign $e_0$ = %s" % (e, sign_str(f)),
                      r"  $M$ = %.6f,   $u = 1 - M$ = %.6f" % (S.M, S.u),
                      r"  $r_1 = r_2$ = %.6f,  $r_3 = r_4$ = %.6f,  $s_1 = s_4$ = %.6f,  $s_2 = s_3$ = %.6f" % (S.v[1]['r'], S.v[3]['r'], S.v[1]['s'], S.v[2]['s']),
                      r"  $f_1, \dots, f_4$ = %s" % ", ".join("%.6f" % S.v[i]['f'] for i in range(1, 5)),
                      r"  $x_1 = x_2$ = %.6f,  $x_3 = x_4$ = %.6f,  $y_1 = y_4$ = %.6f,  $y_2 = y_3$ = %.6f" % (S.x[1], S.x[3], S.y[1], S.y[2]),
                      r"  $z_1, \dots, z_4$ = %s" % ", ".join("%.6f" % S.z[i] for i in range(1, 5)),
                      r"  $K$ = %.6f,  $K'$ = %.6f,  $\Lambda$ = %s" % (S.K, S.Kp, S.lattice_string()),
                      r"  $t_1$ = %s,  $t_2$ = %s" % (S.phase_string(1), S.phase_string(2)),
                      r"  $t_3$ = %s,  $t_4$ = %s" % (S.phase_string(3), S.phase_string(4)),
                      r"  $e_1 t_1 + e_2 t_2 + e_3 t_3 + e_4 t_4$ = " + S.sign_condition_string(e),
                      "  dihedral angles now (deg): " + ", ".join("%.3f" % math.degrees(th[i]) for i in range(1, 5)),
                      r"  (inset: $\theta_1, \dots, \theta_4$ of this block against $\theta_1$ of the base block, $t \geq 0$; marker = now)"]
            ts_ = motion_samples(16); curves = block_curves(f, ts_); xs_ = [theta_of_t(k) for k in ts_]   # against theta_1: bounded, as on the slider
            inset.cla(); inset.set_visible(True)
            for i, cv in enumerate(curves): inset.plot(xs_, cv, lw=1.2, color=THETA_COLS[i], label=r"$\theta_%d$" % (i+1))
            if not state.get('mirror'):                                                  # the curves are for t >= 0
                for i in range(4): inset.plot([theta_of_t(state['t'])], [math.degrees(th[i+1])], 'o', ms=4, color=THETA_COLS[i])
            inset.set_xlabel(r'$\theta_1$ (°)', fontsize=8, color=INK); inset.set_ylabel('deg', fontsize=7, color=INK); style_axes(inset)
            inset.legend(fontsize=6.5, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.25), frameon=False)
        else:
            owners = [kn for kn in subs if f in [(kn[0]+a, kn[1]+b) for a in (-1, 0, 1) for b in (-1, 0, 1)]]
            lines += ["  boundary face; belongs to the 3 x 3 blocks with central faces " + ", ".join(Fn(kn) for kn in owners)]
            inset.set_visible(False)
        face_text.set_text(with_glyphs("\n".join(lines), " "*4)); face_text.set_visible(True)
        if 'layout_panels' in state: state['layout_panels']()
        fig.canvas.draw_idle()
    def hide_face(ev=None):
        face_text.set_visible(False); inset.set_visible(False); state['selected'] = None
        if 'layout_panels' in state: state['layout_panels']()
        fig.canvas.draw_idle()
    def on_release(ev):
        if vars(fig.canvas).get('draw_idle') is view_request: del fig.canvas.draw_idle   # a click: no view change happened
        if state.get('rotating') and fast['kind'] == 'rotate':                     # released anywhere: the view change ends
            state['rotating'] = False; state['press'] = None; fast_end(); redraw(); return
        if ev.inaxes is not ax: return
        if state['press'] is not None and abs(ev.x - state['press'][0]) < 3 and abs(ev.y - state['press'][1]) < 3:
            f = pick_face(ev)
            if f is not None: show_face(f)
            else: hide_face()
        else:
            state['rotating'] = False; redraw()                     # rotation or zoom finished: update shading and hidden lines
        state['press'] = None
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    def on_scroll_ui(ev):
        """mouse wheel: scroll the dialog (when open), the right column, or the report/face panels when they do not fit the window;
        over the picture the wheel redraws after a zoom. With a modifier key the wheel scrolls horizontally."""
        H, W = fig.get_figheight(), fig.get_figwidth(); step = 0.25/H*(1 if ev.button == 'down' else -1)     # a quarter inch per notch
        hstep = 0.25/W*(1 if ev.button == 'down' else -1); horizontal = wheel_horizontal(ev)
        xf_ = ev.x/fig.bbox.width
        if dlg['open'] or cdl['open']:
            if horizontal: offs_x['dlg'] = min(max(0.0, offs_x['dlg'] + hstep), overflow_x('dlg'))
            else: offs['dlg'] = min(max(0.0, offs['dlg'] + step), overflow('dlg'))
            relayout(); return
        col_left = min(mapped('col', a._design)[0] for a in groups['col'])
        if xf_ >= col_left - 0.01:
            offs['col'] = min(max(0.0, offs['col'] + step), overflow('col')); relayout(); return
        over_inset = inset.get_visible() and inset.get_window_extent().contains(ev.x, ev.y)
        for t_ in (report_text, face_text):                                        # the panels on the left (and the plot under the face panel)
            if t_.get_visible():
                bb = measure(t_)
                if (bb.x0 <= ev.x <= bb.x1 + 12 and bb.y0 - 12 <= ev.y <= bb.y1 + 12) or (over_inset and t_ is face_text):
                    max_off = panel_max_off()
                    if horizontal: offs_x['panel'] = min(max(0.0, offs_x['panel'] + hstep), overflow_x('panel'))
                    else: offs['panel'] = min(max(0.0, offs['panel'] + step), max_off)
                    layout_panels()
                    if fast['kind'] == 'panel' or (fast['kind'] is None and fast_begin('panel', panel_arts())): fast_paint()
                    else: fig.canvas.draw_idle()
                    return
        if ev.inaxes is ax and (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()) != state.get('lims_drawn'): redraw()   # (Axes3D of matplotlib 3.10 has no wheel zoom)
    fig.canvas.mpl_connect('scroll_event', on_scroll_ui)
    fig.canvas.mpl_connect('draw_event', lambda ev: state.update(lims_drawn=(ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())))   # the limits of the picture on screen
    def on_resize(ev):
        for g in offs: offs[g] = min(offs[g], overflow(g)) if g != 'panel' else 0.0
        for g in offs_x: offs_x[g] = 0.0
        relayout(); layout_panels()
    fig.canvas.mpl_connect('resize_event', on_resize)
    redraw()
    # ---- the window with the dihedral angles of all blocks along the motion (button Dihedrals)
    def open_dihedrals():
        """window with theta_1..theta_4 of every block along the motion: all plots live on one canvas, at most 3 x 3 of them are
        in view, and the scrollbars / mouse wheel move the canvas continuously (smooth scrolling)."""
        if state['dfig'] is not None and plt.fignum_exists(state['dfig'].number): return
        ts_ = motion_samples(40); xs_ = [theta_of_t(k) for k in ts_]            # plotted against theta_1 (bounded; t is unbounded for some nets)
        nk, nn = m - 2, n - 2                                                    # blocks: kappa = 1..nk (columns), nu = 1..nn (rows)
        vk, vn = min(nk, 3), min(nn, 3)
        dfig = plt.figure(figsize=(3.7*vk + 0.9, 2.75*vn + 1.6), facecolor='white')
        try:
            if dfig.canvas.manager.key_press_handler_id is not None:                   # the same key bindings as the main window
                dfig.canvas.mpl_disconnect(dfig.canvas.manager.key_press_handler_id); dfig.canvas.mpl_connect('key_press_event', default_keys)
        except Exception: pass
        try: dfig.canvas.manager.set_window_title("dihedral angles of the 3 x 3 blocks")
        except Exception: pass
        def geometry():
            """viewport and plot sizes in figure fractions for the CURRENT window size (the plots keep their size in inches, 3.7 x 2.75)."""
            W, Hf = dfig.get_figwidth(), dfig.get_figheight()
            L0, R0 = 0.95/W, (W - 0.45)/W                                        # viewport (room for the y labels and the scrollbars, always reserved)
            B0, T0 = 0.75/Hf, (Hf - 1.18)/Hf
            gx, gy = 0.55/W, 0.62/Hf                                             # gaps between the plots
            tw, th_ = 3.7/W - gx, 2.75/Hf - gy                                   # size of one plot: fixed in inches
            return L0, R0, B0, T0, gx, gy, tw, th_
        L0, R0, B0, T0, gx, gy, tw, th_ = geometry()
        shown_blocks = sorted(subs) if len(subs) <= 100 else sorted(kn for kn in subs if abs(kn[0] - BASE_KN[0]) <= 3 and abs(kn[1] - BASE_KN[1]) <= 3)
        tiles = {}; marks = []; leg_holder = []
        for kn in shown_blocks:
            a_ = dfig.add_axes([0, 0, tw, th_]); a_.set_zorder(0)
            for i, cv in enumerate(block_curves(kn, ts_)): a_.plot(xs_, cv, lw=1.3, color=THETA_COLS[i], label=r"$\theta_%d$" % (i+1))
            a_.set_title(Fn(kn), fontsize=9.5, color=INK)                          # (with the sign e0 now: update_dihedrals)
            vl = a_.axvline(theta_of_t(state['t']), color='#8fa9c9', lw=0.9, ls='--')
            dots = [a_.plot([theta_of_t(state['t'])], [math.degrees(state['th'][kn][i])], 'o', ms=4, color=THETA_COLS[i-1])[0] for i in range(1, 5)]
            style_axes(a_); a_.set_xlabel(r'$\theta_1$ (°) of the base block', fontsize=9, color=INK); a_.set_ylabel('deg', fontsize=8, color=INK); a_.tick_params(labelsize=7)
            tiles[kn] = a_; marks.append((kn, vl, dots))
        # masks above and below the viewport so that partly scrolled plots do not run into the title, the legend and the scrollbars
        masks = []
        def place_masks(L0, R0, B0, T0, gx, gy):
            for p_ in masks: p_.remove()
            masks.clear()
            for rect in ([0, T0 + 0.35*gy, 1, 1 - T0 - 0.35*gy], [0, 0, 1, B0 - 0.45*gy], [0, 0, L0 - 1.25*gx, 1], [R0 + 0.3*gx, 0, 1 - R0 - 0.3*gx, 1]):   # left: clear of the 'deg' label and the minus signs
                p_ = FancyBboxPatch((rect[0], rect[1]), rect[2], rect[3], boxstyle="square,pad=0", transform=dfig.transFigure,
                                    facecolor='white', edgecolor='none', zorder=1)
                dfig.add_artist(p_); masks.append(p_)
        place_masks(L0, R0, B0, T0, gx, gy)
        dfig.text(0.5, 0.985, r"dihedral angles $\theta_1, \dots, \theta_4$ for $t \geq 0$, with respect to the central face $F_{\kappa\nu}$" "\n"
                  r"of the corresponding $3 \times 3$ block, $1 \leq \kappa \leq %d$, $1 \leq \nu \leq %d$;  signs $(e_1, e_2, e_3, e_4)$ = %s" % (nk, nn, tuple(SIGNS)) + ("" if len(subs) <= 100 else "  (large net: only the blocks near the base block)"),
                  ha='center', va='top', fontsize=9.5, color=INK, zorder=3, linespacing=1.25)
        h, l = tiles[shown_blocks[0]].get_legend_handles_labels()
        leg = dfig.legend(h[:4], l[:4], loc='upper center', bbox_to_anchor=(0.5, 1 - 0.54/dfig.get_figheight()), ncol=4, fontsize=9, frameon=False); leg.set_zorder(3)
        leg_holder.append(leg)
        ks = [kn[0] for kn in shown_blocks]; ns = [kn[1] for kn in shown_blocks]
        k_min, k_max, n_min, n_max = min(ks), max(ks), min(ns), max(ns)
        nk, nn = k_max - k_min + 1, n_max - n_min + 1                               # extent of the shown blocks
        view = {'ox': 0.0, 'oy': float(max(0, nn - vn))}                        # continuous offsets: first visible column - 1, rows above the top
        def place():
            L0, R0, B0, T0, gx, gy, tw, th_ = geometry()
            place_masks(L0, R0, B0, T0, gx, gy)
            for kn, a_ in tiles.items():
                c, r = kn[0] - k_min, n_max - kn[1]                              # column index, row index from the top
                x = L0 + (c - view['ox'])*(tw + gx); y = T0 - (r - view['oy'])*(th_ + gy) - th_
                a_.set_position([x, y, tw, th_])
                a_.set_visible(x + tw > L0 - gx and x < R0 + gx and y + th_ > B0 - gy and y < T0 + gy)
            # how many plots fit in the current window, hence how far one can scroll
            fit_k = max(1, int((R0 - L0 + gx + 1e-9)//(tw + gx))); fit_n = max(1, int((T0 - B0 + gy + 1e-9)//(th_ + gy)))
            limits['ox'], limits['oy'] = max(0.0, nk - fit_k), max(0.0, nn - fit_n)
            for key, bar, axis in (('oy', bars['v'], 'y'), ('ox', bars['h'], 'x')):
                mx = limits[key]; view[key] = min(view[key], mx)
                bar.valmax = mx if mx > 0 else 1.0
                (bar.ax.set_ylim if axis == 'y' else bar.ax.set_xlim)(0, bar.valmax)
                bar.ax.set_visible(mx > 0)
                if abs(bar.val - view[key]) > 1e-9:
                    bar.eventson = False; bar.set_val(view[key]); bar.eventson = True
            bars['v'].ax.set_position([R0 + 0.55*gx, B0, 0.011, T0 - B0]); bars['h'].ax.set_position([L0, B0 - 0.72*gy, R0 - L0, 0.011])
            Hf = dfig.get_figheight()
            if leg_holder: leg_holder[0].set_bbox_to_anchor((0.5, 1 - 0.54/Hf))          # the legend stays at a fixed distance (inches) from the top
            dfig.canvas.draw_idle()
        bars = {}
        axv = dfig.add_axes([0.97, 0.1, 0.011, 0.8]); axv.set_zorder(4)
        bars['v'] = Slider(axv, '', 0, 1, valinit=0, orientation='vertical', color='#8fb6e3'); bars['v'].valtext.set_visible(False)
        axh = dfig.add_axes([0.1, 0.01, 0.8, 0.011]); axh.set_zorder(4)
        bars['h'] = Slider(axh, '', 0, 1, valinit=0, color='#8fb6e3'); bars['h'].valtext.set_visible(False)
        limits = {'ox': 0.0, 'oy': 0.0}
        def on_v(val):
            if abs(float(val) - view['oy']) > 1e-9: view['oy'] = float(val); place()
        def on_h(val):
            if abs(float(val) - view['ox']) > 1e-9: view['ox'] = float(val); place()
        bars['v'].on_changed(on_v); bars['h'].on_changed(on_h)
        place()
        def on_scroll(ev):
            st = 0.34 if ev.button == 'up' else -0.34
            if wheel_horizontal(ev) or (limits['oy'] == 0 and limits['ox'] > 0):
                view['ox'] = min(max(0.0, view['ox'] - st), limits['ox'])
            else:
                view['oy'] = min(max(0.0, view['oy'] + st), limits['oy'])
            place()
        dfig.canvas.mpl_connect('scroll_event', on_scroll)
        dfig.canvas.mpl_connect('resize_event', lambda ev: place())              # the plots keep their size, the viewport follows the window
        def closed(ev):
            state['dfig'] = None
            try: b_dih.set_on(False); fig.canvas.draw_idle()
            except Exception: pass
        dfig.canvas.mpl_connect('close_event', closed)
        state['dfig'], state['dmarks'], state['dbars'] = dfig, marks, bars
        update_dihedrals()
        try: dfig.show()
        except Exception: pass
        plt.figure(fig.number)                                                     # the main figure stays the current one
    def update_dihedrals():
        dfig = state['dfig']
        if dfig is None: return
        if not plt.fignum_exists(dfig.number):          # closed by hand on a backend without close events
            state['dfig'] = None; b_dih.set_on(False); return
        th1_ = theta_of_t(state['t'])
        for kn, vl, dots in state['dmarks']:                                                # the plots are for t >= 0 (against theta_1): no marker for t < 0
            vl.set_xdata([th1_, th1_]); vl.set_visible(not state.get('mirror'))
            for i, d in enumerate(dots): d.set_data([th1_], [math.degrees(state['th'][kn][i+1])]); d.set_visible(not state.get('mirror'))
            vl.axes.title.set_text(r"%s   (sign $e_0$ = %s now%s)" % (Fn(kn), sign_str(kn), ", mirrored" if state.get('mirror') else ""))   # it changes along the motion: as in the face panel
        dfig.canvas.draw_idle()
    def close_dihedrals():
        dfig = state['dfig']; state['dfig'] = None
        if dfig is not None and plt.fignum_exists(dfig.number): plt.close(dfig)
    # ---------------------------------------------------------------- controls: the right column
    X0, XW, H = 0.862, 0.125, 0.030
    ax_sl = fig.add_axes([0.926, 0.530, 0.030, 0.385]); ax_sl.set_facecolor('none'); ax_sl.set_visible(False)   # hidden: t is set through theta_1 and the boxes
    sl = make_slider(ax_sl, lo, hi, t0); reg('col', ax_sl)
    sl.set_active(False)                                                            # hidden: no mouse (it lies under the theta_1 box and track); set_val still works
    sl.valmin, sl.valmax = -1e12, 1e12                                                # the hidden model takes any admissible t (t is unbounded for some nets)
    # the value under the track is a text box: click it, type a t and press Enter (only admissible values are taken)
    sl.valtext.set_visible(False)
    ax_tb = fig.add_axes([0.895, 0.497, 0.046, 0.026], zorder=6)                      # t, under its formula
    t_box = TextBox(ax_tb, '', initial='%.3f' % t0, textalignment='center', color='#ffffff', hovercolor='#eef3fa')
    t_box.text_disp.set_fontsize(9); t_box.text_disp.set_color('#2c4a74')
    for sp in ax_tb.spines.values(): sp.set_edgecolor('#b9c7da'); sp.set_linewidth(0.8)
    reg('col', ax_tb)
    def put_box(tb, txt):
        """the text of a value box shown as is: TextBox.set_val would place a text cursor by a full redraw of the window, twice
        per step of the slider (slow, and during a drag it showed the window without the net for a moment: a blink)"""
        tb.text_disp.set_text(txt); tb._shown = txt; tb._edited = False               # _shown: submitted unedited (Enter, Tab, a click elsewhere) it changes nothing
        try: tb.cursor.set_visible(False)
        except Exception: pass
    def box_t(v_): return ("%.1e" % v_) if (FLAT_END and 0 < abs(v_) < 5e-4) else ("%.3f" % v_)      # a tiny t next to the flat end is not shown as 0.000
    def box_th(v_):                                                                # %.1f, unless that would show a value next to the flat end as 180.0
        s_ = "%.1f" % v_
        if abs(v_) != 180.0 and abs(float(s_)) == 180.0:
            s_ = "%.4f" % v_
            if abs(float(s_)) == 180.0: s_ = ("-" if v_ < 0 else "") + "≈180"
        return s_ + "°"
    def sync_t_box():
        """the box shows the slider's value (without firing its own submit)"""
        put_box(t_box, box_t(sl.val))
        try: t_box.cursor.set_visible(False)                                            # set_val shows the text cursor although nothing is edited
        except Exception: pass
    def t_str(v_): return "%.3g" % v_ if (FLAT_END and 0 < abs(v_) < min(t_grid, 1e-4)) else "%.4f" % v_   # a tiny t near the flat end is not shown as 0.0000
    def t_submit(text):
        if text == t_box._shown and not t_box._edited: return                           # the box as shown, not edited: the parameter stays (not rounded to the display)
        try: v = float(text.strip().replace(',', '.').replace('\u2212', '-'))           # (also the minus sign U+2212 of the notes)
        except ValueError:
            note("$t$: '%s' is not a number" % text.strip(), key='tbox'); sync_t_box(); return
        if math.isnan(v): note("$t$: '%s' is not a number" % text.strip(), key='tbox'); sync_t_box(); return
        if not math.isfinite(v):
            note("$t$: must be a finite number ($t$ infinite, $\\theta_1 = 0$, is excluded)", key='tbox'); sync_t_box(); return
        eps = 1e-3*(hi - lo); a_ = abs(v)                                               # the ends may be typed as displayed (rounded)
        if not (I_lo - eps <= a_ <= I_hi + eps):
            note("$t$: must lie in the admissible set $I(x_1)$ = %s" % adm_txt, key='tbox'); sync_t_box(); return
        clamped = a_ < lo_flat; a_ = max(a_, lo_flat)                                   # (only near an end where D = 0: at a flat end lo_flat = 0)
        clamped_hi = a_ > t_hi_w; a_ = min(a_, t_hi_w)                                  # towards theta_1 = 0 (t infinite) or an end where D = 0: as far as the slider goes
        v_ = math.copysign(a_, v)                                                       # '-0' is the flat end of the mirrored half (theta_1 = -180)
        if configuration_at(v_) is None:
            note("$t$: the net cannot be built at %s" % t_str(v_), key='tbox'); sync_t_box(); return
        sl.set_val(v_)                                                                  # on_change: the configuration and the picture follow
        sync_t_box()
        if clamped: note("$t$: the slider stops at %.4f, a little short of the end of the admissible set" % a_, key='tbox')   # after the redraw, so that it stays visible
        if clamped_hi: note("$t$: the slider stops at %.4f, a little short of %s" % (a_, "$\\theta_1 = 0$ ($t$ infinite)" if I_hi == math.inf else "the end of the admissible set"), key='tbox')
    t_box.on_submit(t_submit)
    def mark_t(ok):
        t_box.text_disp.set_color('#2c4a74' if ok else '#b00020')
    def show_t(v_):
        put_box(t_box, box_t(v_))
        try: t_box.cursor.set_visible(False)
        except Exception: pass
    sl.on_changed(lambda val: sync_t_box())
    # ---- the slider: the dihedral angle theta_1 of the base block on the full circle [-180, 180] degrees; t = cot(theta_1/2).
    # t > 0 gives theta_1 in (0, 180); theta_1 < 0 is the mirror image of the net (reflected in the plane of its base face,
    # every dihedral angle negated); where t = 0 is admissible both halves reach the flat end theta_1 = +-180 (t = +0 / -0, the
    # same net). The track shows where a configuration exists (blue), the admissible margins the slider keeps out of (pale:
    # 0.1 % at an end where D = 0, TH_EPS at theta_1 = 0), the penetrating parts (rose), the dead parts (grey), and the
    # isolated parameters where a cotangent of the base block vanishes (theta_i = 180, grey tick) or has a pole (theta_i = 0,
    # a denominator of the formulas, red tick). Nothing moves while the handle is in a dead part.
    th_title2 = fig.text(0.918, 0.543, r'$t = \cot(\theta_1/2)$', ha='center', va='top', fontsize=8.5, color='#2c4a74'); reg('col', th_title2)   # under the theta_1 box
    ax_th = fig.add_axes([0.903, 0.600, 0.030, 0.330]); ax_th.set_facecolor('none')
    kw_th = dict(orientation='vertical', valinit=theta_of_t(t0), valfmt='%.1f', color='none')
    try: th_sl = Slider(ax_th, r'$\theta_1$ (°)', -180.0, 180.0, initcolor='none', track_color='none', handle_style={'facecolor': 'white', 'edgecolor': '#3f6fae', 'size': 11}, **kw_th)
    except TypeError: th_sl = Slider(ax_th, r'$\theta_1$ (°)', -180.0, 180.0, **kw_th)
    th_sl.label.set_fontsize(8.5); th_sl.label.set_color('#2c4a74'); th_sl.label.set_position((0.5, 1.04)); th_sl.valtext.set_visible(False)
    for art in (getattr(th_sl, 'track', None), getattr(th_sl, 'poly', None)):
        if art is not None: art.set_visible(False)
    handle_th = getattr(th_sl, '_handle', None)
    if handle_th is not None: handle_th.set_zorder(5); handle_th.set_clip_on(False); handle_th.set_markersize(12); handle_th.set_markeredgewidth(1.4)
    ax_th.set_xlim(0, 1); ax_th.set_ylim(-180, 180); reg('col', ax_th)
    th_art = []
    ax_thb = fig.add_axes([0.895, 0.564, 0.046, 0.026], zorder=6)                     # theta_1, right under the slider
    th_box = TextBox(ax_thb, '', initial='%.1f°' % theta_of_t(t0), textalignment='center', color='#ffffff', hovercolor='#eef3fa')
    th_box.text_disp.set_fontsize(9); th_box.text_disp.set_color('#2c4a74')
    for sp in ax_thb.spines.values(): sp.set_edgecolor('#b9c7da'); sp.set_linewidth(0.8)
    reg('col', ax_thb)
    t_box._shown, th_box._shown, t_box._edited, th_box._edited = t_box.text, th_box.text, False, False   # the initial texts, as if put by put_box
    for tb_ in (t_box, th_box): tb_.on_text_change(lambda txt, tb_=tb_: setattr(tb_, '_edited', tb_._edited or txt != tb_._shown))   # typed into (Enter itself
                                                                                    # also reports a change, of nothing): a submit counts, even of the same text
    def draw_track(ax_, art_, half_span, bands, ticks, bands_neg, ticks_neg):
        """a rounded grey track over [-half_span, half_span] with coloured bands (a, b, colour, zorder, alpha) and tick marks
        (value, colour) given for the positive half and, separately, for the negative half (as positive values, drawn at
        their negatives), and a white gap at 0 (an excluded point)"""
        for a_ in art_:
            try: a_.remove()
            except Exception: pass
        art_.clear()
        bb_ = ax_.get_position(); asp_ = ((2*half_span)/(fig.get_figheight()*bb_.height))/(1.0/(fig.get_figwidth()*bb_.width))
        cap = FancyBboxPatch((0.28, -half_span), 0.44, 2*half_span, boxstyle="round,pad=0,rounding_size=0.22", mutation_aspect=asp_, facecolor='#e3eaf3', edgecolor='none', zorder=1)
        ax_.add_patch(cap); art_.append(cap)
        for sgn_, bands_, ticks_ in ((1, bands, ticks), (-1, bands_neg, ticks_neg)):
            for a_, b_, colr, z_, alpha in bands_:
                if b_ <= a_ + 1e-12: continue
                lo_, hi_ = sorted((sgn_*a_, sgn_*b_))
                r_ = Rectangle((0.28, lo_), 0.44, hi_ - lo_, facecolor=colr, edgecolor='none', zorder=z_, alpha=alpha)
                ax_.add_patch(r_); r_.set_clip_path(cap); art_.append(r_)
            for v_, colr in ticks_:
                art_.append(ax_.plot([0.28, 0.72], [sgn_*v_, sgn_*v_], color=colr, lw=1.1, zorder=5)[0])
        g_ = 0.0033*half_span
        gap_ = Rectangle((0.24, -g_), 0.52, 2*g_, facecolor='white', edgecolor='none', zorder=5); ax_.add_patch(gap_); art_.append(gap_)
    def draw_tracks():
        pen_c = tuple(PEN_TINT[1])
        def bands(R_): return ([(a_, b_, '#cfdcec', 2, 1.0) for a_, b_ in R_['tail']] + [(a_, b_, '#8fb6e3', 3, 1.0) for a_, b_ in R_['live']]
                               + [(a_, b_, pen_c, 4, 0.75) for a_, b_ in R_['pen']])
        def ticks(R_): return [(z_, '#6b7d99') for z_ in R_['zeros']] + [(p_, '#b00020') for p_ in R_['poles']]
        R, Ro = theta_regions(), theta_regions(other=True)
        draw_track(ax_th, th_art, 180.0, bands(R), ticks(R), bands(Ro), ticks(Ro))
    draw_tracks()
    def mark_theta(ok):
        th_box.text_disp.set_color('#2c4a74' if ok else '#b00020')
    def sync_theta():
        """the theta slider and its box follow t and the mirror flag (without firing their own callbacks)"""
        th_deg = theta_of_t(sl.val)
        th_sl.eventson = False; th_sl.set_val(th_deg); th_sl.eventson = True
        put_box(th_box, box_th(th_deg))
        try: th_box.cursor.set_visible(False)
        except Exception: pass
        mark_theta(True)
    def on_theta(val):
        th_deg = float(val)
        def show_dead():                                                          # the box follows the handle, in red; nothing else moves
            put_box(th_box, box_th(th_deg))
            try: th_box.cursor.set_visible(False)
            except Exception: pass
            mark_theta(False); fig.canvas.draw_idle()
        if abs(th_deg) < TH_EPS: show_dead(); return                               # the excluded point theta_1 = 0
        t_ = t_of_theta(th_deg); a_ = abs(t_); eps_ = 1e-3*(hi - lo)
        a_w = min(max(a_, lo_flat), t_hi_w)                                            # the working range (the margins next to its ends are not entered)
        ok = (lo_flat - eps_ <= a_ <= I_hi + eps_) and configuration_at(math.copysign(float(a_w), t_)) is not None
        if ok: sl.set_val(math.copysign(float(a_w), t_))                              # on_change: the configuration (mirrored below zero)
        else: show_dead()
    th_sl.on_changed(on_theta)
    sl.on_changed(lambda val: sync_theta())
    def th_str(v_):                                                                # %.2f, unless that would show a value next to the flat end as 180.00
        s_ = "%.2f" % v_
        if abs(v_) == 180.0 or abs(float(s_)) != 180.0: return s_
        s_ = "%.12g" % v_
        return s_ if abs(float(s_)) != 180.0 else repr(v_)
    def th_submit(text):
        if text == th_box._shown and not th_box._edited: return                         # the box as shown (also '≈180°'), not edited: theta_1 stays
        try: v = float(text.strip().replace(',', '.').replace('\u2212', '-').replace('\u2248', '').replace('°', '').replace('deg', '').strip())
        except ValueError: v = math.nan
        if math.isnan(v):
            note(r"$\theta_1$: '%s' is not a number" % text.strip(), key='thbox'); sync_theta(); return
        if v == 0 or abs(v) > 180:
            note(r"$\theta_1$: must lie in [-180°, 180°], not 0", key='thbox'); sync_theta(); return
        t_ = abs(t_of_theta(v)) if abs(v) > 1e-300 else math.inf; eps_ = 1e-3*(hi - lo)  # (t of a tiny angle overflows); the ends may be typed as displayed (rounded)
        if not (I_lo - eps_ <= t_ <= I_hi + eps_):                                      # not admissible: rejected as in the t box, before any clamping
            e_ = ["%.2f°" % x_ if 0 < x_ < 180 else "%d°" % x_ for x_ in (th_min_adm, th_max_adm)]
            note(r"$\theta_1$: must lie in the admissible set, $|\theta_1|$ ∈ (%s, %s%s ($t \in I(x_1)$)" % (e_[0], e_[1], "]" if FLAT_END else ")"), key='thbox'); sync_theta(); return
        clamped, clamped_lo = t_ < lo_flat, t_ > t_hi_w                                 # inside the margins next to an end where D = 0 or theta_1 = 0: as far as the slider goes
        if clamped or clamped_lo: v = math.copysign(th_hi_w if clamped else th_lo_w, v)   # (a flat end is reached: lo_flat = 0)
        a_w = min(max(t_, lo_flat), t_hi_w)                                             # checked at the parameter the slider takes (as on_theta)
        if configuration_at(math.copysign(float(a_w), v)) is None:
            note(r"$\theta_1$: the net cannot be built at %s° (a dead part of the slider)" % th_str(v), key='thbox'); sync_theta(); return
        th_sl.set_val(v)
        if clamped: note(r"$\theta_1$: the slider stops at %.1f°, a little short of the end of the admissible set" % abs(v), key='thbox')
        if clamped_lo: note(r"$\theta_1$: the slider stops at %.2f°, a little short of %s" % (abs(v), "$\\theta_1 = 0$" if I_hi == math.inf else "the end of the admissible set"), key='thbox')
    th_box.on_submit(th_submit)
    # beside the track: theta_1..theta_4 of the base block against theta_1 (theta_1 vertical, at the height of the track beside it:
    # the half t > 0, over the working range th_lo_w..th_hi_w; the angle horizontal) in the colours of the dihedral plots, and the
    # parts with penetrating faces marked with the penetration tint
    ax_curve = fig.add_axes([0.942, 0.600, 0.046, 0.330]); ax_curve.set_axis_off(); ax_curve.set_navigate(False); ax_curve.set_facecolor('none')
    ax_curve.set_zorder(0); ax_curve.set_visible(SHOW_SLIDER_CURVE); reg('col', ax_curve)
    def draw_curve():
        ax_curve.cla(); ax_curve.set_axis_off()
        ts_ = motion_samples(40)
        if len(ts_) < 3: return
        curves = block_curves(BASE_KN, ts_); ths_ = [theta_of_t(k) for k in ts_]
        for i, cv in enumerate(curves): ax_curve.plot(cv, ths_, lw=1.0, color=THETA_COLS[i], alpha=0.9)
        if len(faces_all) <= LIGHT_FACES:
            pen = [k for k in ts_ if self_intersections(cache[k][0], faces_all, corners)]
            if pen:                                                              # contiguous runs of penetrating samples -> tinted bands
                runs = []; start = prev = pen[0]
                for k in pen[1:]:
                    if ts_.index(k) != ts_.index(prev) + 1: runs.append((start, prev)); start = k
                    prev = k
                runs.append((start, prev))
                dth_ = (th_hi_w - th_lo_w)/max(1, len(ts_) - 1)                         # (theta_1 decreases as t grows)
                for a_, b_ in runs: ax_curve.axhspan(max(th_lo_w, theta_of_t(b_) - dth_/2), min(th_hi_w, theta_of_t(a_) + dth_/2), color=tuple(PEN_TINT[1]), alpha=0.5, lw=0)
        allv = np.concatenate([cv[~np.isnan(cv)] for cv in curves]); lo_v, hi_v = float(allv.min()), float(allv.max()); pad_v = 0.06*(hi_v - lo_v) + 1
        ax_curve.set_ylim(-180, 180); ax_curve.set_xlim(lo_v - pad_v, hi_v + pad_v)        # the vertical scale of the track
    state_scroll['draw_curve'] = draw_curve
    def take_config(cfg):
        """positions and dihedral angles of the net shown (for t < 0 already the mirror image, see configuration_at)"""
        state['pos'] = cfg[0]; state['th'] = cfg[3]
    def on_change(val):
        v_ = float(val); a_ = abs(v_)
        cfg = configuration_at(v_) if a_ >= lo_flat - 1e-9 else None                   # inside the margin (-lo_flat, lo_flat) at an end where D = 0 nothing is shown
        if cfg is not None:
            state['t'] = a_; state['mirror'] = math.copysign(1.0, v_) < 0; take_config(cfg); redraw()   # (v_ = -0: the flat end of the mirrored half)
            if state['selected'] is not None: show_face(state['selected'])
            update_dihedrals()
        mark_t(cfg is not None); mark_theta(cfg is not None)
        if cfg is None: fig.canvas.draw_idle()
    sl.on_changed(on_change)
    # while the slider is dragged only the self-intersection test of medium-size nets is postponed to the release (the picture style is unchanged)
    # fast repaints (blitting): the window without some artists is drawn once and kept as a picture; each step then repaints
    # only those artists over it. Used while theta_1 is dragged (the net, the panels, the handle and the value boxes) and while
    # the panels on the left are scrolled. The artists stay visible, so an ordinary redraw in between still shows everything.
    # A backend that cannot blit redraws the window as usual.
    fast = {'bg': None, 'arts': [], 'kind': None, 'pending': False, 'capturing': False}
    def fast_begin(kind, arts, rebuild=None):
        """the window without the moving artists, rendered off screen and kept; the screen is not touched (an ordinary draw
        would show that picture for a moment: a blink when a drag or a scroll starts)"""
        if not getattr(fig.canvas, 'supports_blit', False): return False
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            vis = [a_.get_visible() for a_ in arts]
            for a_ in arts: a_.set_visible(False)
            fast['capturing'] = True
            try:
                if isinstance(fig.canvas, FigureCanvasAgg): FigureCanvasAgg.draw(fig.canvas)   # into the buffer only, not shown
                else: fig.canvas.draw()
            finally:
                fast['capturing'] = False
                for a_, v_ in zip(arts, vis): a_.set_visible(v_)
            fast.update(bg=fig.canvas.copy_from_bbox(fig.bbox), arts=list(arts), kind=kind, size=fig.canvas.get_width_height(), rebuild=rebuild)
            fast_paint()                                                            # the whole picture back in the buffer at once
            return True
        except Exception: fast_end(); return False
    def fast_paint():
        if fast['bg'] is None or fast.get('size') != fig.canvas.get_width_height(): fast['pending'] = False; return
        if fast.get('rebuild') is not None:                                          # the picture for the current view (its own
            fast['pending'] = True                                                  # redraw requests are swallowed here)
            try: fast['rebuild']()
            except Exception: report_bug("the redraw while rotating")             # (the repaint goes on, as before)
        fast['pending'] = False
        try:
            fig.canvas.restore_region(fast['bg'])
            for a_ in sorted(fast['arts'], key=lambda a__: a__.get_zorder()):         # in the order of an ordinary draw
                if a_.get_visible(): fig.draw_artist(a_)
            fig.canvas.blit(fig.bbox)
        except Exception: pass
    def fast_request(*a, **k):                                                      # the redraw requests of one step: one repaint
        if fast['pending']: return
        fast['pending'] = True
        tm = fig.canvas.new_timer(interval=0); tm.single_shot = True; tm.add_callback(fast_paint); tm.start(); fast['timer'] = tm
    def fast_end():
        fast.update(bg=None, arts=[], kind=None, pending=False, rebuild=None)
        if 'draw_idle' in vars(fig.canvas): del fig.canvas.draw_idle
    def fast_invalidate(ev):                                                        # any other redraw: the kept picture of the panels is stale
        if fast['kind'] == 'panel' and not fast['capturing']: fast.update(bg=None, arts=[], kind=None)
    fig.canvas.mpl_connect('draw_event', fast_invalidate)
    def drag_arts(): return ([ax, inset, report_text, face_text, b_x_rep.ax, b_info.ax, b_x_face.ax, th_box.ax, t_box.ax]
                             + ([handle_th] if handle_th is not None else []) + [b_.ax for b_ in vbox.values()])
    def panel_arts(): return [report_text, face_text, inset, b_x_rep.ax, b_info.ax, b_x_face.ax] + [b_.ax for b_ in vbox.values()]
    state['fast'] = (fast_begin, fast_paint, fast_end)
    def sl_press(ev):
        if ev.inaxes is ax_th:
            state['dragging'] = True
            if fast_begin('drag', drag_arts()): fig.canvas.draw_idle = fast_request
    def sl_release(ev):
        if state.get('dragging'):
            state['dragging'] = False; fast_end(); redraw()
            if state['selected'] is not None: show_face(state['selected'])
    fig.canvas.mpl_connect('button_press_event', sl_press); fig.canvas.mpl_connect('button_release_event', sl_release)
    def half(y, side): return [X0 if side == 0 else X0 + XW - 0.058, y, 0.058, H]
    step = (hi - lo)/40
    dq = 0.030*DESIGN_H/DESIGN_W                                              # a square in inches -> a circle with full rounding
    ymid_ = 0.600 + 0.330/2                                                   # middle of the track; the two round buttons stacked to its right
    b_plus = FancyButton(fig, [0.903 + 0.030 + 0.014, ymid_ + 0.004, dq, 0.030], '+', fontsize=11, rounding=1.0)
    b_minus = FancyButton(fig, [0.903 + 0.030 + 0.014, ymid_ - 0.034, dq, 0.030], '−', fontsize=11, rounding=1.0)
    reg('col', b_minus.ax, b_plus.ax)
    def bump_theta(d_):
        def f(ev): th_sl.set_val(max(-180.0, min(180.0, th_sl.val + d_)))               # one degree of theta_1
        return f
    b_minus.on_clicked(bump_theta(-1.0)); b_plus.on_clicked(bump_theta(+1.0))
    rows = [0.448, 0.412, 0.376, 0.340, 0.304]
    b_vert = FancyButton(fig, half(rows[0], 0), 'Vertices', toggle=True); b_face = FancyButton(fig, half(rows[0], 1), 'Faces', toggle=True)
    b_edge = FancyButton(fig, half(rows[1], 0), 'Lengths', toggle=True); b_angl = FancyButton(fig, half(rows[1], 1), 'Angles', toggle=True)
    b_hid = FancyButton(fig, half(rows[2], 0), 'Hidden', toggle=True, on=True); b_shad = FancyButton(fig, half(rows[2], 1), 'Shadow', toggle=True, on=SHOW_SHADOW)
    b_rep = FancyButton(fig, half(rows[3], 0), 'Report', toggle=True, on=True); b_reset = FancyButton(fig, half(rows[3], 1), 'Reset')
    b_dih = FancyButton(fig, half(rows[4], 0), 'Dihedrals', toggle=True); b_frz = FancyButton(fig, half(rows[4], 1), 'Freeze', toggle=True)
    reg('col', *[b.ax for b in (b_vert, b_face, b_edge, b_angl, b_hid, b_shad, b_rep, b_reset, b_dih, b_frz)])
    reg('col', fig.text(X0 + XW/2, 0.147, 'save', ha='center', va='bottom', fontsize=8, color='#6b7d99'))
    wts = [0.85, 0.85, 0.85, 1.45]; gap_ = 0.0025; unit_ = (XW - 3*gap_)/sum(wts); xs_ = [X0 + sum(wts[:i])*unit_ + i*gap_ for i in range(4)]
    b_svg = FancyButton(fig, [xs_[0], 0.111, wts[0]*unit_, H], 'SVG', fontsize=8.5); b_png = FancyButton(fig, [xs_[1], 0.111, wts[1]*unit_, H], 'PNG', fontsize=8.5)
    b_obj = FancyButton(fig, [xs_[2], 0.111, wts[2]*unit_, H], 'OBJ', fontsize=8.5); b_seq = FancyButton(fig, [xs_[3], 0.111, wts[3]*unit_, H], 'Motion', fontsize=8.5)
    b_rend = FancyButton(fig, [X0, 0.072, XW, H], 'Render in Blender', fontsize=8.5)
    reg('col', b_svg.ax, b_png.ax, b_obj.ax, b_seq.ax, b_rend.ax)
    def toggle(which, btn):
        def f(ev): state['labels'][which] = btn.on; redraw()
        return f
    b_vert.on_clicked(toggle('vertices', b_vert)); b_face.on_clicked(toggle('faces', b_face))
    b_edge.on_clicked(toggle('edges', b_edge)); b_angl.on_clicked(toggle('angles', b_angl))
    def flip(key, btn):
        def f(ev): state[key] = btn.on; redraw()
        return f
    b_hid.on_clicked(flip('hidden', b_hid)); b_shad.on_clicked(flip('shadow', b_shad))
    def reset(ev):
        ax.view_init(elev=30, azim=-60); state['drawn'] = False; state.pop('L_world', None); state.pop('room_floor', None); state.pop('Lsh_world', None)   # the startup view; theta_regions (net and flexion only) is kept
        if math.copysign(1.0, sl.val) < 0: sl.set_val(-sl.val)                         # on_change: back to the upright family (also from -0, the mirrored flat end)
        else: redraw()
    def switch_flexion(signs=None, e0=None):
        """other signs of all blocks and/or another sign e0 of the base block (Apply in Params, when nothing else changed): the
        same net (same faces, same free lengths), another flexion. The configuration is recomputed at the current t (else at the
        nearest parameter that has one); the frame, the track of the slider, the picture and the face panel follow, and the
        Dihedrals window closes. False, and nothing changes, if this flexion cannot be built in space for this net."""
        global SIGNS, E0
        nonlocal mid, rng
        old = (SIGNS, E0); old_cache, old_other = dict(cache), dict(cache_other)
        if signs is not None: SIGNS = tuple(signs)
        if e0 is not None: E0 = e0
        cache.clear(); cache_other.clear(); state.pop('theta_regions', None); state.pop('theta_regions_other', None); state.pop('room_floor', None)
        cfg = None; sgn_ = -1.0 if state.get('mirror') else 1.0                        # the same half of the slider
        for t_ in [state['t']] + sorted((lo + f_*(hi - lo) for f_ in (0.05, 0.15, 0.3, 0.5, 0.7, 0.9)), key=lambda v_: abs(v_ - state['t'])):
            cfg = configuration_at(sgn_*t_)
            if cfg is not None: state['t'] = t_; break
        if cfg is None:                                                                 # back to the flexion shown
            SIGNS, E0 = old; cache.clear(); cache.update(old_cache); cache_other.clear(); cache_other.update(old_other)
            state.pop('theta_regions', None); state.pop('theta_regions_other', None)
            return False
        P_ = np.array(list(cfg[0].values())); mid = (P_.max(0) + P_.min(0))/2; rng = (P_.max(0) - P_.min(0)).max()*0.56
        state['drawn'] = False                                                          # the frame is fitted to the new flexion; the view angles stay
        close_dihedrals(); b_dih.set_on(False)
        sl.eventson = False; sl.set_val(sgn_*state['t']); sl.eventson = True
        sync_t_box(); sync_theta(); take_config(cfg); draw_tracks(); redraw()
        if state['selected'] is not None: show_face(state['selected'])
        return True
    def freeze(ev):
        """Freeze on: the view angles are locked and the left mouse button drags (pans) the picture; off: the left button rotates."""
        try:
            if b_frz.on: ax.mouse_init(rotate_btn=[], pan_btn=[1, 2], zoom_btn=[3])
            else: ax.mouse_init(rotate_btn=[1], pan_btn=[2], zoom_btn=[3])
        except TypeError:                                                          # older matplotlib: no pan button; only lock the rotation
            ax.mouse_init(rotate_btn=[] if b_frz.on else [1], zoom_btn=[3])
        fig.canvas.draw_idle()
    b_frz.on_clicked(freeze)
    b_reset.on_clicked(reset)
    def dihedrals_open(): return state['dfig'] is not None and plt.fignum_exists(state['dfig'].number)
    def dih(ev):                                        # toggle by the actual state of the window (it may have been closed by hand)
        if dihedrals_open(): close_dihedrals(); b_dih.set_on(False)
        else: open_dihedrals(); b_dih.set_on(True)
    b_dih.on_clicked(dih)
    def sync_dih(ev):                                   # the window may be closed by hand without a close event (macosx backend)
        if state['dfig'] is not None and not plt.fignum_exists(state['dfig'].number):
            state['dfig'] = None; b_dih.set_on(False); fig.canvas.draw_idle()
    fig.canvas.mpl_connect('motion_notify_event', sync_dih)
    # ---- parameters dialog (button Params): a modal panel in the middle of the screen; the picture behind is dimmed
    b_par = FancyButton(fig, half(0.266, 0), 'Params', toggle=True); reg('col', b_par.ax)
    b_col = FancyButton(fig, half(0.266, 1), 'Colours', toggle=True); reg('col', b_col.ax)
    # light: five options. eye = a flashlight in the viewer's hand; top = the sun at the top of the screen (fixed to the screen);
    # studio = the original setup; world = the sun at noon fixed in space; room = a photo studio with fixed lamps and a stand
    reg('col', fig.text(X0 + XW/2, 0.242, 'light', ha='center', va='bottom', fontsize=8, color='#6b7d99'))
    LIGHT_ROWS = [(0.206, [('eye', 'Eye', 0.85), ('top', 'Top', 0.85), ('studio', 'Studio', 1.3)]),     # two rows: room for the labels
                  (0.172, [('world', 'World', 1.0), ('room', 'Room', 1.0)])]
    gap = 0.0025; b_light = {}
    for y_, items in LIGHT_ROWS:
        unit_w = (XW - (len(items) - 1)*gap)/sum(wt for _, _, wt in items); xq = X0
        for key, label, wt in items:
            b_light[key] = FancyButton(fig, [xq, y_, wt*unit_w, H], label, toggle=True, on=(LIGHT_FRAME == key), fontsize=8.5); xq += wt*unit_w + gap
    reg('col', *[b.ax for b in b_light.values()])
    def set_light(frame):
        def f(ev):
            state['light_frame'] = frame; state.pop('L_world', None); state.pop('room_floor', None)   # theta_regions depends only on the net and its flexion: kept
            for key, b in b_light.items(): b.set_on(key == frame)
            redraw()
        return f
    for key, b in b_light.items(): b.on_clicked(set_light(key))
    dlg = {'axes': [], 'dyn': [], 'rowbtns': [], 'open': False}
    veil = fig.add_axes([0, 0, 1, 1], zorder=30); veil.set_axis_off(); veil.set_navigate(False)
    veil.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0", transform=veil.transAxes, facecolor='white', alpha=0.86, edgecolor='none'))
    veil.set_visible(False)
    PX, PY, PW, PH = 0.20, 0.14, 0.60, 0.72                                     # the dialog frame
    frame = fig.add_axes([PX, PY, PW, PH], zorder=31); frame.set_axis_off(); frame.set_navigate(False); reg('dlg', frame)
    frame.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", mutation_aspect=PW*fig.get_figwidth()/(PH*fig.get_figheight()),
                                   transform=frame.transAxes, facecolor='#fbfcfe', edgecolor='#8fa9c9', linewidth=1.2))
    frame.set_visible(False)
    LX = PX + 0.03                                                               # left margin of the contents
    def dtext(x, y, txt, size=9, **kw):
        kw = {**dict(color=INK, va='center', ha='left'), **kw}
        t_ = fig.text(x, y, txt, fontsize=size, zorder=32, visible=False, **kw); dlg['axes'].append(t_); reg('dlg', t_); return t_
    dlg['buttons'] = []                                                          # keep the button objects alive (matplotlib holds callbacks weakly)
    def dbutton(x, y, w, label, h=0.03, **kw):
        b = FancyButton(fig, [x, y - h/2, w, h], label, **kw); b.ax.set_zorder(32); b.ax.set_visible(False)
        dlg['axes'].append(b.ax); dlg['buttons'].append(b); reg('dlg', b.ax); return b
    def set_box(tb, txt):
        """the text of a value box, without the text cursor (TextBox.set_val draws the cursor and may redraw the window)"""
        tb.text_disp.set_text(txt)
        try: tb.cursor.set_visible(False)
        except Exception: pass
    def paint_box(tb):
        """shows a new value at once: only the box is repainted on the screen (blit), not the window; a backend that cannot
        blit gets an ordinary redraw"""
        c_ = fig.canvas
        try:
            if not c_.supports_blit: raise RuntimeError
            a_ = tb.ax; a_.draw_artist(a_.patch)
            for sp_ in a_.spines.values(): a_.draw_artist(sp_)
            a_.draw_artist(tb.text_disp); c_.blit(a_.bbox)
        except Exception: c_.draw_idle()
    def box_int(key, default):
        try: return int(float(dlg[key].text))
        except (ValueError, KeyError, OverflowError): return default             # (OverflowError: 'inf')
    def spinbox(x, y, w, val, key, step, lo_, hi_, fmt, after=None):
        """a text box with up/down arrows. An arrow only writes the new value into the box and repaints the box; 'after'
        (optional) keeps dependent boxes in range. hi_ may be a function (a bound that depends on another box). Everything
        is checked when Apply is pressed."""
        axb = fig.add_axes([x, y - 0.014, w, 0.028], zorder=32); axb.set_visible(False)
        tb = TextBox(axb, '', initial=fmt % val, textalignment='center', color='#ffffff', hovercolor='#ffffff')   # no hover repaint
        tb.text_disp.set_fontsize(9.5); dlg['axes'].append(axb); dlg[key] = tb; reg('dlg', axb); tb.set_active(False)
        dlg.setdefault('textboxes', []).append(tb); dlg.setdefault('exact', {})[key] = val   # the value behind the text (the box shows it rounded by fmt)
        def bump(d):
            def f(ev):
                txt = tb.text.strip()
                try: v = (dlg['exact'][key] if txt == fmt % dlg['exact'][key] else float(txt.replace(',', '.'))) + d*step   # from the full value while the box shows it
                except ValueError: return
                if math.isnan(v): return
                v = min(max(v, lo_), hi_() if callable(hi_) else hi_); dlg['exact'][key] = v
                set_box(tb, fmt % v); paint_box(tb)
                if after: after()
            return f
        up = dbutton(x + w + 0.004, y + 0.0075, 0.02, '▲', h=0.014, fontsize=6, rounding=0.3, light=True)
        dn = dbutton(x + w + 0.004, y - 0.0075, 0.02, '▼', h=0.014, fontsize=6, rounding=0.3, light=True)
        up.on_clicked(bump(+1)); dn.on_clicked(bump(-1))
        if after: tb.on_submit(lambda txt: after())
        return tb
    def keep_in_range(key, size_key):
        """the base block stays inside the net when the net shrinks: kappa_0 <= m - 2, nu_0 <= n - 2"""
        hi_ = max(1, box_int(size_key, 3) - 2)
        if box_int(key, 1) > hi_: set_box(dlg[key], '%d' % hi_); paint_box(dlg[key])
    RX = PX + PW - 0.03                                                          # right margin of the contents: nothing goes beyond it
    SOFT = '#4a5b78'
    def heading(y, title, sub=None):
        """a section: a bold title of one or two words, and under it, if needed, a short line in the soft ink"""
        dtext(LX, y, title, 9.5, fontweight='bold')
        if sub: dtext(LX, y - 0.027, sub, 8, color=SOFT)
    def two_boxes(y, names, keys, vals, his, afters):
        """two labelled value boxes at the right margin, on the line of the section's title"""
        for k_, (nm, key, val, hi_, after) in enumerate(zip(names, keys, vals, his, afters)):
            xb = RX - 0.084 - (1 - k_)*0.119                                     # box 0.06 + arrows 0.024, a gap of 0.035
            dtext(xb - 0.006, y, nm, 9.5, ha='right')
            spinbox(xb, y, 0.06, val, key, 1, 1 if key in ('kap', 'nu') else 3, hi_, '%d', after)
    dtext(PX + PW/2, PY + PH - 0.04, 'Parameters of the net', 12, ha='center', fontweight='bold')
    # --- flat angles
    y = PY + PH - 0.09
    heading(y, 'Flat angles', r'$\alpha_1$, $\beta_1$, $\gamma_1$, $\delta_1$ at the vertex $A_1$ of the base block, in degrees')
    for j, a in enumerate(BASE_ANGLES_DEG): spinbox(LX + j*0.135, y - 0.068, 0.095, a, 'ang%d' % j, 1.0, 1.0, 179.0, '%g')
    # --- net size and base block
    y -= 0.125
    heading(y, 'Net size', r'$m \times n$ faces $F_{ij}$, $0 \leq i < m$, $0 \leq j < n$; $3 \leq m, n \leq %d$, $m n \leq %d$' % (MAX_MN, MAX_FACES))
    two_boxes(y, ('$m$', '$n$'), ('M', 'N'), (m, n), (lambda: min(MAX_MN, MAX_FACES//max(3, box_int('N', 3))), lambda: min(MAX_MN, MAX_FACES//max(3, box_int('M', 3)))),
              (lambda: keep_in_range('kap', 'M'), lambda: (keep_in_range('nu', 'N'), build_row_buttons())))   # the row strip changes: a redraw
    y -= 0.075
    heading(y, 'Base block', r'central face $F_{\kappa_0\nu_0}$, $1 \leq \kappa_0 \leq m-2$, $1 \leq \nu_0 \leq n-2$')
    two_boxes(y, (r'$\kappa_0$', r'$\nu_0$'), ('kap', 'nu'), BASE_KN, (lambda: max(1, box_int('M', 3) - 2), lambda: max(1, box_int('N', 3) - 2)), (None, None))
    # --- the assembly: the type rho_nu of each row of blocks
    y -= 0.075
    heading(y, 'Assembly', r'$\rho_\nu \in \{$a, b, c, d$\}$ for each row of blocks $\nu = 1, \ldots, n-2$, bottom to top; click to change')
    ROWS_Y = y - 0.078
    dlg['row_off'] = 0                                                          # first row shown in the strip of row-type buttons
    dlg['rowtypes'] = dict(ROW_TYPES)                                          # the chosen type of every row (also of the rows not on screen)
    ROWS_PER_LINE, ROW_LINES = 8, 2
    def build_row_buttons():
        for b in dlg['rowbtns']: b.remove(); groups['dlg'].remove(b.ax)
        for t_ in dlg['dyn']:
            if hasattr(t_, 'remove') and hasattr(t_, 'ax'): groups['dlg'].remove(t_.ax); t_.remove()       # a scroll arrow (FancyButton)
            else: t_.remove(); groups['dlg'].remove(t_)
        dlg['rowbtns'], dlg['dyn'] = [], []
        try: nrows = max(1, min(int(dlg['N'].text) - 2, MAX_MN - 2))
        except ValueError: nrows = n - 2
        base_type = dlg['rowtypes'].get(BASE_KN[1], 'a')
        for r in range(1, nrows + 1): dlg['rowtypes'].setdefault(r, base_type)
        cap = ROWS_PER_LINE*ROW_LINES
        dlg['row_off'] = max(0, min(dlg['row_off'], nrows - cap))
        shown = range(dlg['row_off'] + 1, min(nrows, dlg['row_off'] + cap) + 1)
        for k, r in enumerate(shown):
            col, line = k % ROWS_PER_LINE, k // ROWS_PER_LINE
            xx, yy = LX + col*0.0595, ROWS_Y - line*0.052
            t_ = fig.text(xx + 0.024, yy + 0.025, r'$\rho_{%d}$' % r, fontsize=8, color=SOFT, ha='center', va='center', zorder=32, visible=dlg['open'])
            b = FancyButton(fig, [xx, yy - 0.014, 0.048, 0.028], dlg['rowtypes'][r], fontsize=9.5)
            b.ax.set_zorder(32); b.ax.set_visible(dlg['open'])
            def cycle(bb, rr):
                def f(ev):
                    new = 'abcd'[('abcd'.index(bb.text.get_text()) + 1) % 4]; bb.text.set_text(new); dlg['rowtypes'][rr] = new
                return f
            b.on_clicked(cycle(b, r)); dlg['rowbtns'].append(b); dlg['dyn'].append(t_); reg('dlg', b.ax, t_)
        if nrows > cap:                                                          # more rows than fit: arrows scroll the strip
            xr = LX + ROWS_PER_LINE*0.0595 + 0.01
            up = FancyButton(fig, [xr, ROWS_Y - 0.014, 0.022, 0.028], '▲', fontsize=7, rounding=0.3)
            dn = FancyButton(fig, [xr, ROWS_Y - 0.052 - 0.014, 0.022, 0.028], '▼', fontsize=7, rounding=0.3)
            for bb in (up, dn): bb.ax.set_zorder(32); bb.ax.set_visible(dlg['open']); dlg['dyn'].append(bb); reg('dlg', bb.ax)
            def shift(d):
                def f(ev): dlg['row_off'] = max(0, min(dlg['row_off'] + d*ROWS_PER_LINE, nrows - cap)); build_row_buttons()
                return f
            up.on_clicked(shift(-1)); dn.on_clicked(shift(+1))
        relayout()
    # --- the signs of every block and the sign e0 of the base block: the two choices of each, at the right margin
    def pat_text(p_): return "(%s)" % ", ".join(("%d" % v_).replace("-", "\u2212") for v_ in p_)
    y = ROWS_Y - 0.052 - 0.056
    heading(y, 'Signs', r'$(e_1, e_2, e_3, e_4)$ of every $3 \times 3$ block')
    sgn_btns = [dbutton(RX - 0.23 + k_*0.12, y, 0.11, pat_text(p_), toggle=True, on=(tuple(p_) == tuple(SIGNS)), fontsize=8.5)
                for k_, p_ in enumerate(SIGN_PATTERNS)]
    y -= 0.075
    heading(y, r'Sign $e_0$', 'of the base block; every other block takes the sign with which it agrees with its neighbours')
    E0_VALUES = (1, -1)
    e0_btns = [dbutton(RX - 0.11 + k_*0.06, y, 0.05, lab_, toggle=True, on=(E0 == v_), fontsize=9)
               for k_, (v_, lab_) in enumerate(zip(E0_VALUES, ('+1', '\u22121')))]
    dlg['signs'], dlg['e0'] = tuple(SIGNS), E0
    def pick(values, btns, key):
        def make(k_):
            def f(ev):
                dlg[key] = values[k_]
                for j_, b_ in enumerate(btns): b_.set_on(j_ == k_)                  # a click on the chosen one keeps it chosen
            return f
        for k_, b_ in enumerate(btns): b_.on_clicked(make(k_))
    pick(SIGN_PATTERNS, sgn_btns, 'signs'); pick(E0_VALUES, e0_btns, 'e0')
    # --- apply / cancel
    b_apply = dbutton(PX + PW/2 - 0.11, PY + 0.024, 0.10, 'Apply'); b_cancel = dbutton(PX + PW/2 + 0.01, PY + 0.024, 0.10, 'Cancel')
    # --- a message box over the dialog, when the parameters give no net: modal (its transparent veil takes all clicks but OK's)
    pop = {'veil': fig.add_axes([0, 0, 1, 1], zorder=33)}
    pop['veil'].set_axis_off(); pop['veil'].set_navigate(False); pop['veil'].set_visible(False)
    QX, QY, QW, QH = 0.29, 0.39, 0.42, 0.22
    pop['frame'] = fig.add_axes([QX, QY, QW, QH], zorder=34); pop['frame'].set_axis_off(); pop['frame'].set_navigate(False)
    pop['frame'].add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03", mutation_aspect=QW*fig.get_figwidth()/(QH*fig.get_figheight()),
                                          transform=pop['frame'].transAxes, facecolor='#fffdfd', edgecolor='#c98a95', linewidth=1.3))
    pop['frame'].set_visible(False)
    pop['title'] = fig.text(QX + QW/2, QY + QH - 0.032, "", fontsize=11, fontweight='bold', color='#b00020', ha='center', va='center', zorder=35, visible=False)
    pop['text'] = fig.text(QX + QW/2, QY + QH - 0.062, "", fontsize=9, color=INK, ha='center', va='top', zorder=35, visible=False, linespacing=1.4)
    pop['ok'] = FancyButton(fig, [QX + QW/2 - 0.04, QY + 0.022, 0.08, 0.032], 'OK', fontsize=9.5); pop['ok'].ax.set_zorder(36); pop['ok'].ax.set_visible(False)
    dlg['buttons'].append(pop['ok'])
    def popup_visible(): return pop['frame'].get_visible()
    def show_popup(title, text):
        pop['title'].set_text(title); pop['text'].set_text(wrap_math(text, 64))
        for a_ in (pop['veil'], pop['frame'], pop['ok'].ax, pop['title'], pop['text']): a_.set_visible(True)
        fig.canvas.draw_idle()
    def hide_popup(ev=None):
        for a_ in (pop['veil'], pop['frame'], pop['ok'].ax, pop['title'], pop['text']): a_.set_visible(False)
        fig.canvas.draw_idle()
    pop['ok'].on_clicked(hide_popup)
    def popup_keys(ev):
        if popup_visible() and ev.key in ('enter', 'escape', ' '): hide_popup()
    fig.canvas.mpl_connect('key_press_event', popup_keys)
    build_row_buttons()
    def reset_dialog():
        """the widgets back to the current parameters of the net: Cancel discards the draft, and the dialog always opens on the truth"""
        for j, a in enumerate(BASE_ANGLES_DEG): set_box(dlg['ang%d' % j], '%g' % a); dlg['exact']['ang%d' % j] = float(a)
        set_box(dlg['M'], '%d' % m); set_box(dlg['N'], '%d' % n); set_box(dlg['kap'], '%d' % BASE_KN[0]); set_box(dlg['nu'], '%d' % BASE_KN[1])
        dlg['rowtypes'] = dict(ROW_TYPES); dlg['row_off'] = 0
        dlg['signs'], dlg['e0'] = tuple(SIGNS), E0
        for j_, b_ in enumerate(sgn_btns): b_.set_on(SIGN_PATTERNS[j_] == tuple(SIGNS))
        for j_, b_ in enumerate(e0_btns): b_.set_on(E0_VALUES[j_] == E0)
        build_row_buttons()
    def set_dialog(on):
        dlg['open'] = on; state['dlg_open'] = on or cdl['open']
        for tb_ in (t_box, th_box): tb_.set_active(not state['dlg_open'])         # under the veil of a dialog: no typing
        if on:
            try: dlg['snap'] = (fig.canvas.copy_from_bbox(fig.bbox), fig.canvas.get_width_height())   # the window as it is, for Cancel
            except Exception: dlg['snap'] = None
            reset_dialog()
        for tb in dlg.get('textboxes', []): tb.set_active(on)                    # hidden text boxes must not react to clicks (they grab the mouse)
        if not on and fig.canvas.mouse_grabber is not None:
            try: fig.canvas.release_mouse(fig.canvas.mouse_grabber)
            except Exception: pass
        ax.set_visible(not on)                                                    # the 3D scene is not redrawn while the dialog is open (speed)
        veil.set_visible(on); frame.set_visible(on)
        for a_ in dlg['axes']: a_.set_visible(on)
        for b in dlg['rowbtns']: b.ax.set_visible(on)
        for t_ in dlg['dyn']: (t_.ax.set_visible(on) if hasattr(t_, 'ax') else t_.set_visible(on))
        if not on and popup_visible(): hide_popup()
        offs['dlg'] = 0.0; offs_x['dlg'] = 0.0; relayout()
        b_par.set_on(on); fig.canvas.draw_idle()
    def nodraw(*a, **k): pass
    def end_nodraw(*a):                                                             # the next event redraws as usual again
        if vars(fig.canvas).get('draw_idle') is nodraw: del fig.canvas.draw_idle
    for ev_name in ('motion_notify_event', 'button_press_event', 'scroll_event', 'key_press_event'): fig.canvas.mpl_connect(ev_name, end_nodraw)
    def close_dialog():
        """closes the dialog when nothing was applied (Cancel, or the Params button again): the picture of the window kept when the
        dialog opened comes back at once, instead of a redraw of the whole net"""
        snap = dlg.pop('snap', None); c_ = fig.canvas
        if snap is None or snap[1] != c_.get_width_height() or not getattr(c_, 'supports_blit', False):
            set_dialog(False); return
        c_.draw_idle = nodraw                                                        # the rest of this click draws nothing
        tm = c_.new_timer(interval=0); tm.single_shot = True; tm.add_callback(end_nodraw); tm.start(); dlg['nodraw_timer'] = tm
        set_dialog(False)
        try: c_.restore_region(snap[0]); c_.blit(fig.bbox)
        except Exception: end_nodraw(); c_.draw_idle()
    b_par.on_clicked(lambda ev: set_dialog(True) if b_par.on else close_dialog())

    # ---- colours dialog (button Colours): named themes; Advanced = pick the colour of each entry from a palette (no typing)
    import matplotlib.colors as mcolors
    ENTRIES = [('LIGHT_COL', 'peak highlight'), ('SHADOW_COL', 'deepest face tone'), ('COOL_DEEP', 'rim (grazing faces)'), ('WARM_BOUNCE', 'warm bounce'),
               ('COOL_BOUNCE', 'cool bounce'), ('EDGE_COL', 'silhouette ink'), ('EDGE_SOFT', 'interior creases'), ('HIDDEN_COL', 'hidden lines'),
               ('GROUND_COL', 'shadow ink'), ('PEN_LIGHT', 'penetration, light'), ('PEN_DARK', 'penetration, dark')]
    NUMBERS = [('TINT_MIX', 'tint mix', 0.05), ('SPEC_K', 'specular', 0.04), ('GROUND_A', 'shadow opacity', 0.02)]
    cdl = {'axes': [], 'adv_axes': [], 'buttons': [], 'open': False, 'advanced': False, 'vals': dict(THEMES[DEFAULT_THEME]), 'theme': DEFAULT_THEME, 'sel': 'EDGE_COL',
           'swatch': {}, 'pal': {}, 'num_text': {}}
    if APPLIED_THEME[0] is not None: cdl['theme'], cdl['vals'] = APPLIED_THEME[0][0], dict(APPLIED_THEME[0][1]); cdl['applied'] = APPLIED_THEME[0]   # the theme applied before Params restarted the tool
    CX, CY, CW, CH = 0.25, 0.06, 0.50, 0.88
    cframe = fig.add_axes([CX, CY, CW, CH], zorder=31); cframe.set_axis_off(); cframe.set_navigate(False); reg('dlg', cframe)
    cframe.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", mutation_aspect=CW*fig.get_figwidth()/(CH*fig.get_figheight()),
                                    transform=cframe.transAxes, facecolor='#fbfcfe', edgecolor='#8fa9c9', linewidth=1.2))
    cframe.set_visible(False)
    def ctext(x, y, txt, size=9, adv=False, **kw):
        kw = {**dict(color=INK, va='center', ha='left'), **kw}
        t_ = fig.text(x, y, txt, fontsize=size, zorder=32, visible=False, **kw); (cdl['adv_axes'] if adv else cdl['axes']).append(t_); reg('dlg', t_); return t_
    def cbutton(x, y, w, label, adv=False, h=0.03, **kw):
        b = FancyButton(fig, [x, y - h/2, w, h], label, **kw); b.ax.set_zorder(32); b.ax.set_visible(False)
        (cdl['adv_axes'] if adv else cdl['axes']).append(b.ax); cdl['buttons'].append(b); reg('dlg', b.ax); return b
    def cswatch(x, y, w, h, color, adv=True):
        a_ = fig.add_axes([x, y, w, h], zorder=32); a_.set_xticks([]); a_.set_yticks([]); a_.set_visible(False); a_.set_navigate(False)
        a_.set_facecolor(mcolors.to_hex(tuple(float(v) for v in color)))
        for sp in a_.spines.values(): sp.set_edgecolor('#b9c7da'); sp.set_linewidth(0.8)
        (cdl['adv_axes'] if adv else cdl['axes']).append(a_); reg('dlg', a_); return a_
    ctext(CX + CW/2, CY + CH - 0.04, 'Picture theme', 12, ha='center', fontweight='bold')
    # --- themes
    theme_btns = {}
    for i, name in enumerate(THEMES):
        r_, c_ = divmod(i, 6)
        theme_btns[name] = cbutton(CX + 0.03 + c_*0.0735, CY + CH - 0.125 - r_*0.038, 0.068, name, toggle=True, on=(name == cdl['theme']), fontsize=8.5)
    def refresh_swatches():
        for key, a_ in cdl['swatch'].items():
            a_.set_facecolor(mcolors.to_hex(tuple(float(v) for v in cdl['vals'][key])))
            for sp in a_.spines.values(): sp.set_edgecolor('#2c4a74' if key == cdl['sel'] else '#b9c7da'); sp.set_linewidth(2.0 if key == cdl['sel'] else 0.8)
        for key, t_ in cdl['num_text'].items(): t_.set_text("%.2f" % cdl['vals'][key])
        if 'hex_box' in cdl:
            cdl['hex_box'].eventson = False; cdl['hex_box'].set_val(mcolors.to_hex(tuple(float(v) for v in cdl['vals'][cdl['sel']])).upper()); cdl['hex_box'].eventson = True
            if not cdl['hex_box'].capturekeystrokes: cdl['hex_box'].cursor.set_visible(False)   # set_val shows a text cursor (TextBox: a click no longer hides it)
        fig.canvas.draw_idle()
    def pick_theme(name):
        def f(ev):
            cdl['theme'] = name; cdl['vals'] = dict(THEMES[name])
            for nm, b in theme_btns.items(): b.set_on(nm == name)
            refresh_swatches()
        return f
    for name, b in theme_btns.items(): b.on_clicked(pick_theme(name))
    # --- advanced part: entries with swatches (click to select), a palette (click to assign), numbers with - / +
    b_adv = cbutton(CX + 0.03, CY + CH - 0.253, 0.14, 'Advanced ...', toggle=True, fontsize=9)
    ctext(CX + 0.03, CY + CH - 0.293, 'click an entry, then a colour of the wheel (or type a code)', 8, adv=True, color='#4a5b78')
    y = CY + CH - 0.328
    for key, label in ENTRIES:
        ctext(CX + 0.06, y, label, 8.5, adv=True); cdl['swatch'][key] = cswatch(CX + 0.03, y - 0.011, 0.024, 0.022, COLOR_DEFAULTS[key]); y -= 0.033
    for key, label, step in NUMBERS:
        ctext(CX + 0.06, y, label, 8.5, adv=True); cdl['num_text'][key] = ctext(CX + 0.175, y, "%.2f" % COLOR_DEFAULTS[key], 8.5, adv=True)
        def bump(key=key, step=step, d=1):
            def f(ev): cdl['vals'][key] = round(min(1.0, max(0.0, cdl['vals'][key] + d*step)), 2); refresh_swatches()
            return f
        bm = cbutton(CX + 0.215, y, 0.02, '−', adv=True, h=0.022, fontsize=8, rounding=0.3); bp = cbutton(CX + 0.238, y, 0.02, '+', adv=True, h=0.022, fontsize=8, rounding=0.3)
        bm.on_clicked(bump(d=-1)); bp.on_clicked(bump(d=+1)); y -= 0.033
    # the colour wheel (hue around, saturation outwards) with a lightness bar and a colour-code box, to the right of the entries
    import colorsys
    WX, WY, WD = CX + 0.29, CY + CH - 0.60, 0.17
    wheel_ax = fig.add_axes([WX, WY, WD, WD*fig.get_figwidth()/fig.get_figheight()], zorder=32); wheel_ax.set_axis_off(); wheel_ax.set_navigate(False)
    wheel_ax.set_visible(False); cdl['adv_axes'].append(wheel_ax); reg('dlg', wheel_ax)
    NW = 240; yy, xx = np.mgrid[0:NW, 0:NW]; xr, yr = (xx - NW/2 + 0.5)/(NW/2), (yy - NW/2 + 0.5)/(NW/2)
    rad = np.sqrt(xr**2 + yr**2); hue = (np.arctan2(yr, xr)/(2*PI)) % 1.0
    cdl['wheel_light'] = 0.6
    def wheel_image(light):
        img = np.ones((NW, NW, 4))
        h_, s_ = hue.ravel(), np.clip(rad, 0, 1).ravel()
        rgb = np.array([colorsys.hls_to_rgb(hh, light, ss) for hh, ss in zip(h_, s_)]).reshape(NW, NW, 3)
        img[..., :3] = rgb; img[..., 3] = np.clip((1.0 - rad)*(NW/2) + 0.5, 0.0, 1.0)      # one-pixel soft edge: a smooth circle
        return img
    wheel_im = wheel_ax.imshow(wheel_image(0.6), extent=(-1, 1, -1, 1), origin='lower', interpolation='bilinear')
    ctext(WX, WY + WD*fig.get_figwidth()/fig.get_figheight() + 0.02, 'colour wheel', 8.5, adv=True)
    ctext(WX, WY - 0.03, 'lightness', 8.5, adv=True)
    lax = fig.add_axes([WX, WY - 0.055, WD, 0.014], zorder=32); lax.set_visible(False); lax.set_navigate(False)
    cdl['adv_axes'].append(lax); reg('dlg', lax)
    light_sl = Slider(lax, '', 0.05, 0.95, valinit=0.6, color='#8fb6e3'); light_sl.valtext.set_visible(False); light_sl.set_active(False)
    cdl['buttons'].append(light_sl)
    def on_light(val):
        cdl['wheel_light'] = float(val); wheel_im.set_data(wheel_image(float(val))); fig.canvas.draw_idle()
    light_sl.on_changed(on_light)
    ctext(WX, WY - 0.09, 'code', 8.5, adv=True)
    hax = fig.add_axes([WX + 0.045, WY - 0.104, 0.09, 0.026], zorder=32); hax.set_visible(False); cdl['adv_axes'].append(hax); reg('dlg', hax)
    hex_box = TextBox(hax, '', initial='#2C4470', textalignment='center', color='#ffffff', hovercolor='#eef3fa'); hex_box.text_disp.set_fontsize(9)
    hex_box.set_active(False); cdl['buttons'].append(hex_box); cdl['hex_box'] = hex_box
    def on_hex(txt):
        try: cdl['vals'][cdl['sel']] = np.array(mcolors.to_rgb(txt.strip())); cmsg.set_visible(False)
        except ValueError: cmsg.set_text("not a colour: %s (use #RRGGBB or a matplotlib colour name)" % txt.strip()); cmsg.set_visible(True)
        refresh_swatches()
    hex_box.on_submit(on_hex)
    def dialog_click(ev):
        if not cdl['open'] or ev.button != 1 or ev.inaxes is None: return
        if ev.inaxes is wheel_ax and cdl['advanced']:
            x_, y_ = ev.xdata, ev.ydata
            if x_ is not None and x_*x_ + y_*y_ <= 1.0:
                h_ = (math.atan2(y_, x_)/(2*PI)) % 1.0; s_ = min(1.0, math.hypot(x_, y_))
                cdl['vals'][cdl['sel']] = np.array(colorsys.hls_to_rgb(h_, cdl['wheel_light'], s_)); refresh_swatches()
            return
        for key, a_ in cdl['swatch'].items():
            if ev.inaxes is a_: cdl['sel'] = key; refresh_swatches(); return
    fig.canvas.mpl_connect('button_press_event', dialog_click)
    cmsg = fig.text(CX + CW/2, CY + 0.062, "", fontsize=8.5, color='#b00020', ha='center', va='bottom', zorder=32, visible=False); reg('dlg', cmsg)
    b_capply = cbutton(CX + CW/2 - 0.11, CY + 0.03, 0.10, 'Apply'); b_ccancel = cbutton(CX + CW/2 + 0.01, CY + 0.03, 0.10, 'Cancel')
    def show_advanced(on):
        cdl['advanced'] = on
        for a_ in cdl['adv_axes']: a_.set_visible(on and cdl['open'])
        cdl['hex_box'].set_active(on and cdl['open']); light_sl.set_active(on and cdl['open'])   # hidden: no mouse (it lies over the picture)
        b_adv.set_on(on); fig.canvas.draw_idle()
    b_adv.on_clicked(lambda ev: show_advanced(b_adv.on))
    def set_cdialog(on):
        cdl['open'] = on; state['dlg_open'] = on or dlg['open']
        ax.set_visible(not on); veil.set_visible(on); cframe.set_visible(on)
        for a_ in cdl['axes']: a_.set_visible(on)
        for a_ in cdl['adv_axes']: a_.set_visible(on and cdl['advanced'])
        cdl['hex_box'].set_active(on and cdl['advanced']); light_sl.set_active(on and cdl['advanced'])
        for tb_ in (t_box, th_box): tb_.set_active(not state['dlg_open'])         # under the veil of a dialog: no typing
        if not on:
            cmsg.set_visible(False)
            if fig.canvas.mouse_grabber is not None:
                try: fig.canvas.release_mouse(fig.canvas.mouse_grabber)
                except Exception: pass
        offs['dlg'] = 0.0; offs_x['dlg'] = 0.0; relayout(); refresh_swatches()
        b_col.set_on(on); fig.canvas.draw_idle()
    def apply_canvas(vals):
        """the canvas colour and the ink of the panels and the title follow the theme (a dark theme needs pale ink)."""
        fig.set_facecolor(vals.get('FACE_COLOR', '#FFFFFF')); ink = vals.get('INK_COLOR', '#2c4a74')
        state['canvas_color'] = vals.get('FACE_COLOR', '#FFFFFF'); ax.set_facecolor(state['canvas_color'])   # Axes3D draws its patch even with the axis off
        for t_ in (report_text, face_text): t_.set_color(ink); t_.get_bbox_patch().set_facecolor(vals.get('FACE_COLOR', '#FFFFFF') if ink != '#2c4a74' else PANEL_FC)
        title_text.set_color(ink)
    def capply(ev):
        try: apply_colors(cdl['vals']); apply_canvas(cdl['vals'])
        except Exception as ex:
            cmsg.set_text("cannot apply: %s" % (ex,)); cmsg.set_visible(True); fig.canvas.draw_idle(); return
        cdl['applied'] = APPLIED_THEME[0] = (cdl['theme'], dict(cdl['vals']))         # (APPLIED_THEME: also after a restart by Params)
        set_cdialog(False); redraw()                                                # the colours change; the zoom and the view stay
    def ccancel(ev):
        """Cancel: back to the applied theme (the draft would otherwise be exported in the side-car and shown when the dialog reopens)"""
        th, vals = cdl.get('applied', (DEFAULT_THEME, dict(THEMES[DEFAULT_THEME])))
        cdl['theme'] = th; cdl['vals'] = dict(vals)
        for nm, b in theme_btns.items(): b.set_on(nm == th)
        set_cdialog(False)
    b_capply.on_clicked(capply); b_ccancel.on_clicked(ccancel)
    if APPLIED_THEME[0] is not None: apply_canvas(cdl['vals'])                    # restarted by Params: the canvas of the applied theme (the materials are module globals, kept)
    def col_click(ev):
        if dlg['open']: set_dialog(False)
        set_cdialog(b_col.on)
    b_col.on_clicked(col_click)
    b_cancel.on_clicked(lambda ev: close_dialog())                                 # the dialog is reset when it opens again
    def apply_params(ev):
        """checks the draft and applies it; whatever gives no net is explained in the message box, and nothing changes"""
        global BASE_ANGLES_DEG, M_FACES, N_FACES, BASE_KN, ROW_TYPES, SIGNS, E0
        sg, e0_ = tuple(dlg['signs']), dlg['e0']
        def value(key, name, whole=False):
            txt = dlg[key].text.strip(); ex_ = dlg['exact'].get(key)
            if not whole and ex_ is not None and txt == '%g' % ex_: return float(ex_)   # the box still shows its value: all its digits, not the 6 shown
            try: v = float(txt.replace(',', '.').replace('\u00b0', '').replace('\u2212', '-'))
            except ValueError: raise ValueError("%s: '%s' is not a number" % (name, txt))
            if whole and (not math.isfinite(v) or v != int(v)): raise ValueError("%s: '%s' is not a whole number" % (name, txt))
            return int(v) if whole else v
        try:
            angs = tuple(value('ang%d' % j, nm) for j, nm in enumerate((r"$\alpha_1$", r"$\beta_1$", r"$\gamma_1$", r"$\delta_1$")))
            mm, nn_ = value('M', "$m$", True), value('N', "$n$", True)
            kk, nu = value('kap', r"$\kappa_0$", True), value('nu', r"$\nu_0$", True)
            problem = angle_problem(angs)                                                   # the hypotheses on the base angles, in words
            if problem: raise ValueError(problem)
            if mm < 3 or nn_ < 3: raise ValueError(r"the net needs at least $3 \times 3$ faces; here $m$ = %d, $n$ = %d" % (mm, nn_))
            if mm > MAX_MN or nn_ > MAX_MN or mm*nn_ > MAX_FACES:                       # (before anything of that size is built)
                raise ValueError(r"the net can have at most %d faces, at most %d along a side (MAX_FACES, MAX_MN at the top of the file: a larger net "
                                 r"takes minutes to start); here $m$ = %d, $n$ = %d" % (MAX_FACES, MAX_MN, mm, nn_))
            if not (1 <= kk <= mm - 2 and 1 <= nu <= nn_ - 2):
                raise ValueError(r"the base block must satisfy $1 \leq \kappa_0 \leq m-2$ and $1 \leq \nu_0 \leq n-2$; here $\kappa_0$ = %d, $\nu_0$ = %d, $m$ = %d, $n$ = %d" % (kk, nu, mm, nn_))
            rows_new = {r: dlg['rowtypes'][r] for r in range(1, nn_ - 1) if r in dlg['rowtypes']}
            if nu not in rows_new: rows_new[nu] = 'a'
            rows_ok = check_parameters(mm, nn_, (kk, nu), rows_new, verbose=False)
            same_net = (all(abs(a_ - float(b_)) <= 1e-12*abs(float(b_)) for a_, b_ in zip(angs, BASE_ANGLES_DEG)) and (mm, nn_) == (m, n) and (kk, nu) == tuple(BASE_KN)
                        and all(rows_ok[r] == ROW_TYPES.get(r) for r in range(1, nn_ - 1)))
            if same_net:                                                                # only the signs: the window stays
                if (sg, e0_) == (tuple(SIGNS), E0): close_dialog(); return               # nothing changed at all
                if not switch_flexion(signs=sg, e0=e0_):
                    raise ValueError("the flexion with these signs could not be built in space for this net (numerically)")
                set_dialog(False); return
            sub_t = assemble_net(angs, mm, nn_, (kk, nu), rows_ok)                     # trial assembly, flexion and construction
            try: subs_t = {kn: Block(kn[0], kn[1], q) for kn, q in sub_t.items()}      # before touching the window
            except (ValueError, AssertionError, ZeroDivisionError) as ex:
                raise ValueError("in floating point, the flexion of the blocks could not be computed (%s). For these angles the flexion "
                                 "exists, so this is rounding, for angles very close to violating one of the conditions: change one of them slightly" % (ex,))
            lo_t, hi_t = admissible_t_range(subs_t[(kk, nu)])
            th_t = None
            for f_ in (0.15, 0.05, 0.25, 0.35, 0.5, 0.65, 0.8, 0.92):
                th_t = synchronize(subs_t, (kk, nu), sg, e0_, lo_t + f_*(hi_t - lo_t))
                if th_t is not None: break
            if th_t is None:
                raise ValueError("in floating point, the blocks could not be made to agree on their common faces. For these angles "
                                 "the flexion exists, so this is rounding, for angles very close to violating one of the conditions: "
                                 "change one of them slightly")
            try: build_net(vertex_face_angles(sub_t), th_t, mm, nn_, (kk, nu), BASE_EDGE)
            except AssertionError as ex:
                spread = side_spread(vertex_face_angles(sub_t), mm, nn_, (kk, nu))
                if spread > 1e6:
                    raise ValueError("in floating point, the construction of the net in space failed (%s). These angles force sides of the "
                                     "central faces that differ by a factor of %.0e, too much for double precision: make the smallest angle larger"
                                     % (ex.args[0] if ex.args else ex, spread))
                raise ValueError("in floating point, the construction of the net in space failed (%s). For these angles the net "
                                 "exists, so this is rounding: change one of the angles slightly" % (ex.args[0] if ex.args else ex,))
        except Exception as ex:
            if not isinstance(ex, (ValueError, AssertionError, ZeroDivisionError)): import traceback; traceback.print_exc()   # not a parameter problem: a bug, in full on the console
            show_popup("These parameters give no net", str(ex.args[0] if ex.args else ex)); return
        BASE_ANGLES_DEG, M_FACES, N_FACES, BASE_KN, ROW_TYPES, SIGNS, E0 = angs, mm, nn_, (kk, nu), rows_ok, sg, e0_
        close_dihedrals(); plt.close(fig)
        state['rebuild'] = True
    b_apply.on_clicked(apply_params)
    # ---- the small buttons on the panels: (i) opens the explanations, (x) closes the panel
    def small(label, toggle=False, italic=False):
        b = FancyButton(fig, [0.4, 0.9, 0.022, 0.026], label, toggle=toggle, fontsize=8.5, rounding=0.3, italic=italic); b.ax.set_zorder(20)
        b.ax.set_visible(False); return b
    b_info = small('i', toggle=True, italic=True); b_x_rep, b_x_face = small('×'), small('×')
    # ---- the view lines of the report: elevation and azimuth of the view and the zoom = the size of the picture against the
    # startup frame mid +- rng (1 at the start and after Reset, 2 = twice as large); they follow the view, and a typed value sets it
    def zoom_now():                                                                 # the mean of the three half-ranges (all equal unless
        return rng/float(np.mean([(b_ - a_)/2 for a_, b_ in (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())]))   # the toolbar's box zoom was used)
    def norm_deg(v_): w_ = (v_ + 180.0) % 360.0 - 180.0; return v_ if -180.0 < v_ <= 180.0 else (180.0 if w_ == -180.0 else w_)   # an angle in (-180, 180] (as is when there)
    def box_deg(v_): s_ = "%.1f" % norm_deg(v_); return {"-180.0": "180.0", "-0.0": "0.0"}.get(s_, s_) + "°"
    def box_zoom(z_): return "%.2f" % z_ if 0.1 <= z_ < 100 else ("%.4g" % z_ if 100 <= z_ < 1e4 else "%.3g" % z_)   # 3-4 digits also far out
    vbox, vcache = {}, {}
    for key_ in ('elev', 'azim', 'zoom'):                                          # (placed by layout_panels, over the blank columns of the view lines)
        vb_ = PanelBox(fig.add_axes([0.0, 0.0, 0.05, 0.02], zorder=6, visible=False),
                       lambda: bool(state.get('dlg_open')) or fast['kind'] in ('rotate', 'drag') or state.get('press') is not None or bool(state.get('rotating')),
                       textalignment='center', color='#ffffff', hovercolor='#ffffff')   # no hover colour: crossing a box over the picture costs no redraw
        vb_.text_disp.set_fontsize(8); vb_.text_disp.set_color('#2c4a74')
        for sp in vb_.ax.spines.values(): sp.set_edgecolor('#b9c7da'); sp.set_linewidth(0.8)
        vb_._shown = None; vb_.on_text_change(lambda txt, tb_=vb_: txt != tb_._shown and setattr(tb_, '_shown', None))   # an edit counts (not Enter or Tab alone)
        vbox[key_] = vb_
    def sync_view():                                                                # the boxes show the view and the zoom (no draw)
        put_box(vbox['elev'], box_deg(ax.elev)); put_box(vbox['azim'], box_deg(ax.azim)); put_box(vbox['zoom'], box_zoom(zoom_now()))
    def view_submit(key_):
        def f(text):
            if text == vbox[key_]._shown: return                                        # the box as shown, not edited: the view stays (not rounded to the display)
            try: v = float(text.strip().replace(',', '.').replace('\u2212', '-').replace('°', '').replace('deg', '').replace('\u00d7', '').strip())
            except ValueError: v = math.nan
            if not math.isfinite(v) or (key_ == 'zoom' and not 0.01 <= v <= 1000):
                sync_view()                                                             # the box shows the view again; why, in the report
                note("%s: '%s' is not a number" % (key_, text.strip().replace('$', r'\$')) if math.isnan(v) else "%s: must be a finite number" % key_ if key_ != 'zoom' else
                     "zoom: must lie in [0.01, 1000] (1 = the startup frame)", key=key_); return
            if key_ == 'zoom':
                for get_, set_ in ((ax.get_xlim3d, ax.set_xlim3d), (ax.get_ylim3d, ax.set_ylim3d), (ax.get_zlim3d, ax.set_zlim3d)):
                    c_ = sum(get_())/2; set_(c_ - rng/v, c_ + rng/v)                     # about the current centre: a pan stays
            else:
                el_, az_ = (norm_deg(v), ax.azim) if key_ == 'elev' else (ax.elev, norm_deg(v))
                folded = abs(norm_deg(el_)) > 90                                        # upside down in mplot3d (the first drag flips it): the
                if folded: el_, az_ = math.copysign(180.0, norm_deg(el_)) - norm_deg(el_), norm_deg(az_ + 180.0)   # same eye from the other side, upright
                ax.view_init(elev=el_, azim=az_)
            redraw()                                                                    # (redraw keeps the limits and the view, and syncs the boxes)
            if key_ != 'zoom' and folded:
                note("elev: %s is the view from elev %s, azim %s (upright)" % (box_deg(v), box_deg(ax.elev), box_deg(ax.azim)), key='elev')
        return f
    for key_, vb_ in vbox.items(): vb_.on_submit(view_submit(key_))
    state['sync_view'] = sync_view; sync_view()
    def place_view():
        """the view boxes over the blank columns of the view lines (the report is monospace: a column is the width of a line over its
        length; the rows are matplotlib's own layout of the text, exact also under lines with mathtext); sizes in the text's units"""
        r_ = state.get('renderer') or fig.canvas.get_renderer()
        key = (report_text.get_text(), report_text.get_fontsize(), fig.dpi, getattr(r_, 'dpi', None))
        fs_ = report_text.get_fontsize()*fig.dpi/72
        if vcache.get('key') != key:
            k_ = report_text.get_text().split("\n").index(VIEW_LINES[1])
            try: info = report_text._get_layout(r_)[1]; rows = (info[k_][3], info[k_ + 2][3])
            except Exception: rows = (-(0.73 + 1.09*k_)*fs_, -(0.73 + 1.09*(k_ + 2))*fs_)   # (a uniform line pitch)
            cell = r_.get_text_width_height_descent(VIEW_LINES[1], report_text.get_fontproperties(), ismath=False)[0]/len(VIEW_LINES[1])
            vcache.clear(); vcache.update(key=key, rows=rows, cell=cell)
        X, Y = fig.transFigure.transform(report_text.get_position()); c_, h_ = vcache['cell'], 1.5*fs_; W_, H_ = fig.bbox.width, fig.bbox.height
        for b_, col, row in ((vbox['elev'], len(VIEW_A), 0), (vbox['azim'], len(VIEW_A) + VIEW_W + len(VIEW_B), 0), (vbox['zoom'], len(VIEW_A), 1)):
            b_.ax.set_position([(X + col*c_)/W_, (Y + vcache['rows'][row] + 0.38*fs_ - h_/2)/H_, VIEW_W*c_/W_, h_/H_]); b_.ax.set_visible(True)
    def refresh_panels():
        report_text.set_text(report_string()); report_text.set_visible(state['report']); layout_panels(); fig.canvas.draw_idle()
    extent_cache = {}
    def measure(text):
        """window extent of a text with the best renderer at hand, never triggering a draw (a draw from inside a draw event freezes
        the interface): the renderer of the last draw event, else the canvas renderer. A panel text that did not change is not
        laid out again: its last extent is shifted to the new position (the wheel moves the long explanations many times)."""
        key = (id(text), text.get_text(), text.get_fontsize(), text.get_visible(), fig.dpi, tuple(fig.bbox.size))
        pos = text.get_position(); hit = extent_cache.get(key)
        if hit is not None and text.get_transform() is fig.transFigure:
            (px, py), bb0 = hit
            return bb0.translated((pos[0] - px)*fig.bbox.width, (pos[1] - py)*fig.bbox.height)
        r = state.get('renderer')
        try: bb = text.get_window_extent(r) if r is not None else text.get_window_extent()
        except Exception:
            try: bb = text.get_window_extent(fig.canvas.get_renderer())
            except Exception: return Bbox.from_extents(0, 0, 1, 1)
        if len(extent_cache) > 64: extent_cache.clear()
        extent_cache[key] = (pos, bb)
        return bb
    def corner_of(text):
        """figure coordinates of the top-right corner of the box of a panel text (whatever its current size)."""
        bb = measure(text)
        try: padf = text.get_bbox_patch().get_boxstyle().pad
        except Exception: padf = 0.6
        pad = padf*text.get_fontsize()*fig.dpi/72
        return fig.transFigure.inverted().transform((bb.x1 + pad, bb.y1 + pad))
    def bottom_of(text):
        """figure coordinates of the bottom-left corner of the box of a panel text."""
        bb = measure(text)
        try: padf = text.get_bbox_patch().get_boxstyle().pad
        except Exception: padf = 0.6
        pad = padf*text.get_fontsize()*fig.dpi/72
        return fig.transFigure.inverted().transform((bb.x0 - pad, bb.y0 - pad))
    def extent(text):
        """the text extent in figure fractions, measured with the most recent renderer (after a resize the old one is stale)."""
        bb = measure(text)
        inv = fig.transFigure.inverted(); (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        return x0, y0, x1, y1
    def place_small(btns, text):
        """the round buttons at the right end of the first line, inside the box: sizes and gaps in inches."""
        W, H = fig.get_figwidth(), fig.get_figheight()
        x0, y0, x1, y1 = extent(text)
        bw, bh = 0.24/W, 0.22/H
        mid = 0.5*text.get_fontsize()/72.0                                                # the middle of the first line below the top of the text (inches)
        for k, b in enumerate(btns):
            b.ax.set_position([x1 - bw - k*(bw + 0.05/W), y1 - (mid + 0.11)/H, bw, bh]); b.ax.set_visible(True)
    def panel_max_off():
        """how far the left panels can be scrolled up: until the lowest one (or the plot under the face panel) ends just above the
        bottom of the window; 0 when they fit (figure fractions)"""
        lows = [measure(t).y0/fig.bbox.height for t in (report_text, face_text) if t.get_visible()]
        if inset.get_visible(): lows.append(inset.get_position().y0 - 0.45/fig.get_figheight())
        return max(0.0, -min(lows) + 0.02 + offs['panel']) if lows else 0.0
    def layout_panels(clamp=True):
        """(x) and (i) in the top-right corners of the panels (over the short first lines only), and the view boxes on their lines of the
        report (place_view). The face panel sits under the
        report box (or to its right while the explanations are unrolled), and its plot under the face panel, with gaps and a plot
        size fixed in inches, so that nothing overlaps whatever the window size; the mouse wheel scrolls the whole left group."""
        W, H = fig.get_figwidth(), fig.get_figheight()
        xf, top = 0.012 - offs_x['panel'], 0.968 + offs['panel']
        report_text.set_position((xf, top))
        if report_text.get_visible():
            x1, y1 = corner_of(report_text)
            place_small([b_x_rep, b_info], report_text); place_view()
            if state['unrolled'] > 0: xf = x1 + 0.012                              # explanations unrolled: the face panel goes to the right (x1 includes the scroll)
            else: top = bottom_of(report_text)[1] - 0.25/H                         # otherwise under the report box
        else:
            b_x_rep.ax.set_visible(False); b_info.ax.set_visible(False)
            for b_ in vbox.values(): b_.ax.set_visible(False)
        face_text.set_position((xf, top))
        if face_text.get_visible():
            place_small([b_x_face], face_text)
            x0, y0 = bottom_of(face_text)                                         # the plot goes right under the panel, fixed size in inches
            w, h = 2.3/W, 1.9/H
            inset.set_position([xf + 0.55/W, y0 - 0.55/H - h, w, h])
        else:
            b_x_face.ax.set_visible(False)
        if clamp and (offs['panel'] > panel_max_off() + 1e-6 or offs_x['panel'] > overflow_x('panel') + 1e-6):   # the panels shrank (report hidden or
            offs['panel'] = min(offs['panel'], panel_max_off()); offs_x['panel'] = min(offs_x['panel'], overflow_x('panel'))   # rolled up, face closed):
            layout_panels(False)                                                  # no scroll beyond their end, the wheel reaches them
    state['layout_panels'] = layout_panels
    def after_draw(ev):
        """the (x)/(i) buttons, the view boxes and the plot are placed from the measured text boxes; after a resize the boxes are only known once the
        figure has been drawn, so their placement is corrected here (a second draw is requested only if something moved)."""
        if state.get('in_after_draw') or fast['capturing']: return              # never re-enter; not in the capture draw (panels hidden on purpose)
        state['in_after_draw'] = True
        try:
            state['renderer'] = getattr(ev, 'renderer', None)
            arts = (inset, b_x_rep.ax, b_info.ax, b_x_face.ax, *[b_.ax for b_ in vbox.values()])
            before = [tuple(a.get_position().bounds) for a in arts]
            layout_panels(False)                                                # (no clamp of the scroll: a fast repaint or a saved picture draws without the panels)
            after = [tuple(a.get_position().bounds) for a in arts]
            moved = any(max(abs(u - v) for u, v in zip(p, q)) > 1e-3 for p, q in zip(before, after))
            state['redraw_chain'] = state.get('redraw_chain', 0) + 1 if moved else 0
            if moved and state['redraw_chain'] <= 2: fig.canvas.draw_idle()   # at most two corrective draws in a row
        finally:
            state['in_after_draw'] = False
    fig.canvas.mpl_connect('draw_event', after_draw)
    def set_info(ev):
        state['info'] = b_info.on; state['unrolled'] = len(INFO_LINES) if state['info'] else 0; refresh_panels()
    def set_report(ev):
        state['report'] = b_rep.on; refresh_panels()
    def close_report(ev):                               # (x): the whole box goes, rolled up again when reopened
        state['report'] = False; state['info'] = False; state['unrolled'] = 0
        b_rep.set_on(False); b_info.set_on(False); refresh_panels()
    b_rep.on_clicked(set_report); b_info.on_clicked(set_info); b_x_rep.on_clicked(close_report); b_x_face.on_clicked(hide_face)
    layout_panels()
    # ---- saving
    def t_txt(fmt):                                         # the signed t by fmt when that is exact (round values), else to 1e-6 (the configurations
        v_ = -state['t'] if state.get('mirror') else state['t']; s_ = fmt % v_     # are kept per 1e-6 of t), by %.6g next to the flat end
        return s_ if abs(float(s_) - v_) <= 1e-9 else ("%.6g" % v_ if abs(v_) < 1e-3 else "%.6f" % v_)
    def fname():                                            # + the flexion when it is not signs (1, -1, 1, -1), e0 = 1: two flexions at one t differ
        flex = "" if (tuple(SIGNS), E0) == (SIGN_PATTERNS[0], 1) else "_s%s_e%+d" % ("".join("+" if s_ > 0 else "-" for s_ in SIGNS), E0)
        return "gqs_net_%dx%d_t%s%s" % (m, n, t_txt("%.3f"), flex)
    def picture_bbox():
        """bounding box (inches) of the drawn net and its shadow on the screen, with a margin: the crop for the saved pictures."""
        pts = list(state['pos'].values())
        if state.get('shadow_box'):
            x0, x1, y0, y1, z0 = state['shadow_box']; pts += [np.array([x, y, z0]) for x in (x0, x1) for y in (y0, y1)]
        if state.get('shadow_pts'): pts += list(state['shadow_pts'])
        D = ax.transData.transform(project(ax, pts)[:, :2])/fig.dpi
        lo_, hi_ = D.min(axis=0), D.max(axis=0); mg = 0.04*(hi_ - lo_).max() + 0.25
        return Bbox.from_extents(lo_[0] - mg, lo_[1] - mg, hi_[0] + mg, hi_[1] + mg)
    def save(fn, **kw):
        if SAVE_FULL_WINDOW:
            fig.savefig(fn, **kw); return
        others = [a for a in fig.axes if a is not ax and a.get_visible()] + [t for t in fig.texts if t.get_visible()]
        for a in others: a.set_visible(False)                                  # picture only: no panels, no controls
        try: fig.savefig(fn, bbox_inches=picture_bbox(), **kw)
        finally:
            for a in others: a.set_visible(True)
            fig.canvas.draw_idle()
    def save_png(ev): save(fname() + ".png", dpi=PNG_DPI)
    def save_svg(ev):
        polys = [c for c in ax.collections if isinstance(c, LayerPolys)]
        for c in polys: c.set_edgecolor(c.get_facecolor()); c.set_linewidth(0.4)  # same-colour strokes: no hairline seams in vector viewers
        try: save(fname() + ".svg")
        finally:
            for c in polys: c.set_edgecolor('none'); c.set_linewidth(1.0)
    def save_obj(ev):
        fn = fname()
        with open(fn + ".obj", "w", encoding="utf-8") as fh:
            idx = {}
            for k, (vv, p) in enumerate(sorted(state['pos'].items())):
                idx[vv] = k + 1; fh.write("v %.10f %.10f %.10f\n" % tuple(p))
            for a in range(m):
                for b in range(n):
                    fh.write("f %d %d %d %d\n" % tuple(idx[x] for x in [(a, b), (a+1, b), (a+1, b+1), (a, b+1)]))
        with open(fn + "_report.txt", "w", encoding="utf-8") as fh:                     # utf-8: the report has '∈' (not in cp1252, Windows' default)
            fh.write("signs (e1, e2, e3, e4) of all blocks: %s,  sign e0 of the base block F%d%d: %+d,  t = %s\n\n"
                     % (tuple(SIGNS), BASE_KN[0], BASE_KN[1], E0, t_txt("%.6f")))
            fh.write(verify(state['pos'], state['th'], state['t'], tex=False) + "\n" + admissible_sets_text(False) + "\n\nEdge lengths:\n")
            for e in edges_all: P, Q = sorted(e); fh.write("  V%d%d-V%d%d: %.10f\n" % (P + Q + (edge_length(state['pos'], e),)))
            fh.write("\nFlat angles at inner vertices (deg), measured / prescribed:\n")
            for vv in sorted(ang):
                for f in sorted(ang[vv]): fh.write("  V%d%d in F%d%d: %.6f / %.6f\n" % (vv + f + (math.degrees(face_angle(state['pos'], vv, f)), math.degrees(ang[vv][f]))))
            fh.write("\nDihedral angles of the 3 x 3 blocks (deg), and the sign e0 of each block:\n")
            for kn in sorted(state['th']): fh.write("  F%d%d: %s  e0 = %s\n" % (kn + (str([round(math.degrees(state['th'][kn][i]), 6) for i in range(1, 5)]), sign_str(kn, tex=False))))
    def write_obj(fn, pos_):
        with open(fn, "w", encoding="utf-8") as fh:
            idx = {}
            for k, (vv, p) in enumerate(sorted(pos_.items())):
                idx[vv] = k + 1; fh.write("v %.10f %.10f %.10f\n" % tuple(p))
            for a in range(m):
                for b in range(n):
                    fh.write("f %d %d %d %d\n" % tuple(idx[x] for x in [(a, b), (a+1, b), (a+1, b+1), (a, b+1)]))
    def save_motion(ev):
        """MOTION_FRAMES configurations over the slider's range for t >= 0 (from the flat end where t = 0 is admissible) as OBJ files. Every frame is checked from its vertices as the
        drawn net is (check_values) and tested for penetrating faces; frames that fail are skipped. An index file lists the parameter
        of each frame and ends with the largest deviations over all frames. Written to the folder gqs_net_MxN_motion next to the
        other saves; the frames and the index of an earlier Motion there are removed first (other files stay)."""
        folder = "gqs_net_%dx%d_motion" % (m, n); os.makedirs(folder, exist_ok=True)
        R = theta_regions(); th_lo_, th_hi_ = R['tail'][0][1], R['live'][-1][1] if R['live'] else R['tail'][1][0]   # the sampled range of theta_1
        ths_ = np.linspace(th_hi_, th_lo_, MOTION_FRAMES); kept = 0; skipped = 0; pen = 0; failed = 0         # evenly in the angle, from the upper end (180 itself at a flat end)
        worst = dict(len=0.0, fa=0.0, flat=0.0, plan=0.0, dih=0.0); worst_failed = 0.0
        L_max, L_min = max(ref_lengths.values()), min(ref_lengths.values())
        tol_ = dict(len=CHECK_TOL*L_max, **{key: max(CHECK_TOL, 1e-10*L_max/L_min) for key in ('fa', 'flat', 'plan', 'dih')})   # the rounding level (CHECK_TOL)
        for f_ in os.listdir(folder):                                                   # an earlier Motion of this size (other flexion or net): no mixed sequence
            if re.fullmatch(r"frame_\d{3,}\.obj|index\.txt", f_): os.remove(os.path.join(folder, f_))
        with open(os.path.join(folder, "index.txt"), "w", encoding="utf-8") as fh:
            fh.write("# frame  theta_1 (deg, base block)  t = cot(theta_1/2)  file\n")
            for k, th_ in enumerate(ths_):
                t_ = t_of_theta(float(th_))
                try: cfg = configuration(float(t_))
                except (ArithmeticError, ValueError, AssertionError): cfg = None
                except Exception: report_bug("Motion"); cfg = None
                if cfg is None: skipped += 1; continue
                cv = check_values(cfg[0], cfg[3])
                if not cv['convex'] or any(cv[key] > tol_[key] for key in worst):
                    failed += 1; worst_failed = max([worst_failed] + [cv[key]/tol_[key] for key in worst]); continue
                if self_intersections(cfg[0], faces_all, corners): pen += 1; continue
                for key in worst: worst[key] = max(worst[key], cv[key])
                fn = "frame_%03d.obj" % kept; write_obj(os.path.join(folder, fn), cfg[0]); fh.write("%3d  %9.4f  %.6f  %s\n" % (kept, th_, t_, fn)); kept += 1
            fh.write("# checks of all frames, from their vertices (largest deviations): edge lengths vs. initial %.1e, face angles vs. "
                     "initial %.1e rad, flat angles vs. prescribed %.1e rad, planarity %.1e, dihedral angles vs. formulas %.1e rad; all faces "
                     "convex\n" % (worst['len'], worst['fa'], worst['flat'], worst['plan'], worst['dih']) if kept else "# no frame kept: nothing checked\n")
        note = "%d frames skipped: %d penetration, %d no construction" % (pen + skipped, pen, skipped) + (", %d failed a check (up to %.0f times its tolerance)" % (failed, worst_failed) if failed else "")
        state['report_str'] = state['report_str'].rstrip() + "\n  motion saved: %d OBJ files in %s (%s)" % (kept, folder, note)
        report_text.set_text(report_string()); layout_panels(); fig.canvas.draw_idle()
    def write_sidecar(fn_json):
        """the view and the picture theme next to an OBJ, for render_net.py (Blender), and the configuration shown (signed t, signs, e0)."""
        cols = {k: (list(map(float, v)) if not isinstance(v, (str, float, int)) else v) for k, v in cdl['vals'].items()}
        cols.update(EDGE_COL=list(EDGE_COL), EDGE_SOFT=list(EDGE_SOFT), HIDDEN_COL=list(HIDDEN_COL), LIGHT_COL=list(map(float, LIGHT_COL)),
                    SHADOW_COL=list(map(float, SHADOW_COL)), WARM_BOUNCE=list(map(float, WARM_BOUNCE)), COOL_BOUNCE=list(map(float, COOL_BOUNCE)),
                    GROUND_COL=list(GROUND_COL), GROUND_A=float(GROUND_A), FACE_COLOR=cdl['vals'].get('FACE_COLOR', '#FFFFFF'))
        meta = dict(elev=float((state['elev'] + 180.0) % 360.0 - 180.0), azim=float((state['azim'] + 180.0) % 360.0 - 180.0), focal_length=FOCAL_LENGTH, light_frame=state.get('light_frame', LIGHT_FRAME),
                    size=[m, n], t=float(-state['t'] if state.get('mirror') else state['t']), theme=cdl['theme'], colors=cols, hidden=bool(state.get('hidden', True)),
                    mirror=bool(state.get('mirror')), signs=[int(s_) for s_ in SIGNS], e0=int(E0))   # t as in the file name (negative on the mirrored half; mirror: also for t = -0)
        if state.get('light_frame', LIGHT_FRAME) == 'room':                                    # the studio's floor and lamps, for the whole flexion
            meta['floor_z'] = -float(room_height()); meta['room_lamp_view'] = [float(x) for x in ROOM_LAMP_VIEW]
        with open(fn_json, "w", encoding="utf-8") as fh: json.dump(meta, fh, indent=1)
    _save_obj = save_obj
    def save_obj_with_meta(ev):
        _save_obj(ev); write_sidecar(fname() + ".json")
    def render_blender(ev):
        """export the current configuration and render it with Blender in the background (render_net.py next to this file);
        the progress of Cycles ('Sample k/N' lines) is shown in the report box."""
        import subprocess, threading, signal, re as _re
        recipe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_net.py")   # a render_net.py next to the script overrides the built-in recipe
        blender = find_blender()
        if blender is None:
            note("Blender not found at BLENDER_PATH = %s: correct it at the top of the file, or set it to \"\" to search the usual install folders" % BLENDER_PATH if BLENDER_PATH else
                 "Blender not found: no 'blender' on the PATH and none in the usual install folders; set BLENDER_PATH at the top of the file"); return
        proc = state.get('blender_proc')
        if proc is not None and proc.poll() is None:
            note("Blender: a render is still running, wait for it", key='blender'); return
        fn = fname(); _save_obj(ev); write_sidecar(fn + ".json"); out = fn + "_blender.png"
        for stale in (out, out + ".progress"):                                   # never report an old picture as the new one
            try: os.remove(stale)
            except Exception: pass
        tmp_recipe = None
        if not os.path.exists(recipe):
            try:
                import tempfile
                fd, tmp_recipe = tempfile.mkstemp(prefix="gqs_render_net_", suffix=".py"); recipe = tmp_recipe   # a new file of its own (a fixed name may be taken), removed after the render
                with os.fdopen(fd, "w", encoding="utf-8") as fh: fh.write(RENDER_RECIPE)             # the recipe carried inside this file
            except Exception as ex:
                try: os.remove(tmp_recipe)
                except Exception: pass
                note("could not write the Blender recipe to a temporary file: %s" % ex); return
        cmd = [blender, "-b", "--python-exit-code", "1", "--python", recipe, "--", fn + ".obj", out, fn + ".json", "--samples", str(RENDER_SAMPLES), "--engine", RENDER_ENGINE, "--size", RENDER_SIZE]
        note("Blender: starting ...", key='blender')
        def run():
            """the progress line ('... | Sample 64/128') is read from two sources, because Blender provides it differently by
            version: the recipe's render_stats handler writes it to out.progress (Blender passes it the line from 4.2 on, and
            nothing at all up to 4.1), and Blender prints the same line to stdout, flushed at each update (up to 4.5; 5.0 prints
            it only with --log-level info). The piped output is also kept for the error message."""
            import time as _time
            prog = out + ".progress"
            try:
                pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)   # UTF-8 whatever the locale: a byte it cannot decode must not stop the reader
                state['blender_proc'] = pr
                buf = []; passes = 0                                                # Freestyle renders the strokes as a second Cycles pass
                def reader():
                    for line in pr.stdout: buf.append(line.rstrip())
                threading.Thread(target=reader, daemon=True).start()
                last = None
                while pr.poll() is None:
                    _time.sleep(0.25)
                    try:
                        with open(prog, encoding='utf-8', errors='replace') as fh: txt = fh.read()
                    except Exception: txt = ""
                    if 'Finished' in txt and last and 'sample' in last: passes += 1
                    msm = next((mm for src in list(reversed(buf[-8:])) + [txt] for mm in [_re.search(r"Sample (\d+)/(\d+)", src)] if mm), None)
                    if msm:
                        k_, n_ = int(msm.group(1)), int(msm.group(2))
                        cur = "Blender: %s %3d %%  (sample %d of %d)" % ("line strokes" if passes else "rendering", 100*k_//max(1, n_), k_, n_)
                    elif any(ch.isdigit() for ch in txt):
                        tail_ = txt.strip().split("|")[-1].strip(); cur = "Blender: " + (tail_ if len(tail_) <= 60 else tail_[:60].rsplit(" ", 1)[0] + " ...")
                    else: cur = None
                    if cur and cur != last: state['pending_note'] = (cur, 'blender'); last = cur
                _time.sleep(0.3); tail = buf[-40:]
                if pr.returncode == 0 and os.path.exists(out): msg = "Blender: done, %s written" % out
                else:                                                               # the cause: the recipe's Python exception, else the first error or crash line
                    tb = [k for k, l in enumerate(tail) if l.startswith('Traceback')]
                    bad = [l for l in tail[tb[-1] + 1:] if l and not l[0].isspace()][:1] if tb else []
                    for kw in (('terminat', 'exception', 'segmentation fault'), ('error',), ('crash',)):   # the most specific line first ('crash' is also in a crash-log path)
                        bad = bad or [l for l in tail if any(s_ in l.lower() for s_ in kw) and 'strokes set empty' not in l]   # Freestyle's 'strokes set empty' is harmless
                    try: how = "crashed (%s" % signal.Signals(-pr.returncode).name if pr.returncode < 0 else "failed (exit %s" % pr.returncode
                    except Exception: how = "failed (exit %s" % pr.returncode
                    msg = "Blender %s, see the terminal): " % how + (bad[0] if bad else (tail[-1] if tail else "no message")); print("\n".join(tail))
                try: pr.stdout.close()
                except Exception: pass
            except Exception as ex: msg = "Blender: %s" % ex
            for f_ in filter(None, (prog, tmp_recipe)):                             # the progress file, and the temporary recipe (Blender is done with it)
                try: os.remove(f_)
                except Exception: pass
            state['pending_note'] = (msg, 'blender')
        threading.Thread(target=run, daemon=True).start()
    def note(msg, key=None):
        """a line appended to the report box; with a key, the line replaces the previous line of the same key (progress)."""
        lines = state['report_str'].rstrip().split("\n")
        marks = state.setdefault('note_keys', {})
        if key is not None and key in marks and marks[key] < len(lines) and lines[marks[key]].strip().startswith(marks.get(key + '_prefix', '\x00')):
            lines[marks[key]] = "  " + msg
        else:
            lines.append("  " + msg)
            if key is not None: marks[key] = len(lines) - 1; marks[key + '_prefix'] = msg.split(':')[0]
        state['report_str'] = "\n".join(lines)
        if key == 'blender': state['blender_line'] = msg
        report_text.set_text(report_string()); layout_panels(); fig.canvas.draw_idle()
    def poll_notes():
        msg = state.pop('pending_note', None)
        if msg: note(*msg) if isinstance(msg, tuple) else note(msg)
    state['poll_notes'] = poll_notes
    b_svg.on_clicked(save_svg); b_png.on_clicked(save_png); b_obj.on_clicked(save_obj_with_meta); b_seq.on_clicked(save_motion); b_rend.on_clicked(render_blender)
    fig._widgets = state['widgets'] = [sl, b_minus, b_plus, b_vert, b_face, b_edge, b_angl, b_hid, b_shad, b_rep, b_reset, b_dih, b_frz, b_par, b_col, *b_light.values(), *cdl['buttons'],
                                       b_svg, b_png, b_obj, b_seq, b_rend, b_info, b_x_rep, b_x_face]   # keep the widgets alive
    # Tab / shift+Tab while typing in a value box: submit it and move to the next / previous box of its group
    tab_groups = [[dlg[k_] for k_ in ('ang0', 'ang1', 'ang2', 'ang3', 'M', 'N', 'kap', 'nu') if k_ in dlg], [th_box, t_box, vbox['elev'], vbox['azim'], vbox['zoom']]]
    def on_tab(ev):
        if ev.key not in ('tab', 'shift+tab'): return
        for grp in tab_groups:
            for k_, tb_ in enumerate(grp):
                if getattr(tb_, 'capturekeystrokes', False):
                    d_ = 1 if ev.key == 'tab' else -1                                   # the next box that takes keys (the view boxes: not with the report hidden)
                    nxt = next((g_ for j_ in range(1, len(grp)) for g_ in [grp[(k_ + j_*d_) % len(grp)]] if not isinstance(g_, PanelBox) or not g_.ignore(ev)), None)
                    if nxt is not None and not (nxt.ax.get_visible() and nxt.get_active()): nxt = None
                    tb_.stop_typing()                                                   # submits the box (its on_submit runs)
                    if nxt is not None:
                        nxt.begin_typing(); nxt.cursor_index = len(nxt.text)
                        try: nxt._rendercursor()
                        except Exception: pass
                    fig.canvas.draw_idle(); return
    fig.canvas.mpl_connect('key_press_event', on_tab)
    if SHOW_SLIDER_CURVE:
        try: draw_curve()
        except Exception: report_bug("draw_curve")
    if STARTUP_ERROR[0] is not None:                                            # parameters of the file failed: show why, in the dialog
        set_dialog(True)
        show_popup("The parameters at the top of the file give no net",
                   STARTUP_ERROR[0].split("\nChange ")[0].replace("\n", " ").strip().rstrip(".") + ". The default net is shown; edit the values here and press Apply.")
        STARTUP_ERROR[0] = None
    spread = side_spread(vertex_face_angles(sub), m, n, BASE_KN)
    if spread > 1e6:                                                            # built, but beyond double precision: say so (the checks show it)
        note("warning: the sides of the central faces differ by a factor of %.0e;\n  the thin boundary faces may not keep their shape along the slider" % spread)
    if "--frames" in sys.argv:          # headless: save a few frames and exit (optional: --view ELEV,AZIM)
        view = command_view()                                                    # (checked by main() before the startup)
        if view: ax.view_init(elev=view[0], azim=view[1]); state['drawn'] = False
        for k, th_f in enumerate(np.linspace(th_hi_w, th_lo_w, 6)):                  # the slider's whole range, evenly in theta_1
            t = t_of_theta(float(th_f)); cfg = configuration(float(t))
            if cfg is None: continue
            state['t'] = float(t); take_config(cfg); state['info'] = (k == 1); state['unrolled'] = len(INFO_LINES) if k == 1 else 0
            state['labels'] = {'vertices': k == 0, 'faces': k == 0, 'edges': k == 0, 'angles': False}
            try:
                sl.eventson = False; sl.set_val(float(t)); sl.eventson = True; sync_t_box(); sync_theta(); b_info.set_on(k == 1)   # the sliders, their boxes and the (i) button follow the frame
            except Exception: pass
            redraw()
            fig.savefig("frame_%02d.png" % k, dpi=100)
        return
    # run the event loop as long as the main window exists: closing a secondary window (Dihedrals) must not end the program,
    # which plt.show() alone can do on some backends (macOS) when a window created during the loop is closed
    try:
        plt.show(block=False)
    except TypeError:
        plt.show()
    non_gui = matplotlib.get_backend().lower() in ('agg', 'pdf', 'ps', 'svg', 'pgf', 'cairo', 'template')   # (TkAgg, QtAgg, ... are GUI backends)
    if non_gui and '--frames' not in sys.argv:                                  # a Python without any GUI toolkit (e.g. a bare Linux install)
        print("matplotlib has no window backend here (backend: %s): install one, e.g. 'sudo apt install python3-tk' or 'pip install PyQt6',\n"
              "or run headless with: python %s --frames --view 30,-60" % (matplotlib.get_backend(), os.path.basename(__file__)))
    while plt.fignum_exists(fig.number) and not non_gui:
        try:
            fig.canvas.flush_events(); fig.canvas.start_event_loop(0.05)          # always on the main canvas (never on a closed one)
            if 'poll_notes' in state: state['poll_notes']()
        except Exception:
            if plt.fignum_exists(fig.number): report_bug("the event loop")         # (not when the window was closed under the call)
            break
    return state.get('rebuild', False)

# ============================================================================================
# The Blender recipe, carried inside this file: the Render button writes it to a temporary file and runs Blender on it
# (a render_net.py placed next to this script takes precedence, for tinkering). Same content as render_net.py.
# ============================================================================================
RENDER_RECIPE = r'''# render_net.py -- Blender recipe (headless) for an m x n net exported by gqs.py
#
#   Blender -b --python render_net.py -- net.obj out.png [net.json] [--samples N] [--engine cycles|eevee] [--size WxH]
#
# net.json is the side-car written by the tool next to the OBJ: the view (elevation, azimuth, focal length), the light mode,
# the size of the net, the colours of the picture theme and the Hidden-lines switch. Without it, a default view and the Paper
# colours are used.
#
# The scene: the net imported with z up and flat shading (planar faces, sharp creases); a matte clay in the theme's light tone;
# a cool ambient at the theme's shadow tone; one large warm key (following the light mode of the tool) and a cool-white rim
# light from behind, both lighting the net only; a separate, nearly vertical light that lights only the shadow catcher and is
# blocked by the net (the ground shadow, as the tool casts it); Freestyle strokes: the outline in the theme's ink, the interior
# creases in the softer ink, the occluded creases faint; composited over the theme's canvas colour.
#
# The look was calibrated on reference figures (face tones 0.79 - 0.96, cool shade / warm light, ~2 px cobalt
# creases, a soft compact shadow); the tints are derived from the theme's own lit and shaded tones, so every theme keeps its
# intent, boosted by TINT_BOOST (1 = exactly the tool's tints, 3 = the reference figures' stronger warm/cool split; the boost tapers off
# for themes whose light tone is itself coloured, so a lavender or blue theme is not turned cream).
import bpy, sys, os, json, math
from mathutils import Vector, Matrix

# ----------------------------------------------------------------------------------------------- arguments
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(argv) < 2:
    print("usage: Blender -b --python render_net.py -- net.obj out.png [net.json] [--samples N] [--engine cycles|eevee] [--size WxH]")
    sys.exit(1)
obj_path, out_path = argv[0], argv[1]
json_path = argv[2] if len(argv) > 2 and not argv[2].startswith("--") else None
def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default
samples = int(opt("--samples", "128"))
engine = opt("--engine", "cycles").lower()
width, height = (int(v) for v in opt("--size", "1920x1080").lower().split("x"))

meta = {}
if json_path and not os.path.exists(json_path): print("side-car %s not found: using the OBJ's own .json or the defaults" % json_path)
if json_path and os.path.exists(json_path):
    with open(json_path) as fh: meta = json.load(fh)
elif os.path.exists(os.path.splitext(obj_path)[0] + ".json"):
    with open(os.path.splitext(obj_path)[0] + ".json") as fh: meta = json.load(fh)

elev = float(meta.get("elev", 30.0)); azim = float(meta.get("azim", -60.0)); focal = meta.get("focal_length", 0.4)
light_mode = str(meta.get("light_frame", "studio")).lower()
colors = meta.get("colors", {})
def hex_rgb(h):
    """'#RRGGBB' or '#RGB' -> RGB triple in 0..1; anything else: white, with a note"""
    try:
        h = h.lstrip("#")
        if len(h) == 3: h = "".join(2*ch for ch in h)
        return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
    except Exception:
        print("colour %r not understood: using white" % (h,)); return (1.0, 1.0, 1.0)
def col(key, default):
    v = colors.get(key, default)
    if isinstance(v, str): return hex_rgb(v)
    return tuple(float(x) for x in v[:3])
def s2l(c):
    """sRGB -> scene-linear. The side-car carries the tool's sRGB values; every colour handed to Blender (materials, lights,
    world, line styles) is scene-linear, so it has to be converted, otherwise everything renders too light and too pale (the
    ink #2C4470 = (0.17, 0.27, 0.44) taken as linear displays as (0.45, 0.55, 0.69), a pale blue-grey). The numpy step at the
    end is the exception: it works on the byte pixels of the PNG, which are sRGB-encoded, and uses the values unconverted."""
    return tuple(v/12.92 if v <= 0.04045 else ((v + 0.055)/1.055)**2.4 for v in c)
LIGHT = col("LIGHT_COL", (0.980, 0.988, 0.988)); SHADOW = col("SHADOW_COL", (0.780, 0.784, 0.776))
EDGE = col("EDGE_COL", (0.173, 0.267, 0.439)); EDGE_SOFT = col("EDGE_SOFT", (0.353, 0.420, 0.486)); HIDDEN = col("HIDDEN_COL", (0.557, 0.612, 0.659))
WARM = col("WARM_BOUNCE", (0.965, 0.961, 0.914)); COOL = col("COOL_BOUNCE", (0.847, 0.890, 0.918))
CANVAS = colors.get("FACE_COLOR", "#FFFFFF")
canvas_rgb = hex_rgb(CANVAS) if isinstance(CANVAS, str) else tuple(float(x) for x in CANVAS[:3])
mix = float(colors.get("TINT_MIX", 0.35)); spec_k = float(colors.get("SPEC_K", 0.12))

# ---- the look: calibrated on reference figures; these are the knobs of a variant
TINT_BOOST = 3.0       # the theme's warm/cool tints times this: 1 = exactly the tool's tints, 3 = the reference figures' stronger split
TRANSLUCENCY = 0.0     # weight of a Translucent BSDF mixed in. The faces have no thickness, so such a lobe carries the key
                       # light straight through every face into the folds behind it and flattens the creases: keep it at 0
                       # (a small value, 0.05 - 0.10, gives a hint of vellum without losing the fold contrast).
SHEEN = 0.2            # Principled sheen: a soft brightening of faces seen at grazing angles (paper fibre)
KEY_DIST, KEY_SIZE = 4.0, 3.0   # the warm key: distance and width in net sizes (far and wide: soft gradients across the faces)
RIM_SHARE = 0.35       # a cool-white rim light from behind, above and right of the viewer (irradiance relative to the key's):
RIM_DIR = (0.45, 0.75, -0.5)    # a glow along the far edges of the net; direction in the screen basis (right, up, toward the viewer)
LINE_SIL = 1.25        # thickness of the outline (silhouette and border) relative to the interior creases
PEAK = 0.98            # the brightest face reaches this fraction of the theme's light tone (the specular adds the rest)

# ----------------------------------------------------------------------------------------------- scene
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# import the net: z up (the tool's convention), flat shading, no merging
if hasattr(bpy.ops.wm, "obj_import"):
    bpy.ops.wm.obj_import(filepath=obj_path, forward_axis='Y', up_axis='Z')
else:
    bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='Y', axis_up='Z')           # Blender < 4.0
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH' and len(o.data.polygons)]
if not meshes: sys.exit("no faces found in %s" % obj_path)
net = meshes[0]; net.name = "net"
for p in net.data.polygons: p.use_smooth = False
try: net.data.shade_flat()
except Exception: pass
def mark_all_edges(mesh):
    """Freestyle edge marks on every edge (a fold or a border of the net): the boolean edge attribute 'freestyle_edge'
    (Blender 4/5) and, where it exists, the legacy MeshEdge.use_freestyle_mark."""
    try:
        attr = mesh.attributes.get("freestyle_edge") or mesh.attributes.new("freestyle_edge", 'BOOLEAN', 'EDGE')
        attr.data.foreach_set("value", [True]*len(mesh.edges))
    except Exception as ex: print("edge-mark attribute:", ex)
    try:
        for e in mesh.edges: e.use_freestyle_mark = True
    except Exception: pass
mark_all_edges(net.data)

# bounding box and size
pts = [net.matrix_world @ Vector(v) for v in net.bound_box]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
cen = (lo + hi)/2; size = max(max(hi - lo), 1e-9)
floor_z = lo.z - 0.02*size
if light_mode == 'room':
    # the photo studio: the net stands on its base face F_kn (z = 0) on an invisible stand whose height the tool chose for the
    # whole flexion (side-car floor_z). Without a side-car: the base plane when nothing hangs below it, else 12 % under the lowest point.
    n_base = sum(1 for v in net.data.vertices if abs((net.matrix_world @ v.co).z) < 1e-4*size)
    if meta.get("floor_z") is not None: floor_z = min(float(meta["floor_z"]), lo.z - 0.02*size)
    elif n_base >= 3: floor_z = -0.003*size if lo.z >= -1e-4*size else lo.z - 0.12*size
    print("room: the floor at z = %.3f (lowest point of the net %.3f)" % (floor_z, lo.z))

# ----------------------------------------------------------------------------------------------- colours of the light
def tilt(c, hue_, k):                                                                   # borrow the hue, keep the value (as in the tool)
    m_c, m_h = sum(c)/3.0, sum(hue_)/3.0
    if m_h < 1e-6: return tuple(c)
    return tuple(max(0.0, min(1.0, c[i]*(1 - k) + m_c*(hue_[i]/m_h)*k)) for i in range(3))
def mean1(lc):                                                                          # a linear triple scaled to mean 1
    m_ = sum(lc)/3.0; return tuple(v/max(m_, 1e-6) for v in lc)
def chroma_of(h): return max(h) - min(h)
def set_chroma(h, chroma):
    """scale a mean-1 hue to the given chroma (max - min), keeping its direction; chroma 0 = white light"""
    cur = chroma_of(h)
    if cur < 1e-9 or chroma <= 0: return (1.0, 1.0, 1.0)
    return mean1(tuple(max(0.05, 1.0 + (v - 1.0)*chroma/cur) for v in h))
def tone_ratio(base, bounce, k):
    """how the tool tints a tone: tilt(base, bounce, k) relative to base, in linear light, mean 1"""
    t = tilt(base, bounce, k); lb, lt = s2l(base), s2l(t)
    return mean1(tuple(lt[i]/max(lb[i], 1e-6) for i in range(3)))
lin_L, lin_S = s2l(LIGHT), s2l(SHADOW)
shade_floor = min(max(sum(lin_S)/max(sum(lin_L), 1e-6), 0.0), 0.95)                   # the theme's shade floor relative to its light tone
f_world = 0.90*shade_floor                                                              # share of the ambient in the irradiance of a fully lit face
boost = 1.0 + (TINT_BOOST - 1.0)*max(0.0, 1.0 - chroma_of(mean1(lin_L))/0.06)          # the full boost for a neutral light tone, none for a coloured one
lit_ratio = tone_ratio(LIGHT, WARM, mix)                                                # the tool's tint of the lit faces (relative to LIGHT)
shade_ratio = tone_ratio(SHADOW, COOL, mix)                                             # ... and of the shaded faces (relative to SHADOW)
world_hue = set_chroma(shade_ratio, min(0.20, boost*chroma_of(shade_ratio)))            # the ambient carries the shade tint
# a lit face sees f_world of the ambient and f_key of the key: the key must carry the lit tint and undo the ambient's share
key_hue = mean1(tuple(max(0.05, lit_ratio[i] - f_world*world_hue[i]) for i in range(3)))
key_hue = set_chroma(key_hue, min(0.55, boost*chroma_of(key_hue)))
rim_hue = set_chroma(world_hue, 0.06)

# ----------------------------------------------------------------------------------------------- material
body = LIGHT                                                                            # the theme's light tone; the lights carry the tints
mat = bpy.data.materials.new("clay")
if hasattr(mat, "use_nodes"): mat.use_nodes = True
nt = mat.node_tree
for nd in list(nt.nodes): nt.nodes.remove(nd)
out_node = nt.nodes.new('ShaderNodeOutputMaterial')
bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
bsdf.inputs["Base Color"].default_value = (*s2l(body), 1.0)
bsdf.inputs["Roughness"].default_value = 0.5
for name, val in (("Specular IOR Level", 0.1 + 1.5*spec_k), ("Specular", 0.1 + 1.5*spec_k), ("Sheen Weight", SHEEN), ("Sheen", SHEEN), ("Coat Weight", 0.0)):
    if name in bsdf.inputs: bsdf.inputs[name].default_value = val
if TRANSLUCENCY > 0:
    trans = nt.nodes.new('ShaderNodeBsdfTranslucent'); trans.inputs["Color"].default_value = (*s2l(tilt(LIGHT, WARM, 0.5)), 1.0)
    mixsh = nt.nodes.new('ShaderNodeMixShader'); mixsh.inputs["Fac"].default_value = TRANSLUCENCY
    nt.links.new(bsdf.outputs["BSDF"], mixsh.inputs[1]); nt.links.new(trans.outputs["BSDF"], mixsh.inputs[2])
    nt.links.new(mixsh.outputs["Shader"], out_node.inputs["Surface"])
else:
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
net.data.materials.clear(); net.data.materials.append(mat)

# shadow catcher under the net
bpy.ops.mesh.primitive_plane_add(size=12*size, location=(cen.x, cen.y, floor_z))
floor = bpy.context.active_object; floor.name = "floor"
try: floor.is_shadow_catcher = True
except Exception as ex: print("shadow catcher unavailable:", ex)
floor_mat = bpy.data.materials.new("floor")
if hasattr(floor_mat, "use_nodes"): floor_mat.use_nodes = True
fb = next((nd for nd in floor_mat.node_tree.nodes if nd.type == 'BSDF_PRINCIPLED'), None)
if fb is not None: fb.inputs["Base Color"].default_value = (*s2l(canvas_rgb), 1.0)
floor.data.materials.append(floor_mat)
def own_collection(obj, name):
    """move obj into a collection of its own (light linking works on collections)"""
    coll = bpy.data.collections.new(name); scene.collection.children.link(coll)
    for c in list(obj.users_collection): c.objects.unlink(obj)
    coll.objects.link(obj); return coll
net_coll, floor_coll = own_collection(net, "net"), own_collection(floor, "floor")

# ----------------------------------------------------------------------------------------------- camera: mplot3d's orbit
# The screen basis of mplot3d for any elevation (it flips its vertical axis past the pole, and the two flips cancel): the
# screen-right vector is always (-sin a, cos a, 0) and screen-up is w x u. The perspective strength of mplot3d corresponds to
# an eye at 18.2 * focal_length half-widths of the axes box (measured).
e, a = math.radians(elev), math.radians(azim)
elev_n = (elev + 180.0) % 360.0 - 180.0                                                # mplot3d's _norm_angle, for the messages
w = Vector((math.cos(e)*math.cos(a), math.cos(e)*math.sin(a), math.sin(e)))          # from the centre towards the viewer
u = Vector((-math.sin(a), math.cos(a), 0.0))                                           # screen right
up = w.cross(u)                                                                         # screen up
dist = 18.2*float(focal)*0.55*size if focal else 40.0*size                              # orthographic in the tool: a distant camera
cam_data = bpy.data.cameras.new("camera"); cam_data.clip_start = 1e-3*size; cam_data.clip_end = 1e3*size
cam = bpy.data.objects.new("camera", cam_data); scene.collection.objects.link(cam); scene.camera = cam
cam.location = cen + dist*w
cam.matrix_world = Matrix(((u.x, up.x, w.x, cam.location.x), (u.y, up.y, w.y, cam.location.y), (u.z, up.z, w.z, cam.location.z), (0, 0, 0, 1)))
below = cam.location.z < floor_z + 0.01*size                                           # the view from underneath: the floor would hide the net

# ----------------------------------------------------------------------------------------------- light directions (the tool's modes)
if light_mode == 'world':                                                              # the sun at noon, a little towards the viewer
    key_dir = (Vector((0, 0, 1)) + 0.35*w - 0.25*u).normalized()
elif light_mode == 'eye':                                                              # a flashlight in the viewer's hand
    key_dir = (w + 0.35*up - 0.20*u).normalized()
elif light_mode == 'top':                                                              # the sun at the top of the screen
    key_dir = (up + 0.35*w - 0.20*u).normalized()
elif light_mode == 'room':                                                             # the studio's lamps, fixed in the room: set up for the
    e0, a0 = (math.radians(float(x)) for x in meta.get("room_lamp_view", (30.0, -60.0)))   # view room_lamp_view, then left there
    w0 = Vector((math.cos(e0)*math.cos(a0), math.cos(e0)*math.sin(a0), math.sin(e0))); u0 = Vector((-math.sin(a0), math.cos(a0), 0.0)); up0 = w0.cross(u0)
    key_dir = (-0.55*u0 + 0.75*up0 + 0.40*w0).normalized()
else:                                                                                  # studio: high, upper left, in front of the viewer
    key_dir = (-0.55*u + 0.75*up + 0.40*w).normalized()                                # calibrated on the reference figures
if light_mode in ('studio', 'top') and not below and key_dir.z < 0.6:                 # looking down, 'up on the screen' is nearly horizontal and the
    h = Vector((key_dir.x, key_dir.y, 0.0))                                            # key would sink to the horizon: keep it at least ~37 degrees up
    h = h.normalized()*0.8 if h.length > 1e-6 else Vector((-0.8, 0.0, 0.0))
    key_dir = Vector((h.x, h.y, 0.6)).normalized()
rim_dir = (RIM_DIR[0]*u + RIM_DIR[1]*up + RIM_DIR[2]*w).normalized()
if light_mode == 'room': rim_dir = (RIM_DIR[0]*u0 + RIM_DIR[1]*up0 + RIM_DIR[2]*w0).normalized()
if key_dir.dot(w) < 0 and light_mode != 'room':                                              # the key behind the visible side (the world sun seen from
    key_dir = Vector((key_dir.x, key_dir.y, -key_dir.z))                               # underneath): mirror it below, the tool lights the visible side
if light_mode == 'world':   shadow_dir = Vector((0, 0, 1))                                # the tool's shadow lights (Lsh): vertical footprint,
elif light_mode == 'eye':                                                                # the flashlight itself (shadow behind the net), kept above the floor
    shadow_dir = key_dir if key_dir.z >= 0.15 else Vector((key_dir.x, key_dir.y, 0.0)).normalized()*0.99 + Vector((0, 0, 0.15))
    shadow_dir = shadow_dir.normalized()
elif light_mode == 'top':                                                                # straight below the net on the screen, as on the tool's screen floor
    h_ = Vector((up.x, up.y, 0.0)); shadow_dir = (h_.normalized()*0.35 + Vector((0, 0, 0.94))).normalized() if h_.length > 1e-6 else Vector((0, 0, 1))
elif light_mode == 'room':  shadow_dir = (Vector((0, 0, 1)) - 0.45*u0 - 0.25*w0).normalized()  # the studio's soft lamp: high, upper left of the reference view, a little behind
else:                       shadow_dir = (Vector((0, 0, 1)) - 0.18*u - 0.10*w).normalized()   # studio: nearly vertical, a little towards the viewer
# the shadow catcher: the plane P.catch_n = catch_h with the net on its + side, normally the floor (catch_n up, catch_h = floor_z). Seen
# from below, the floor lies between the camera and the net, and a shadow catcher there cuts the net out of the picture (tested). Where
# the tool casts the shadow behind the net, the catcher goes there: a flashlight under the net ('eye') onto a ceiling above it, 'top'
# onto its floor in the viewer's frame (seen from 30 degrees above). 'studio' and 'world' have it on the floor in front of the net (a
# translucent glass plane in the tool) and 'room' has none: no ground shadow from below there (catch = False).
catch_n, catch_h, catch = Vector((0, 0, 1)), floor_z, not below
if below and light_mode == 'eye' and key_dir.z < 0:                                     # the flashlight under the net: a ceiling, as in the tool
    shadow_dir = (key_dir if key_dir.z <= -0.15 else Vector((key_dir.x, key_dir.y, 0.0)).normalized()*0.99 - Vector((0, 0, 0.15))).normalized()
    catch_n, catch_h, catch = Vector((0, 0, -1)), -(hi.z + 0.02*size), "on a ceiling above the net"
elif below and light_mode == 'top':                                                     # the tool's screen floor (light along its normal)
    catch_n = (math.cos(math.radians(30))*up + math.sin(math.radians(30))*w).normalized(); shadow_dir = catch_n
    catch_h, catch = min((net.matrix_world @ v.co).dot(catch_n) for v in net.data.vertices) - 0.06*size, "on the floor of the viewer's frame"
if below and catch:                                                                     # the catcher turned to face the net, through the centre's projection
    e2_ = catch_n.cross(u); o_ = cen + (catch_h - cen.dot(catch_n))*catch_n
    floor.matrix_world = Matrix(((u.x, e2_.x, catch_n.x, o_.x), (u.y, e2_.y, catch_n.y, o_.y), (u.z, e2_.z, catch_n.z, o_.z), (0, 0, 0, 1)))
    print("camera below the ground plane (elevation %.1f): the shadow %s" % (elev_n, catch))

# ----------------------------------------------------------------------------------------------- framing
# fit the projected extent of the net and of its shadow on the floor (as the tool's crop does), centred, with a margin
def _frame_pts():
    for vtx in net.data.vertices:
        P = net.matrix_world @ vtx.co; yield P
        if catch and not engine.startswith("eevee") and shadow_dir.dot(catch_n) > 0.05 and P.dot(catch_n) > catch_h:   # (Eevee draws no ground shadow)
            d_ = P.dot(catch_n) - catch_h; S = P - shadow_dir*(0.7*d_/shadow_dir.dot(catch_n)); r_ = 0.08*d_   # most of its shadow on the floor (the far penumbra may fade out of the frame)
            for dx in (-r_, r_):
                yield S + dx*u; yield S + dx*w
xs, ys = [], []
for P in _frame_pts():
    d = P - cam.location; z = max(-(d.dot(w)), 1e-6*size)
    xs.append(d.dot(u)/z); ys.append(d.dot(up)/z)
hx, hy = (max(xs) - min(xs))/2, (max(ys) - min(ys))/2; cx, cy = (max(xs) + min(xs))/2, (max(ys) + min(ys))/2
aspect = width/float(height)
half = 1.08*max(hx, hy*aspect, 1e-6) if aspect >= 1 else 1.08*max(hy, hx/aspect, 1e-6)   # half-extent of the frame along the sensor-fit axis
cam_data.sensor_fit = 'HORIZONTAL' if aspect >= 1 else 'VERTICAL'
cam_data.angle = 2*math.atan(half)
cam_data.shift_x = cx/(2*half); cam_data.shift_y = cy/(2*half)                          # centre the frame on the net + shadow (units of the fit axis)

# ----------------------------------------------------------------------------------------------- lights
# exposure: the 'Standard' view transform has no highlight roll-off (everything above 1.0 clips to flat white). An area light of
# power P at the distance d gives the irradiance E = P/(pi d^2) on a surface facing it, and a matt surface of albedo rho has the
# radiance rho E/pi: with E_tot = pi*PEAK a fully lit face renders exactly PEAK times the theme's light tone.
E_tot = math.pi*PEAK
E_world = f_world*E_tot; E_key = max(E_tot - E_world, 0.05*E_tot)
def area_light(name, location, color, energy, size_):
    ld = bpy.data.lights.new(name, 'AREA'); ld.energy = energy; ld.color = color; ld.size = size_
    lo_ = bpy.data.objects.new(name, ld); scene.collection.objects.link(lo_); lo_.location = location
    dd = (cen - lo_.location).normalized(); up_ = Vector((0, 0, 1)) if abs(dd.z) < 0.99 else Vector((0, 1, 0))
    r = dd.cross(up_).normalized(); t = r.cross(dd)
    lo_.matrix_world = Matrix(((r.x, t.x, -dd.x, lo_.location.x), (r.y, t.y, -dd.y, lo_.location.y), (r.z, t.z, -dd.z, lo_.location.z), (0, 0, 0, 1)))
    return lo_
def link_lights(light, receivers, blockers):
    """Blender 4+: the light reaches only the receivers and is blocked only by the blockers; older versions ignore it"""
    try: light.light_linking.receiver_collection = receivers; light.light_linking.blocker_collection = blockers
    except Exception as ex: print("light linking unavailable (%s): the key also casts the shadow" % (ex,))
d_key = KEY_DIST*size
key = area_light("key", cen + d_key*key_dir, key_hue, E_key*math.pi*d_key**2, KEY_SIZE*size)   # large warm key: soft gradients
link_lights(key, net_coll, net_coll)                                                    # the key shades the net only (no shadow on the floor)
if RIM_SHARE > 0 and not below:
    d_rim = 5.0*size
    rim = area_light("rim", cen + d_rim*rim_dir, rim_hue, RIM_SHARE*E_key*math.pi*d_rim**2, 2.5*size)   # cool-white glow from behind
    link_lights(rim, net_coll, net_coll)
if not catch:
    print("camera below the ground plane (elevation %.1f): no floor, no ground shadow (the floor would lie between the camera and the net)" % elev_n)
    bpy.data.objects.remove(floor, do_unlink=True)
else:
    d_sh = 6.0*size
    shadow_light = area_light("shadow", cen + d_sh*shadow_dir, (1.0, 1.0, 1.0), 0.5*E_tot*math.pi*d_sh**2, (0.8 if light_mode == 'room' else 1.5)*size)   # soft, compact footprint
    link_lights(shadow_light, floor_coll, net_coll)                                     # lights the floor only, blocked by the net: the shadow
scene.world = bpy.data.worlds.new("world")
try:
    if hasattr(scene.world, "use_nodes"): scene.world.use_nodes = True
    bg = next(nd for nd in scene.world.node_tree.nodes if nd.type == 'BACKGROUND')
    bg.inputs["Color"].default_value = (*world_hue, 1.0); bg.inputs["Strength"].default_value = E_world/math.pi   # the ambient, part of the exposure
except Exception as ex:
    print("world background:", ex)
    try: scene.world.color = world_hue
    except Exception: pass

# ----------------------------------------------------------------------------------------------- Freestyle
# the outline (silhouette and border) in the ink, the interior creases thinner in the softer ink, the occluded creases faint
try:
    scene.render.use_freestyle = True
    scene.render.line_thickness_mode = 'ABSOLUTE'; scene.render.line_thickness = 1.0
    vl = scene.view_layers[0]; fs = vl.freestyle_settings
    fs.crease_angle = math.radians(178.0)                                             # every real fold counts as a crease
    while len(fs.linesets): fs.linesets.remove(fs.linesets[0])
    def lineset(name, visibility, color, thickness, alpha, kinds, exclude=(), combine='OR'):
        ls = fs.linesets.new(name)
        ls.select_by_visibility = True; ls.visibility = visibility
        try: ls.select_by_collection = True; ls.collection = net_coll                  # never the floor plane's border
        except Exception as ex: print("lineset collection filter unavailable:", ex)
        ls.select_by_edge_types = True
        for k in ('silhouette', 'border', 'crease', 'edge_mark'): setattr(ls, 'select_' + k, k in kinds or k in exclude)
        for k in exclude:                                                              # an exclude flag only acts on a selected type
            try: setattr(ls, 'exclude_' + k, True)
            except Exception: pass
        try: ls.edge_type_combination = combine
        except Exception as ex: print("lineset edge-type combination unavailable:", ex)
        st = ls.linestyle; st.color = s2l(color); st.thickness = thickness; st.alpha = alpha
        try: st.caps = 'ROUND'
        except Exception: pass
        return ls
    thick = max(1.5*width/1000.0, 2.0)                                                  # ink scales with the resolution (2.9 at 1920), never under 2 px
    lum_L = 0.2126*LIGHT[0] + 0.7152*LIGHT[1] + 0.0722*LIGHT[2]
    if lum_L < 0.8: thick *= 1.3                                                        # darker themes: the mid-tone faces take contrast from the lines
    if light_mode == 'room':
        # the ink rule of the reference figures: the silhouette always in the ink; borders and creases in the ink on the top side of
        # the sheet (the side its base face turns up on the table), in the soft ink on faces seen from underneath (Freestyle face marks)
        M3 = net.matrix_world.to_3x3(); polys = net.data.polygons
        signed = [((M3 @ p.normal).dot(w), p.area) for p in polys]
        base_polys = [p for p in polys if all(abs((net.matrix_world @ net.data.vertices[i].co).z) < 1e-4*size for i in p.vertices)]
        if base_polys:                                                                        # the sheet's top side: where the base face's normal points
            flip = (M3 @ max(base_polys, key=lambda p: p.area).normal).z < 0
        else:                                                                                 # no base face: the side that shows most in this view
            front_area = sum(a*s for s, a in signed if s > 0); back_area = sum(-a*s for s, a in signed if s < 0); flip = back_area > front_area
        marks = [bool((s < 0) != flip) for s, a in signed]                                # True: seen from the back
        try:
            fattr = net.data.attributes.get("freestyle_face") or net.data.attributes.new("freestyle_face", 'BOOLEAN', 'FACE')
            fattr.data.foreach_set("value", marks)
        except Exception as ex: print("face-mark attribute:", ex)
        try:
            for p, mk in zip(polys, marks): p.use_freestyle_mark = mk
        except Exception: pass
        lineset("silhouette", 'VISIBLE', EDGE, 0.75*thick*LINE_SIL, 0.9, kinds=('silhouette',))
        for side, colr, neg in (("front", EDGE, 'EXCLUSIVE'), ("back", EDGE_SOFT, 'INCLUSIVE')):
            for name, thk, kinds, excl in (("border", 0.75*thick*LINE_SIL, ('border',), ('silhouette',)),
                                           ("creases", 0.8*thick, ('edge_mark',), ('silhouette', 'border'))):
                ls_ = lineset(name + "_" + side, 'VISIBLE', colr, thk, 0.9, kinds=kinds, exclude=excl, combine='AND')
                ls_.select_by_face_marks = True; ls_.face_mark_negation = neg; ls_.face_mark_condition = 'ONE'
    else:
        lineset("outline", 'VISIBLE', EDGE, 0.75*thick*LINE_SIL, 0.9, kinds=('silhouette', 'border'))
        lineset("creases", 'VISIBLE', EDGE_SOFT, 0.8*thick, 0.9, kinds=('edge_mark',), exclude=('silhouette', 'border'), combine='AND')
    if meta.get("hidden", True):                                                       # the tool's Hidden button (True when the side-car does not say)
        lineset("hidden", 'HIDDEN', HIDDEN, 0.6*thick, 0.5, kinds=('silhouette', 'border', 'crease', 'edge_mark'))
except Exception as ex:
    print("Freestyle setup failed (%s): rendering without line strokes" % (ex,)); scene.render.use_freestyle = False

# ----------------------------------------------------------------------------------------------- colour management, render settings
# the plain 'Standard' transform keeps the theme's hex colours (AgX / Filmic pull pastels towards grey)
try:
    scene.view_settings.view_transform = 'Standard'; scene.view_settings.look = 'None'; scene.view_settings.exposure = 0.0
    scene.display_settings.display_device = 'sRGB'
except Exception as ex: print("view transform:", ex)
scene.render.resolution_x, scene.render.resolution_y = width, height
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
if engine.startswith("eevee"):
    for name in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
        try: scene.render.engine = name; break
        except Exception: continue
    try: scene.eevee.taa_render_samples = max(16, samples)
    except Exception: pass
    if catch:
        floor.hide_render = True                                                       # the shadow catcher and light linking are Cycles-only
        print("Eevee: no ground shadow (the shadow catcher is a Cycles feature)")
else:
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    try: scene.cycles.use_denoising = True; scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    except Exception: pass
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for dev in ('METAL', 'OPTIX', 'CUDA', 'HIP', 'ONEAPI'):
            try:
                prefs.compute_device_type = dev; prefs.get_devices()
                if any(d.type != 'CPU' for d in prefs.devices):
                    for d in prefs.devices: d.use = (d.type != 'CPU')                     # GPU only: a CPU device in the mix is the slowest work
                    scene.cycles.device = 'GPU'; print("Cycles on", dev)
                    try: scene.cycles.denoising_use_gpu = True
                    except Exception: pass
                    break
            except Exception: continue
    except Exception as ex: print("GPU setup skipped:", ex)

# progress: the render statistics (e.g. 'Sample 64/128') are written to out.progress for the tool at each update. From Blender
# 4.2 on the handler is called with that line as a string; up to 4.1 it is called with nothing at all, and then nothing is
# written here: the tool reads the same line from Blender's own stdout instead (printed up to 4.5; 5.x prints it only with
# --log-level info, but there the handler gives the string). Freestyle renders its strokes as a second pass: the counter restarts.
progress_path = os.path.abspath(out_path) + ".progress"
def _stats(*args):
    txt = next((a_ for a_ in args if isinstance(a_, str)), None)
    if not txt: return
    try:
        with open(progress_path, "w", encoding="utf-8") as fh: fh.write(txt)
    except Exception: pass
try: bpy.app.handlers.render_stats.append(_stats)
except Exception as ex: print("no render_stats handler:", ex)
print("Blender", bpy.app.version_string)

# render, then flatten the transparent image onto the theme's canvas colour with numpy (Blender's Python ships numpy):
# this avoids the compositor, whose API changed between Blender 3, 4 and 5
scene.render.image_settings.file_format = 'PNG'; scene.render.image_settings.color_mode = 'RGBA'
scene.render.filepath = os.path.abspath(out_path)
bpy.ops.render.render(write_still=True)
try:
    import numpy as np
    img = bpy.data.images.load(scene.render.filepath)
    W_, H_ = img.size[0], img.size[1]
    try:
        px = np.empty(W_*H_*4, dtype=np.float32); img.pixels.foreach_get(px); px = px.reshape(H_, W_, 4)
    except Exception:
        px = np.array(img.pixels[:], dtype=np.float32).reshape(H_, W_, 4)
    a_ = px[..., 3:4]
    # Pixels of the shadow catcher come out black with alpha = the fraction of the light blocked (Cycles composes the combined
    # image of a transparent film from the matte, which is empty there, and 1 - shadow_catcher as its alpha), so they are the
    # pixels with a black RGB and an alpha strictly between 0 and 1. Very dark ink strokes over the transparent film look the
    # same, so pixels within a stroke's width of the opaque net are left alone. They become the ground ink, its opacity scaled so that the
    # deepest shadow has the theme's opacity: the fraction itself depends on how much of the light comes from the ambient, which
    # casts no shadow. These colours are sRGB, like the byte pixels they are mixed with.
    ground_a = float(colors.get("GROUND_A", 0.14)); ground = np.array(col("GROUND_COL", (0.28, 0.32, 0.36)), dtype=np.float32)
    opaque = a_[..., 0] >= 0.999
    near = opaque.copy()
    band = max(3, int(math.ceil(0.75*max(1.5*width/1000.0, 2.0)*LINE_SIL/2)) + 2)             # half the outline width plus 2 px
    for _ in range(band):
        n2 = near.copy()
        n2[1:, :] |= near[:-1, :]; n2[:-1, :] |= near[1:, :]; n2[:, 1:] |= near[:, :-1]; n2[:, :-1] |= near[:, 1:]
        near = n2
    shadow = (px[..., :3].max(axis=-1, keepdims=True) < 0.04) & (a_ > 0.002) & (a_ < 0.999) & (~near[..., None])
    amax = float(np.percentile(a_[shadow], 99)) if shadow.any() else 1.0
    a2 = np.where(shadow, np.minimum(a_*(ground_a/max(amax, 1e-6)), ground_a), a_)
    # the shadow fades out towards the frame border (the frame is fitted to the net and most of the footprint; the far penumbra
    # may reach the edge, and a soft shadow must never end in a hard cut)
    yy, xx = np.mgrid[0:H_, 0:W_]
    edge = np.minimum(np.minimum(xx, W_ - 1 - xx)/(0.10*W_), np.minimum(yy, H_ - 1 - yy)/(0.10*H_)).clip(0, 1)
    edge = edge*edge*(3 - 2*edge)
    a2 = np.where(shadow, a2*edge[..., None], a2)
    rgb = np.where(shadow, ground, px[..., :3])
    flat = rgb*a2 + np.array(canvas_rgb, dtype=np.float32)*(1 - a2)
    px[..., :3] = flat; px[..., 3] = 1.0
    try: img.pixels.foreach_set(px.ravel())
    except Exception: img.pixels = px.ravel().tolist()
    img.filepath_raw = scene.render.filepath; img.file_format = 'PNG'; img.save()
    try: os.remove(progress_path)
    except Exception: pass
except Exception as ex:
    print("flattening onto the canvas colour failed (%s); the PNG is left with a transparent background" % (ex,))
print("rendered", scene.render.filepath)
'''

if __name__ == "__main__":
    while main(): pass
