# Interactive visualization of flexible m x n GQS-nets assembled from 3 x 3 GQS-nets
# (companion to "Kokotsakis polyhedra and GQS-nets"; generalizes the 3 x 3 visualization of
#  A. Nurmatov, D. L. Michels, "Quasi-symmetric nets: a constructive approach to the equimodular elliptic type of
#  Kokotsakis polyhedra", Computer-Aided Design 199 (2026) 104102, referred to below as "the CAD paper").
#
# Input: base flat angles (alpha, beta, gamma, delta) in degrees at the vertex A_1 of the base 3 x 3 block,
#        the net size m x n, the position of the base block, and the relation type ('a'..'d', see RELATION_TEXT) of each row of blocks.
# The flat angles of the whole net are assembled from the base block by the relations between the flat angles of
# adjacent blocks; the dihedral angles are given by the explicit flexion formulas (one flexion per block, the sign
# patterns chosen consistently); the net is built in
# space with planar convex faces by the construction of the appendix of the CAD paper, face by face, and drawn with a
# slider for the flexion parameter t = cot(theta_1/2) of the base block.
#
# Dihedral angles are oriented interior dihedral angles: pi when adjacent faces are coplanar, positive when the face
# bends towards the side of the normal of the central face (Bricard variables cot(theta/2)).
# Requires numpy and matplotlib only.

import math, itertools, sys, json, re, textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'          # LaTeX-like indices and symbols in all labels
from matplotlib.widgets import Slider, TextBox
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

PI = math.pi

# ============================================================================================
# 1. PARAMETERS
# ============================================================================================
BASE_ANGLES_DEG = (60.0, 80.0, 120.0, 60.0)   # (alpha_1, beta_1, gamma_1, delta_1) of the base block
M_FACES, N_FACES = 5, 5                        # the net has M_FACES x N_FACES faces F_ij, i < M_FACES, j < N_FACES
BASE_KN = (2, 2)                               # central face F_{kappa nu} of the base block (1 <= kappa <= M-2, 1 <= nu <= N-2)
ROW_SYSTEMS = {1: 'c', 2: 'a', 3: 'b'}         # relation type ('a'..'d', see RELATION_TEXT) of the row of 3 x 3 blocks with central faces F_{. nu},
                                               # nu = 1 .. N_FACES-2 (rows not listed get the type of the base row)
BASE_SIGNS = (1, -1, 1, -1)                    # signs (e1,e2,e3,e4) of the flexion of the base block
E0 = 1                                         # branch of the base flexion
BASE_EDGE = 1.0                                # length of the edge V_{kappa+1,nu} V_{kappa nu} of the base central face
SHOW_SHADOW = True
LIGHT_FRAME = 'studio'                         # 'eye': a flashlight in the viewer's hand (shadow behind the net: floor from above, ceiling from below);
                                               # 'top': the sun at the top of the screen (shadow always straight below the net on the screen);
                                               # 'studio': the original setup (shading light attached to the viewer, shadow on the room's floor);
                                               # 'world': the sun at noon fixed in space (vertical footprint on the floor). Buttons Eye | Top | Studio | World                             # soft ground shadow (button Shadow)
FOCAL_LENGTH = 0.4                             # perspective strength (matplotlib focal length; smaller = stronger; None = orthographic)
NSUB = 8                                       # each face is split into NSUB x NSUB pieces for the painter's depth sorting
LIGHT_FACES = 150                              # nets with more faces are drawn in the light mode (plain depth sorting, no hidden lines, no intersection test)
STRIP_CHOICES = 8                              # alternatives per boundary face in the backtracking construction of the boundary strips
LIVE_PEN_FACES = 49                            # nets with at most this many faces (7 x 7) get the self-intersection test also while the slider is dragged;
                                               # larger nets (up to LIGHT_FACES) get it when the slider is released
SAVE_FULL_WINDOW = False                       # buttons SVG/PNG: False = the picture only (cropped, no panels/controls), True = the whole window
PNG_DPI = 300                                  # resolution of the saved PNG

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
    def squeeze(mt): return mt.group(0).replace(" ", "")
    out = []
    for para in text.split("\n"):
        para = re.sub(r"\$[^$]*\$", squeeze, para)
        out += textwrap.wrap(para, width=width, break_long_words=False, break_on_hyphens=False) or [""]
    return "\n".join(out)

def angle_problem(angs_deg, tol_deg=1e-6):
    """None if the base angles satisfy the hypotheses; otherwise a sentence naming the violated one:
    (1) no signed sum alpha +- beta +- gamma +- delta is 0 modulo 360 degrees (elliptic type);
    (2) the barred numbers sigma - alpha, ..., sigma - delta lie in (0, 180) degrees, sigma = (alpha + beta + gamma + delta)/2."""
    al, be, ga, de = angs_deg
    names = [r"$\alpha_1$", r"$\beta_1$", r"$\gamma_1$", r"$\delta_1$"]
    for sb, sg, sd in itertools.product((1, -1), repeat=3):
        v = (al + sb*be + sg*ga + sd*de) % 360.0
        if min(v, 360.0 - v) < tol_deg:
            expr = r"$\alpha_1 %s \beta_1 %s \gamma_1 %s \delta_1$" % ("+" if sb > 0 else "-", "+" if sg > 0 else "-", "+" if sd > 0 else "-")
            return expr + " = 0 (mod 360°): the block is not of elliptic type; change one of the angles"
    sig = (al + be + ga + de)/2
    bars = [sig - al, sig - be, sig - ga, sig - de]
    bad = [(nm, b) for nm, b in zip([r"$\bar\alpha_1$", r"$\bar\beta_1$", r"$\bar\gamma_1$", r"$\bar\delta_1$"], bars) if not (tol_deg < b < 180 - tol_deg)]
    if bad:
        return ("the barred numbers must lie in (0°, 180°); here " + ", ".join("%s = %.1f°" % (nm, b) for nm, b in bad)
                + r"  ($\sigma_1 = %.1f°$, $\bar\alpha_1 = \sigma_1 - \alpha_1$, ...)" % sig)
    return None

def check_parameters(m, n, base_kn, row_systems, verbose=True):
    """clear messages for the usual mistakes: net too small, base block out of range, rows without a relation type."""
    assert m >= 3 and n >= 3, "M_FACES and N_FACES must be at least 3 (the net must contain a 3 x 3 block)"
    k0, n0 = base_kn
    assert 1 <= k0 <= m - 2 and 1 <= n0 <= n - 2, "BASE_KN = (kappa, nu) must satisfy 1 <= kappa <= M_FACES-2 and 1 <= nu <= N_FACES-2"
    assert n0 in row_systems, "ROW_SYSTEMS must contain the row of the base block"
    rows = dict(row_systems)
    missing = [nu for nu in range(1, n - 1) if nu not in rows]
    for nu in missing: rows[nu] = rows[n0]
    if missing and verbose: print("ROW_SYSTEMS: rows %s not given, using the relation type '%s' of the base row for them" % (missing, rows[n0]))
    for nu, r in rows.items(): assert r in RELATION, "ROW_SYSTEMS: unknown relation type %r (use 'a', 'b', 'c' or 'd')" % (r,)
    return rows

def assemble_net(base_deg, m, n, base_kn, row_systems):
    """quadruples of all 3 x 3 blocks (kappa, nu) of the m x n net, propagated from the base block by the relations between the flat angles of adjacent blocks."""
    q1 = tuple(math.radians(x) for x in base_deg)
    k0, n0 = base_kn
    sub = {base_kn: block_from_vertex1(q1, row_systems[n0])}
    rev = lambda q: (q[3], q[2], q[1], q[0])          # (delta, gamma, beta, alpha)
    sw  = lambda q: (q[1], q[0], q[3], q[2])          # (beta, alpha, delta, gamma)
    # vertical propagation along the column kappa = k0
    for nu in range(n0 + 1, n - 1):
        prev = sub[(k0, nu - 1)]
        sub[(k0, nu)] = block_from_vertex1(rev(prev[4]), row_systems[nu])
        assert np.allclose(sub[(k0, nu)][2], rev(prev[3]))
    for nu in range(n0 - 1, 0, -1):
        nxt = sub[(k0, nu + 1)]
        q4 = rev(nxt[1]); q1 = RELATION[row_systems[nu]](q4)
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

def bricard_P(v, X, Y):
    """P_i(X, Y) of Bricard's equations for the vertex data v."""
    ab, bb, gb, db = v['bars']; be, s = v['be'], v['sig']
    return (math.sin(db)*math.sin(db-be)*X*X*Y*Y + math.sin(ab)*math.sin(ab-be)*X*X + math.sin(gb)*math.sin(gb-be)*Y*Y
            - 2*math.sin(v['al'])*math.sin(v['ga'])*X*Y + math.sin(s)*math.sin(bb))

def sgn(a): return 1.0 if a > 0 else (-1.0 if a < 0 else 0.0)
def rsqrt(a, tol=1e-9):
    """real square root of a radicand that is nonnegative up to rounding (all radicands of the flexion formulas are >= 0 on the admissible set)."""
    if a < -tol*(1 + abs(a)): raise ValueError("negative radicand")
    return math.sqrt(max(a, 0.0))

def thmain_formulas(x1, x3, y1, y2, z1, z2, z3, u, eps, e, e0, t):
    """the explicit flexion formulas: (cot theta_1/2, cot theta_2/2, cot theta_3/2, cot theta_4/2) = (Z, W_2, U, W_1) at the parameter
    t = Z, for the sign pattern e = (e1, e2, e3, e4) and the branch e0 = +-1; eps = (eps_1, eps_2, eps_3, eps_4)."""
    e1, e2, e3, e4 = e
    D = (x1*t*t - 1)*(1 - u*x1*t*t)
    W2 = eps[1]*(t*rsqrt(u*x1*y2*(1 + z2)*(1 + u*z2)) + e0*e2*rsqrt(u*y2*z2*D))/(y2*u*(z2 + x1*t*t))
    W1 = eps[0]*(t*rsqrt(u*x1*y1*(1 + z1)*(1 + u*z1)) - e0*e1*rsqrt(u*y1*z1*D))/(y1*u*(z1 + x1*t*t))
    if abs(u*z2*z3 - 1) > 1e-9:                                                                    # case (a)
        den = u*z2*z3 - 1
        H1 = (abs(z2)*rsqrt(z3*(1 + z3)*(1 + u*z3)) + e2*e3*abs(z3)*rsqrt(z2*(1 + z2)*(1 + u*z2)))/den
        H2 = (abs(z2*z3)*rsqrt((1 + u*z2)*(1 + u*z3)) + e2*e3*rsqrt(z2*z3*(1 + z2)*(1 + z3)))/den
        H3 = (u*rsqrt(z2*z3*(1 + z2)*(1 + z3))*sgn(z2*z3) + e2*e3*rsqrt((1 + u*z2)*(1 + u*z3)))/den
    elif e2 == -e3:                                                                                 # case (b)
        rt = 2*rsqrt(z2*z3*(1 + z2)*(1 + z3))
        H1 = z2*z3*(z3 - z2)/rt; H2 = z2*z3*(z2 + z3 + 2)/rt; H3 = (z2 + z3 + 2*z2*z3)*sgn(z2*z3)/rt
    else:                                                                                           # case (c)
        U = eps[1]*eps[2]*rsqrt(z2*z3/(x1*x3))*sgn(y2)/t
        return t, W2, U, W1
    U = eps[1]*eps[2]*sgn(z2)*rsqrt(z2*z3/(x1*x3))*(abs(x1)*H2*H3*t + e0*e2*H1*rsqrt(x1*D))/(z2*z3 + u*x1*H1*H1*t*t)
    return t, W2, U, W1

def admissible_range(x, u):
    """the admissible set I(x) = {t : x D(x, t) >= 0} as a list of intervals (t > 0 half; the set is symmetric in t)."""
    if x > 0 and u > 0: return [(1/math.sqrt(x), 1/math.sqrt(u*x))] if u < 1 else [(1/math.sqrt(x), math.inf)]
    if x > 0 and u < 0: return [(1/math.sqrt(x), math.inf)]
    if x < 0 and u < 0: return [(0.0, 1/math.sqrt(u*x))]
    return [(0.0, math.inf)]                    # x < 0, u > 0: x D >= 0 for every t

class Block:
    """a 3 x 3 block (Kokotsakis polyhedron): vertex data, elliptic data and the explicit flexions."""
    def __init__(self, k, n, quads):
        self.k, self.n = k, n
        self.A, self.B, self.C = block_labels(k, n)
        self.q = quads
        self.v = {i: vdata(*quads[i]) for i in range(1, 5)}
        M = self.v[1]['M']
        assert max(abs(self.v[i]['M'] - M) for i in range(1, 5)) < 1e-9, "not equimodular"
        assert abs(self.v[1]['r'] - self.v[2]['r']) < 1e-9 and abs(self.v[3]['r'] - self.v[4]['r']) < 1e-9, "amplitudes at common vertices do not match"
        assert abs(self.v[1]['s'] - self.v[4]['s']) < 1e-9 and abs(self.v[2]['s'] - self.v[3]['s']) < 1e-9, "amplitudes at common vertices do not match"
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
        self._witnesses = None

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

    def in_lattice(self, z, tol=1e-9):
        """z in Lambda = {4K m + 2iK' n} (M < 1) or {4K m + (2K + 2iK') n} (M > 1)."""
        if self.M < 1: a, b = z.real/(4*self.K), z.imag/(2*self.Kp)
        else: b = z.imag/(2*self.Kp); a = (z.real - 2*self.K*b)/(4*self.K)
        return abs(a - round(a)) < tol and abs(b - round(b)) < tol

    def fourth_sign(self, e1, e2, e3):
        """e_4 from the sign condition e_1 t_1 + e_2 t_2 + e_3 t_3 + e_4 t_4 in Lambda; 0 if neither sign works."""
        for e4 in (1, -1):
            if self.in_lattice(e1*self.t[1] + e2*self.t[2] + e3*self.t[3] + e4*self.t[4]): return e4
        return 0

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

    def cots(self, e, e0, t, shift=0):
        """{i: cot(theta_i/2)} from the flexion formulas with the parameter t = cot(theta_{1+shift}/2) and the branch e0."""
        x1, x3, y1, y2, z1, z2, z3, eps, idx = self.shifted_data(shift)
        es = tuple(e[i - 1] for i in idx)
        vals = thmain_formulas(x1, x3, y1, y2, z1, z2, z3, self.u, eps, es, e0, t)
        return {idx[j]: vals[j] for j in range(4)}

    def angles_from_cots(self, c): return {i: 2*math.atan(1/c[i]) for i in range(1, 5)}

    def admissible(self, shift=0):
        """I(x_1), I(y_2), I(x_3), I(y_4) for shift = 0, 1, 2, 3."""
        x1 = self.shifted_data(shift)[0]
        return admissible_range(x1, self.u)

    def bricard_residual(self, c):
        Z, W2, U, W1 = c[1], c[2], c[3], c[4]
        return max(abs(bricard_P(self.v[1], Z, W1)), abs(bricard_P(self.v[2], Z, W2)), abs(bricard_P(self.v[3], U, W2)), abs(bricard_P(self.v[4], U, W1)))

    def witnesses(self):
        """sign patterns e = (1, e_2, e_3, e_4) satisfying the sign condition on the phase shifts. The flexion formulas involve
        e_1, e_2, e_3 only (e_4 is the sign fixed by the condition); a triple (1, e_2, e_3) belongs to an admissible pattern if
        and only if the formulas solve Bricard's equations, which is tested at three admissible values of t; e_4 is then read
        off from the condition with the phase shifts."""
        if self._witnesses is None:
            lo, hi = self.admissible()[0]; hi = min(hi, lo + 5.0 if lo > 0 else 5.0)
            tests = [lo + f*(hi - lo) for f in (0.23, 0.41, 0.67)]
            out = []
            for e2, e3 in itertools.product([1, -1], repeat=2):
                e = (1, e2, e3, 1); ok = True
                for t in tests:
                    try:
                        for e0 in (1, -1):
                            if self.bricard_residual(self.cots(e, e0, t)) > 1e-8: ok = False
                    except (ValueError, ZeroDivisionError): ok = False
                if ok: out.append((1, e2, e3, self.fourth_sign(1, e2, e3)))
            self._witnesses = out
        return self._witnesses

    def base_config(self, e, t, e0):
        """dihedral angles of the base configuration at the parameter t = cot(theta_1/2), branch e0."""
        try: c = self.cots(e, e0, t)
        except (ValueError, ZeroDivisionError): return None
        return self.angles_from_cots(c)

    def configs_through_pair(self, e, i, X, Y, tol=1e-7):
        """dihedral angles theta_1..theta_4 of the points of the family e whose central angles at the vertex A_i have the
        cotangents (X, Y) = (cot theta_{i-1}/2, cot theta_i/2), theta_0 = theta_4: the parameter is theta_{i-1} (cyclic shift
        so that it is the first angle), and the branch e0 is the one whose adjacent angle theta_i has the cotangent Y."""
        ia = 4 if i == 1 else i - 1
        out = []
        for e0 in (1, -1):                                                          # parameter theta_{i-1}, branch fixed by theta_i
            try: c = self.cots(e, e0, X, shift=ia - 1)
            except (ValueError, ZeroDivisionError): continue
            if abs(c[i] - Y) < tol*(1 + abs(Y)):
                out.append(self.angles_from_cots(c))
        if not out:                                                                 # the other way round: parameter theta_i, branch fixed by theta_{i-1}
            for e0 in (1, -1):                                                      # (needed when theta_{i-1} sits at the boundary of its admissible set)
                try: c = self.cots(e, e0, Y, shift=i - 1)
                except (ValueError, ZeroDivisionError): continue
                if abs(c[ia] - X) < tol*(1 + abs(X)):
                    out.append(self.angles_from_cots(c))
        return out

# ============================================================================================
# 4. GEOMETRY OF ONE 3 x 3 SUBNET (to read off the non-central dihedral angles at its vertices)
# ============================================================================================
def unit(v): return v/np.linalg.norm(v)

def convex_quad(d1, d2, d3, a1=1.0):
    """planar convex quadrilateral A_1 A_2 A_3 A_4 with angles d_i and |A_2 A_1| = a1 (appendix of the CAD paper, Section B)."""
    d4 = 2*PI - d1 - d2 - d3
    s23, s12 = d2 + d3, d1 + d2
    if s23 >= PI: a2 = a1 if s12 >= PI else 0.8*a1*math.sin(d1)/math.sin(s12)
    elif s12 >= PI: a2 = 1.2*a1*math.sin(s23)/math.sin(d3)
    else: a2 = a1*(math.sin(s23)/math.sin(d3) + math.sin(d1)/math.sin(s12))/2
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
    B = {1: A[1] + math.cos(al[1])*(-e[1]) + math.sin(al[1])*d[1], 2: A[2] + math.cos(al[2])*e[1] + math.sin(al[2])*d[1],
         3: A[3] + math.cos(al[3])*(-e[3]) + math.sin(al[3])*d[3], 4: A[4] + math.cos(al[4])*e[3] + math.sin(al[4])*d[3]}
    C = {2: A[2] + math.cos(ga[2])*(-e[2]) + math.sin(ga[2])*d[2], 3: A[3] + math.cos(ga[3])*e[2] + math.sin(ga[3])*d[2],
         1: A[1] + math.cos(ga[1])*e[4] + math.sin(ga[1])*d[4], 4: A[4] + math.cos(ga[4])*(-e[4]) + math.sin(ga[4])*d[4]}
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

def dihedral(S, i, pts):
    """oriented interior dihedral angle theta_i of block S read from 3D points (None if points are missing)."""
    A, B, C = S.A, S.B, S.C
    n0 = face_normal([A[1], A[2], A[3], A[4]], pts)
    orders = {1: [B[2], A[2], A[1], B[1]], 2: [A[3], A[2], C[2], C[3]], 3: [A[4], A[3], B[3], B[4]], 4: [C[1], A[1], A[4], C[4]]}
    ni = face_normal(orders[i], pts)
    if n0 is None or ni is None: return None
    nxt = A[i+1] if i < 4 else A[1]
    d = n0[0]*ni[0] + n0[1]*ni[1] + n0[2]*ni[2]
    bend = math.acos(max(-1.0, min(1.0, d)))
    e = (pts[A[i]][0] - pts[nxt][0], pts[A[i]][1] - pts[nxt][1], pts[A[i]][2] - pts[nxt][2])
    c = _cross(n0, e)                                  # det(ni, n0, e) = ni . (n0 x e)
    det = ni[0]*c[0] + ni[1]*c[1] + ni[2]*c[2]
    return (1 if det > 0 else -1)*(PI - bend)

# ============================================================================================
# 5. SYNCHRONIZATION OF THE SUBNETS
# ============================================================================================
def compatible(Sa, tha, ptsa, Sb, thb, ptsb, tol=1e-6):
    for i in range(1, 5):
        d = dihedral(Sb, i, ptsa)
        if d is not None and abs(d - thb[i]) > tol: return False
        d = dihedral(Sa, i, ptsb)
        if d is not None and abs(d - tha[i]) > tol: return False
    return True

def candidates(S, ref_pts, families):
    out = []
    for i in range(1, 5):
        ia = 4 if i == 1 else i - 1
        da, db = dihedral(S, ia, ref_pts), dihedral(S, i, ref_pts)
        if da is None or db is None: continue
        for e in families:
            for th in S.configs_through_pair(e, i, 1/math.tan(da/2), 1/math.tan(db/2)):
                pts, clo = block_points(S, th)
                if clo > 1e-7: continue
                if not any(max(abs(th[j] - th2[j]) for j in range(1, 5)) < 1e-7 for (_, th2, _) in out):
                    out.append((e, th, pts))
    return out

def propagation_order(subs, base_kn):
    order, seen, queue = [], {base_kn}, [base_kn]
    while queue:
        k, n = queue.pop(0)
        for nb in [(k, n-1), (k, n+1), (k-1, n), (k+1, n), (k-1, n-1), (k+1, n-1), (k-1, n+1), (k+1, n+1)]:
            if nb in subs and nb not in seen: seen.add(nb); queue.append(nb); order.append(nb)
    return order

class _Budget(Exception): pass

def synchronize(subs, base_kn, base_signs, e0, t, assignment=None, max_nodes=None):
    """configurations (theta dicts) of all blocks, consistent on common faces; searches the families if assignment is None.
    max_nodes bounds the backtracking search (used by the parameters dialog; an exhausted budget counts as 'no')."""
    B = subs[base_kn]; nodes = [0]
    thB = B.base_config(base_signs, t, e0)
    if thB is None: return None, None
    ptsB, clo = block_points(B, thB)
    if clo > 1e-7: return None, None
    assigned = {base_kn: (base_signs, thB, ptsB)}
    order = propagation_order(subs, base_kn)
    def rec(idx):
        if idx == len(order): return True
        nodes[0] += 1
        if max_nodes is not None and nodes[0] > max_nodes: raise _Budget()
        kn = order[idx]; S = subs[kn]
        fams = [assignment[kn]] if assignment else S.witnesses()
        near = sorted((o for o in assigned if abs(o[0] - kn[0]) <= 2 and abs(o[1] - kn[1]) <= 2),
                      key=lambda o: abs(o[0] - kn[0]) + abs(o[1] - kn[1]))                  # only blocks that can share faces, nearest first
        cands = []
        for ref in near:                                                                    # a reference sharing a vertex with this block
            cands = candidates(S, assigned[ref][2], fams)
            if cands: break
        for (e, th, pts) in cands:
            if all(compatible(S, th, pts, subs[o], assigned[o][1], assigned[o][2]) for o in near):
                assigned[kn] = (e, th, pts)
                if rec(idx + 1): return True
                del assigned[kn]
        return False
    try:
        if not rec(0): return None, None
    except _Budget: return None, None
    return {kn: assigned[kn][1] for kn in assigned}, {kn: assigned[kn][0] for kn in assigned}

# ============================================================================================
# 6. THE NET IN SPACE WITH CONVEX FACES (the construction of the appendix of the CAD paper, generalized face by face)
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
        cr = [(pts[(k+1)%4]-pts[k])[0]*(pts[(k+2)%4]-pts[(k+1)%4])[1] - (pts[(k+1)%4]-pts[k])[1]*(pts[(k+2)%4]-pts[(k+1)%4])[0] for k in range(4)]
        return all(c > 1e-9 for c in cr) or all(c < -1e-9 for c in cr)
    # The construction is done as originally (base face by the construction of the CAD paper's appendix, heuristic lengths for the faces attached
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
        phR = ang[R][f] if inner(R) else (2*PI - phP - phQ)/2
        phS = 2*PI - phP - phQ - phR
        assert 0 < phR < PI and 0 < phS < PI
        L = np.linalg.norm(pos[Q] - pos[P])
        lP = (free.setdefault(('len', f, P, R), booster*Lref) if record else free[('len', f, P, R)]) if not inner(R) else (Lin[frozenset((P, R))] if Lin is not None else inner_boost[0]*(lambda d1, d2, d3: (L if d2+d3 >= PI and d1+d2 >= PI else 0.8*L*math.sin(d1)/math.sin(d1+d2) if d2+d3 >= PI else 1.2*L*math.sin(d2+d3)/math.sin(d3) if d1+d2 >= PI else L*(math.sin(d2+d3)/math.sin(d3) + math.sin(d1)/math.sin(d1+d2))/2))(phQ, phP, phR))
        lQ = (lP*math.sin(phR) - L*math.sin(phP + phR))/math.sin(phS)
        assert lP > 0 and lQ > 0
        pos[R] = pos[P] + lP*(math.cos(phP)*u + math.sin(phP)*dnew)
        pos[S] = pos[Q] + lQ*(-math.cos(phQ)*u + math.sin(phQ)*dnew)
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
                if strip_mode[0] == 'legacy':                                          # as originally: the first convex length of the list
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
    built = None; last_error = None
    if record:
        for sc in [1.0, 0.8, 1.25, 0.65, 1.55, 0.5, 1.9, 0.4, 2.4, 0.3]:                # the original construction, several scales
            inner_boost[0] = sc; pos.clear(); place_base_face()
            try: built = build_inner(); break
            except AssertionError as ex: last_error = ex
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
    def strip(faces):
        f0 = faces[0]; cs = corners(f0)
        P, Q = next((cs[j], cs[(j+1) % 4]) for j in range(4) if cs[j] in pos and cs[(j+1) % 4] in pos)
        oldf = next(g for g in built if P in corners(g) and Q in corners(g))
        def undo(g):
            for v in corners(g):
                if not inner(v) and v in pos and not any(v in corners(h) for h in built_faces): del pos[v]
            if record:
                for key in [k for k in free if k[1] == g]: del free[key]
        built_faces = set()
        def place(k):                                                            # depth-first over the faces of the strip
            if k == len(faces): return True
            g = faces[k]
            for choice in range(STRIP_CHOICES if (record and strip_mode[0] == 'balanced') else 1):
                try: attach3(g, choice)
                except AssertionError:
                    undo(g); return False if not record else False
                built_faces.add(g)
                if place(k + 1): return True
                built_faces.discard(g); undo(g)
            return False
        modes = ['legacy', 'balanced'] if record else [free.get(('strip_mode', f0), 'legacy')]
        for mode in modes:
            strip_mode[0] = mode
            for booster in ((0.8, 0.6, 1.0, 0.45, 1.3, 0.3, 1.7, 0.2, 2.5, 0.12) if record else (None,)):
                try:
                    if record: attach2(f0, P, Q, oldf, booster)
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

def self_intersections(pos, faces_all, corners):
    """pairs of faces with no common vertex that penetrate each other."""
    pairs = []
    P = {f: np.array([pos[x] for x in corners(f)]) for f in faces_all}
    lo = {f: P[f].min(axis=0) for f in faces_all}; hi = {f: P[f].max(axis=0) for f in faces_all}
    for i, f in enumerate(faces_all):
        for g in faces_all[i+1:]:
            if (lo[f] > hi[g]).any() or (lo[g] > hi[f]).any(): continue           # bounding boxes do not meet
            if set(corners(f)) & set(corners(g)): continue
            if faces_intersect(P[f], P[g]) or faces_intersect(P[g], P[f]): pairs.append((f, g))
    return pairs

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

def layered_pieces(M, sf, polys, cols, max_split=2):
    """occlusion layer of every piece (the number of faces in front of it), with the pieces that straddle an occlusion boundary
    (9 sample points hidden by different numbers of faces) split into 4 until the counts agree or max_split is reached.
    Returns the final pieces, their colours and their layers (pieces of a higher layer are drawn earlier)."""
    out_p, out_c, out_l = [], [], []
    Pp, cc, depth = np.array(polys), np.array(cols), 0
    while len(Pp):
        c = Pp.mean(axis=1, keepdims=True)
        smp = np.concatenate([c, (Pp + c)/2, 0.9*Pp + 0.1*c], axis=1).reshape(-1, 3)
        cnt = occlusion_counts(M, sf, smp).reshape(len(Pp), 9)
        uniform = (cnt == cnt[:, :1]).all(axis=1)
        done = uniform | (depth >= max_split)
        out_p += list(Pp[done]); out_c += list(cc[done]); out_l += list(np.minimum(cnt[done].max(axis=1), 3))
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
    3D pieces at draw time. Pieces hidden by more faces (higher layer) are drawn earlier."""
    def __init__(self, verts, layer=0, **kwargs):
        self.layer, self.verts3d = layer, np.asarray(verts, dtype=float)
        super().__init__(np.zeros((0, 4, 2)), **kwargs)
    def do_3d_projection(self, *args, **kwargs):
        from mpl_toolkits.mplot3d import proj3d
        V = self.verts3d.reshape(-1, 3)
        tx, ty, tz = proj3d.proj_transform(V[:, 0], V[:, 1], V[:, 2], self.axes.M)
        self.set_verts(np.stack([tx, ty], axis=1).reshape(self.verts3d.shape[0], 4, 2), True)
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
    'Paper':     dict(COLOR_DEFAULTS),                      # the measured palette
    'Classic':   theme([0.985, 0.985, 0.975], [0.760, 0.790, 0.850], [0.760, 0.790, 0.850], [0.985, 0.985, 0.975], [0.760, 0.790, 0.850], 0.00, 0.12,
                       (0.160, 0.270, 0.460), (0.160, 0.270, 0.460), (0.450, 0.550, 0.720), (0.28, 0.31, 0.40), 0.17, '#FFFFFF'),   # the first look of the tool
    'Porcelain': theme([0.980, 0.988, 0.988], [0.745, 0.776, 0.800], [0.651, 0.671, 0.694], [0.957, 0.949, 0.890], [0.839, 0.886, 0.918], 0.28, 0.12,
                       (0.118, 0.208, 0.349), (0.404, 0.427, 0.463), (0.275, 0.376, 0.541), (0.40, 0.42, 0.45), 0.10, '#FFFFFF'),
    'Vellum':    theme([0.988, 0.992, 0.992], [0.902, 0.910, 0.914], [0.839, 0.851, 0.859], [0.973, 0.969, 0.929], [0.902, 0.933, 0.949], 0.45, 0.12,
                       (0.141, 0.251, 0.420), (0.541, 0.580, 0.635), (0.431, 0.486, 0.576), (0.42, 0.45, 0.50), 0.08, '#FFFFFF'),
    'Chalk':     theme([0.972, 0.969, 0.961], [0.800, 0.792, 0.776], [0.714, 0.706, 0.686], [0.965, 0.949, 0.910], [0.878, 0.886, 0.886], 0.22, 0.00,
                       (0.227, 0.243, 0.271), (0.549, 0.557, 0.565), (0.416, 0.431, 0.463), (0.38, 0.38, 0.38), 0.09, '#FCFCFB'),
    'Quartz':    theme([0.976, 0.973, 0.988], [0.745, 0.737, 0.784], [0.655, 0.647, 0.706], [0.957, 0.941, 0.918], [0.855, 0.859, 0.918], 0.34, 0.12,
                       (0.200, 0.188, 0.369), (0.435, 0.420, 0.502), (0.322, 0.306, 0.478), (0.40, 0.39, 0.47), 0.11, '#FDFCFF'),
    'Blueprint': theme([0.969, 0.980, 0.992], [0.749, 0.788, 0.831], [0.659, 0.706, 0.765], [0.949, 0.957, 0.949], [0.796, 0.863, 0.918], 0.40, 0.12,
                       (0.114, 0.243, 0.561), (0.420, 0.486, 0.576), (0.290, 0.435, 0.647), (0.34, 0.40, 0.50), 0.12, '#FBFCFE'),
    'Celadon':   theme([0.965, 0.980, 0.969], [0.753, 0.800, 0.765], [0.667, 0.722, 0.686], [0.945, 0.953, 0.894], [0.827, 0.886, 0.878], 0.40, 0.12,
                       (0.122, 0.310, 0.290), (0.431, 0.518, 0.502), (0.263, 0.439, 0.416), (0.35, 0.45, 0.43), 0.12, '#FCFDFB'),
    'Ivory':     theme([0.992, 0.984, 0.957], [0.812, 0.780, 0.722], [0.729, 0.694, 0.627], [0.957, 0.922, 0.839], [0.875, 0.878, 0.863], 0.42, 0.12,
                       (0.353, 0.275, 0.196), (0.604, 0.549, 0.471), (0.541, 0.471, 0.384), (0.45, 0.41, 0.35), 0.14, '#FFFDF8'),
    'Copper':    theme([0.871, 0.729, 0.612], [0.565, 0.396, 0.318], [0.478, 0.325, 0.259], [0.906, 0.780, 0.639], [0.686, 0.588, 0.561], 0.38, 0.30,
                       (0.243, 0.137, 0.090), (0.541, 0.384, 0.314), (0.420, 0.267, 0.200), (0.40, 0.28, 0.22), 0.18, '#FAF6F3'),
    'Zinc':      theme([0.808, 0.827, 0.847], [0.478, 0.506, 0.541], [0.404, 0.431, 0.471], [0.820, 0.808, 0.769], [0.690, 0.745, 0.796], 0.35, 0.34,
                       (0.122, 0.157, 0.200), (0.361, 0.400, 0.447), (0.243, 0.282, 0.333), (0.28, 0.31, 0.35), 0.20, '#F4F6F7'),
    'Graphite':  theme([0.541, 0.580, 0.620], [0.227, 0.251, 0.282], [0.184, 0.204, 0.231], [0.604, 0.576, 0.518], [0.494, 0.549, 0.612], 0.40, 0.12,
                       (0.784, 0.831, 0.894), (0.424, 0.463, 0.514), (0.306, 0.357, 0.420), (0.02, 0.03, 0.04), 0.45, '#14171C', ink='#C8D4E4'),
}

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

def edge_ink(e, pos, w, faces_all, corners):
    """navy on the silhouette and the creases facing the camera, soft grey on the others."""
    def front(f):
        c = [pos[v] for v in corners(f)]
        return np.dot(unit(np.cross(c[1] - c[0], c[3] - c[0])), w) > 0
    fs = [f for f in faces_all if set(e) <= set(corners(f))]
    return EDGE_COL if len(fs) < 2 or front(fs[0]) != front(fs[1]) else EDGE_SOFT

def shaded_pieces(pos, faces_all, corners, up, w, L, cen, size, tints, nsub):
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
        if np.dot(n, w) < 0: n = -n                                                   # normal of the side we see
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

def draw_shadow_frame(ax, pos, faces_all, corners, e1, e2, nrm, size, ncell=80):
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
    levels = np.linspace(0.05, 1.0, 12)[:-1]
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


def draw_shadow(ax, pos, faces_all, corners, L, size, ncell=80):
    """returns the extents (x0, x1, y0, y1, z0) of the drawn shadow, or None."""
    """soft shadow of the net cast along the light L onto the horizontal plane z = z0 on the far side of the net from the light:
    the floor below the net when the light comes from above, the ceiling above it when the light comes from below (a flashlight
    held by a viewer under the net). Filled contour bands of a blurred mask (nested bands of growing opacity)."""
    if abs(L[2]) < 0.2: return None
    allP = np.array(list(pos.values()))
    z0 = allP[:, 2].min() - 0.015*size if L[2] > 0 else allP[:, 2].max() + 0.015*size
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
    levels = np.linspace(0.05, 1.0, 12)[:-1]                      # nested regions {img >= level}, each a little darker
    a = 1 - (1 - GROUND_A)**(1/len(levels))                      # GROUND_A = opacity at the centre of the shadow
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
    def __init__(self, fig, rect, label, toggle=False, on=False, fontsize=9, rounding=0.16, italic=False):
        self.fig, self.toggle, self.on, self.hover, self.callbacks = fig, toggle, on, False, []
        self.ax = fig.add_axes(rect); self.ax.set_axis_off(); self.ax.set_navigate(False)
        w_in, h_in = rect[2]*fig.get_figwidth(), rect[3]*fig.get_figheight()
        rounding = min(rounding, 0.45*h_in/w_in)                   # the corner radius must stay below half the height (wide buttons)
        self.patch = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=%.3f" % rounding, mutation_aspect=w_in/h_in,
                                    transform=self.ax.transAxes, clip_on=False, linewidth=0.9)
        self.ax.add_patch(self.patch)
        self.text = self.ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=fontsize, transform=self.ax.transAxes,
                                 fontstyle='italic' if italic else 'normal')
        self._style()
        fig.canvas.mpl_connect('button_release_event', self._release); fig.canvas.mpl_connect('motion_notify_event', self._motion)
    def _style(self):
        fc, ec, tc = self.COLORS[('hover_' if self.hover else '') + ('on' if self.on else 'off')]
        self.patch.set_facecolor(fc); self.patch.set_edgecolor(ec); self.text.set_color(tc)
        self.text.set_fontweight('semibold' if self.on else 'normal')
    def _motion(self, ev):
        if not self.ax.get_visible(): return
        h = ev.inaxes is self.ax
        if h != self.hover: self.hover = h; self._style(); self.fig.canvas.draw_idle()
    def _release(self, ev):
        if ev.inaxes is not self.ax or not self.ax.get_visible(): return
        if self.toggle: self.on = not self.on
        self._style()
        for cb in self.callbacks: cb(ev)
        self.fig.canvas.draw_idle()
    def on_clicked(self, cb): self.callbacks.append(cb)
    def set_on(self, on): self.on = on; self._style()

def make_slider(ax_sl, lo, hi, t0):
    kw = dict(orientation='vertical', valinit=t0, valfmt='%.3f', color='#8fb6e3')
    try: sl = Slider(ax_sl, 'flexion parameter $t$', lo, hi, initcolor='none', track_color='#e3eaf3',
                     handle_style={'facecolor': 'white', 'edgecolor': '#3f6fae', 'size': 11}, **kw)
    except TypeError: sl = Slider(ax_sl, 'flexion parameter $t$', lo, hi, **kw)
    sl.label.set_fontsize(8.5); sl.label.set_color('#2c4a74'); sl.valtext.set_fontsize(9); sl.valtext.set_color('#2c4a74')
    return sl

def set_projection(ax):
    if FOCAL_LENGTH is None: ax.set_proj_type('ortho'); return
    try: ax.set_proj_type('persp', focal_length=FOCAL_LENGTH)
    except TypeError: ax.set_proj_type('persp')

# ============================================================================================
# 8. MAIN: assemble, synchronize, draw with a slider
# ============================================================================================
def admissible_t_range(B):
    """the part of I(x_1) used by the slider (positive half; capped when unbounded)."""
    lo, hi = B.admissible()[0]
    if lo == 0.0: lo = 0.03*hi if hi < math.inf else 0.05
    if hi == math.inf: hi = 4*lo
    return lo + 0.005*(hi - lo), hi - 0.005*(hi - lo)

DEFAULTS = dict(BASE_ANGLES_DEG=(60.0, 80.0, 120.0, 60.0), M_FACES=5, N_FACES=5, BASE_KN=(2, 2), ROW_SYSTEMS={1: 'c', 2: 'a', 3: 'b'}, BASE_SIGNS=(1, -1, 1, -1), E0=1)   # the built-in example
STARTUP_ERROR = [None]

def main():
    global BASE_SIGNS, E0, BASE_ANGLES_DEG, M_FACES, N_FACES, BASE_KN, ROW_SYSTEMS
    try:
        return _main()
    except SystemExit as ex:                                                    # the parameters at the top of the file give no net:
        msg = str(ex).strip()                                                   # start with the built-in defaults and show the message in the dialog
        if STARTUP_ERROR[0] is not None or (M_FACES, N_FACES, BASE_KN, tuple(BASE_ANGLES_DEG)) == (DEFAULTS['M_FACES'], DEFAULTS['N_FACES'], DEFAULTS['BASE_KN'], tuple(DEFAULTS['BASE_ANGLES_DEG'])) and ROW_SYSTEMS == DEFAULTS['ROW_SYSTEMS']:
            raise
        print(msg + "\nStarting with the default net instead; the message is shown in the Params dialog.")
        STARTUP_ERROR[0] = msg
        BASE_ANGLES_DEG, M_FACES, N_FACES, BASE_KN = DEFAULTS['BASE_ANGLES_DEG'], DEFAULTS['M_FACES'], DEFAULTS['N_FACES'], DEFAULTS['BASE_KN']
        ROW_SYSTEMS, BASE_SIGNS, E0 = dict(DEFAULTS['ROW_SYSTEMS']), DEFAULTS['BASE_SIGNS'], DEFAULTS['E0']
        return _main()

def _main():
    global BASE_SIGNS, E0
    m, n = M_FACES, N_FACES
    rows = check_parameters(m, n, BASE_KN, ROW_SYSTEMS); ROW_SYSTEMS.update(rows)
    sub = assemble_net(BASE_ANGLES_DEG, m, n, BASE_KN, rows)
    ang = vertex_face_angles(sub)
    subs = {kn: Block(kn[0], kn[1], q) for kn, q in sub.items()}
    B = subs[BASE_KN]
    lo, hi = admissible_t_range(B)
    t0 = lo + 0.15*(hi - lo)
    thetas, assignment = synchronize(subs, BASE_KN, BASE_SIGNS, E0, t0)
    if thetas is None:                                                          # the given base pattern / branch does not extend: try the others
        for e0_try in (E0, -E0):
            for e_try in B.witnesses():
                if tuple(e_try) == tuple(BASE_SIGNS) and e0_try == E0: continue
                thetas, assignment = synchronize(subs, BASE_KN, e_try, e0_try, t0)
                if thetas is not None:
                    print("BASE_SIGNS %s with e0 = %+d do not extend to a flexible net with these row types; using %s with e0 = %+d instead" % (BASE_SIGNS, E0, e_try, e0_try))
                    BASE_SIGNS, E0 = tuple(e_try), e0_try; break
            if thetas is not None: break
    if thetas is None:
        raise SystemExit("\nWith the row types %s (bottom row first) no admissible sign pattern of the base block extends to a flexible "
                         "%d x %d net.\nWhich combinations of row types work depends on the angles and on the position of the base block "
                         "(for the angles %s, for example, nothing can be placed above a row of type 'b' or 'd').\n"
                         "Change ROW_SYSTEMS, BASE_KN or the angles and start again; the Params dialog shows which combinations work."
                         % ("".join(rows[nu] for nu in range(1, n - 1)), m, n, BASE_ANGLES_DEG))
    cache = {}; free_lengths = {}
    def configuration(t):
        key = round(t, 6)
        if key not in cache:
            th, _ = synchronize(subs, BASE_KN, BASE_SIGNS, E0, t, assignment)
            if th is None: cache[key] = None
            else:
                try:
                    pos, worst, plan, fr = build_net(ang, th, m, n, BASE_KN, BASE_EDGE, free_lengths if free_lengths else None)
                    if not free_lengths: free_lengths.update(fr)
                    cache[key] = (pos, worst, plan, th)
                except AssertionError as ex: cache[key] = None; state_fail['reason'] = str(ex)   # construction failed at this t: skipped
        return cache[key]
    state_fail = {}
    first = configuration(t0)
    if first is None:                                                           # try other admissible values of t before giving up
        for f_ in (0.05, 0.25, 0.35, 0.5, 0.65, 0.8, 0.92):
            first = configuration(lo + f_*(hi - lo))
            if first is not None: t0 = lo + f_*(hi - lo); break
    if first is None:
        raise SystemExit("\nThe net could not be built in space for any value of the flexion parameter (%s).\nChange the base angles or the row types." % state_fail.get('reason', 'unknown reason'))
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
    def verify(pos, th, t, tex=True):
        """everything recomputed from the drawn vertices: congruence of all faces, prescribed flat angles, planarity, dihedral angles
        (tex: face names as mathtext for the panels; plain names for the text file)."""
        d_len = max(abs(edge_length(pos, e) - ref_lengths[e]) for e in edges_all)
        d_fa = max(abs(face_angle(pos, v, f) - ref_face_angles[(v, f)]) for f in faces_all for v in corners(f))
        d_flat = max(abs(face_angle(pos, v, f) - ang[v][f]) for v in ang for f in ang[v])
        def normal(f): c = [pos[v] for v in corners(f)]; return unit(np.cross(c[1]-c[0], c[3]-c[0]))
        d_plan = max(abs(np.dot(pos[c[2]] - pos[c[0]], normal(f)))/np.linalg.norm(pos[c[2]] - pos[c[0]]) for f in faces_all for c in [corners(f)])
        d_dih = 0.0                                          # oriented dihedral angles read from the vertices vs. the formula values
        for f in th:
            for i in range(1, 5):
                d_dih = max(d_dih, abs(dihedral(subs[f], i, pos) - th[f][i]))
        d_conv = 0.0
        for f in faces_all:
            c = [pos[v] for v in corners(f)]; N = normal(f)
            cr = [np.dot(np.cross(c[(k+1)%4]-c[k], c[(k+2)%4]-c[(k+1)%4]), N) for k in range(4)]
            d_conv = max(d_conv, 0.0 if all(x > 0 for x in cr) else 1.0)
        light = len(faces_all) > LIGHT_FACES
        live_pen = len(faces_all) <= LIVE_PEN_FACES                                 # small nets: the intersection test also while dragging
        pairs = [] if (light or (state.get('dragging', False) and not live_pen)) else self_intersections(pos, faces_all, corners)
        state_pen = ("(not tested for nets with more than %d faces)" % LIGHT_FACES if light else ("(checked when the slider is released)" if (state.get('dragging', False) and not live_pen) else "none")) if not pairs else ", ".join(Fn(f, tex) + " & " + Fn(g, tex) for f, g in pairs[:3]) + (" (+%d more)" % (len(pairs) - 3) if len(pairs) > 3 else "")
        rep = (("$t$" if tex else "t") + " = %.4f\n\n  edge lengths vs. initial:      %.1e (all %d edges)\n  face angles vs. initial:       %.1e rad (all %d faces)\n"
               "  flat angles vs. prescribed:    %.1e rad (%d inner vertices)\n  planarity of faces:            %.1e\n"
               "  dihedral angles vs. formulas:  %.1e rad (%d central edges)\n  convex faces:                  %s\n"
               "  self-intersection:             %s"
               % (t, d_len, len(edges_all), d_fa, len(faces_all), d_flat, len(ang), d_plan, d_dih, 4*len(th), "all" if d_conv == 0 else "NO", state_pen))
        state['penetrating'] = set(sum([[f, g] for f, g in pairs], []))
        return rep
    P0 = np.array(list(pos0.values())); mid = (P0.max(0) + P0.min(0))/2; rng = (P0.max(0) - P0.min(0)).max()*0.56

    # ---------------------------------------------------------------- figure and panels
    fig = plt.figure(figsize=(12.5, 9.5), facecolor='white')
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
    def relayout():
        for group in groups:
            for a in groups[group]:
                p = mapped(group, a._design)
                if isinstance(a, matplotlib.text.Text): a.set_position(p[:2])
                else: a.set_position(list(p))
        fig.canvas.draw_idle()
    state_scroll = {'relayout': relayout}
    ax = fig.add_axes([0.13, 0.02, 0.72, 0.93], projection='3d')
    title_text = fig.text(0.56, 0.975, "%d \u00d7 %d GQS-net" % (m, n), ha="center", va="top", fontsize=13.5, color=INK, fontweight="bold")
    state = {'t': t0, 'pos': pos0, 'th': th0, 'labels': {'vertices': False, 'faces': False, 'edges': False, 'angles': False},
             'elev': 30, 'azim': -55, 'report': True, 'info': False, 'unrolled': 0,
             'hidden': True, 'shadow': SHOW_SHADOW, 'selected': None, 'dfig': None}
    def panel(x, y, size, **kw):
        return fig.text(x, y, "", fontsize=size, family='monospace', va='top', color=INK, linespacing=1.25,
                        bbox=dict(boxstyle='round,pad=0.6,rounding_size=0.9', facecolor=PANEL_FC, edgecolor=PANEL_EC, alpha=0.88), **kw)
    report_text = panel(0.012, 0.968, 7.9)
    n_edges, n_faces, n_inner, n_blocks = m*(n + 1) + n*(m + 1), m*n, (m - 1)*(n - 1), (m - 2)*(n - 2)
    INFO = ("HOW THE CHECKS ARE COMPUTED (all from the current vertex\n"
            "coordinates, recomputed at every position of the slider)\n\n"
            "edge lengths vs. initial\n"
            r"  $|V_{ij} - V_{kl}|$ for each of the %d edges, minus the same" % n_edges + "\n"
            "  length in the initial configuration; maximum over edges.\n\n"
            "face angles vs. initial   (rigidity: nothing changes)\n"
            "  at every corner of every face, including the corners at\n"
            "  the boundary of the net, the angle between the two edges\n"
            "  (arccos of the dot product of the unit edge vectors) minus\n"
            "  the value it had in the first configuration; maximum over\n"
            "  the %d angles. With the edge lengths: every face stays\n" % (4*n_faces) +
            "  congruent to itself, so the net moves as a mechanism.\n"
            "  This line does not say WHICH angles the faces have.\n\n"
            "flat angles vs. prescribed   (identity: the right values)\n"
            "  only at the %d inner vertices (%d angles, four faces\n" % (n_inner, 4*n_inner) +
            "  meet there): the measured angle minus the value the flat\n"
            "  angle must have, namely the base angles at the base block\n"
            "  propagated by the relations between adjacent 3 x 3 blocks.\n"
            "  At the boundary vertices only two faces meet and nothing\n"
            "  is prescribed, so they appear in the previous line only.\n\n"
            "planarity of faces\n"
            "  for each face, the distance of the fourth vertex from\n"
            "  the plane of the other three, divided by the diagonal.\n\n"
            "dihedral angles vs. formulas\n"
            r"  at each edge of each central face $F_{kn}$, the oriented" "\n"
            "  dihedral angle read from the drawn vertices (with the\n"
            r"  normals of the two faces and its sign) minus $\theta_i$ from" "\n"
            "  the explicit flexion formulas of the 3 x 3 block.\n\n"
            "convex faces\n"
            "  for every face the four cross products of consecutive\n"
            "  edge vectors point to the same side of the face normal.\n\n"
            "self-intersection\n"
            "  for every pair of faces without a common vertex, the\n"
            "  segment in which one face meets the plane of the other\n"
            "  is clipped by that other face; a nonempty result means\n"
            "  the faces penetrate each other (drawn in red tones).\n\n"
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
            "  footprint and stays with the net when you move.\n"
            "  Freeze locks the rotation so that the\n"
            "  left mouse button drags the picture instead.\n"
            "  Colours: the picture theme. Paper is the palette measured\n"
            "  from the reference renders; the other themes change the face\n"
            "  tones, the bounce hues, the specular strength, the inks of\n"
            "  the edges and hidden lines, the shadow and the canvas.\n"
            "  Advanced: pick each entry's colour on the wheel (with the\n"
            "  lightness slider) or type its code; - / + for tint mix,\n"
            "  specular and shadow opacity. Apply repaints the net.\n"
            "  Params: the net itself (angles, size, base block, row types,\n"
            "  signs, branch); it checks every choice before Apply.\n"
            "  Click on a face for its angles and\n"
            "  edges; for a central face also the data of its 3 x 3\n"
            r"  block ($M$, $r_i$, $s_i$, $f_i$, $K$, $K'$, phase shifts, signs $e_i$)" "\n"
            "  and a plot of its dihedral angles along the motion.\n"
            r"  Button Dihedrals: a window with $\theta_1, \dots, \theta_4$ of" "\n"
            "  all %d 3 x 3 blocks\n" % n_blocks +
            r"  as functions of $t$ (marker = current $t$)." "\n\n"
            "Values of 1e-14 ... 1e-16 are double-precision rounding;\n"
            "a geometric inconsistency would appear as 1e-3 or larger.")
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
    def report_string():                             # the report, continued by the unrolled part of the explanations
        body = state['report_str'] + (("\n\n" + "\n".join(INFO_LINES[:state['unrolled']])) if state['unrolled'] > 0 else "")
        return with_glyphs(body, " "*8)
    face_text = panel(0.012, 0.62, 7.9, visible=False)
    inset = fig.add_axes([0.05, 0.16, 0.14, 0.20]); inset.set_visible(False)
    set_projection(ax)
    def redraw():
        state['elev'], state['azim'] = ax.elev, ax.azim
        lims = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()) if state.get('drawn') else None   # keep the user's zoom
        ax.cla(); pos = state['pos']
        report = verify(state['pos'], state['th'], state['t'])      # also updates state['penetrating']
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
        else:
            # 'world': the sun at noon, fixed in space: the shadow is the vertical footprint of the net on the floor; the shading
            # light is anchored high above the first view (so that vertical faces still show their modelling)
            if 'L_world' not in state:
                el0 = state['elev'] if abs(state['elev']) <= 90 else 180 - state['elev']
                u0, up0, w0 = view_axes(max(el0, 15.0), state['azim'])
                state['L_world'] = unit(np.array([0, 0, 1.0]) + 0.35*w0 - 0.25*u0)
            L, Lsh = state['L_world'], np.array([0, 0, 1.0])
        allP = np.array(list(pos.values())); size = (allP.max(0) - allP.min(0)).max(); cen = (allP.max(0) + allP.min(0))/2
        state['shadow_box'] = None; state['shadow_pts'] = None
        if state['shadow']:
            lims_ = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
            if Lsh is None:                                                       # 'top': floor of the viewer's frame, seen from 30 degrees above
                nrm = unit(math.cos(math.radians(30))*up + math.sin(math.radians(30))*w)
                state['shadow_pts'] = draw_shadow_frame(ax, pos, faces_all, corners, u, unit(np.cross(nrm, u)), nrm, size)
            else:
                n_before = len(ax.collections)
                state['shadow_box'] = draw_shadow(ax, pos, faces_all, corners, Lsh, size)      # None when the light is too flat
                if state['shadow_box'] is not None and (w[2] > 0) != (Lsh[2] > 0):
                    # the viewer is on the other side of the shadow plane than the light (under the floor, or above the ceiling): the
                    # shadow lies on a glass plane BETWEEN the viewer and the net, so it is drawn in front of the net, translucent
                    for c in ax.collections[n_before:]:
                        c.do_3d_projection = (lambda orig: (lambda *a, **k: (orig(*a, **k), -2e9)[1]))(c.do_3d_projection)
            ax.set_xlim3d(lims_[0]); ax.set_ylim3d(lims_[1]); ax.set_zlim3d(lims_[2])   # contourf autoscales; undo
        sf = screen_faces(ax, pos, faces_all, corners)                 # projected faces with the current projection
        state['screen_faces'] = sf
        tints = {f: PEN_TINT for f in state.get('penetrating', set())}          # only penetrating faces are tinted
        quick = len(faces_all) > LIGHT_FACES                                        # the picture style never changes while dragging
        nsub = max(2, min(NSUB, int(round(NSUB*math.sqrt(25.0/len(faces_all))))))    # fewer pieces per face for large nets
        polys, cols = shaded_pieces(pos, faces_all, corners, up, w, L, cen, size, tints, nsub)
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
            polys, cols, layer = layered_pieces(ax.get_proj(), sf, polys, cols, max_split=2 if len(faces_all) <= 36 else 1)
            for lv in range(4):
                idx = np.nonzero(layer == lv)[0]
                if len(idx): ax.add_collection(LayerPolys([polys[i] for i in idx], layer=lv, facecolors=[cols[i] for i in idx], edgecolors='none', antialiased=False), autolim=False)
        # edges: visible parts in dark blue, parts behind faces as faint thin lines
        segs_all = {0: [], 1: [], 2: [], 3: []}; segs_edge = {0: []}
        for e in edges_all:
            P, Q = sorted(e)
            if quick: continue                                                     # quick / light mode: edges already drawn with the faces
            PQ_faces = {f for f in faces_all if P in corners(f) and Q in corners(f)}
            for lv, sg in edge_segments(ax, pos[P], pos[Q], PQ_faces, sf).items():
                segs_all[lv] += sg
                if lv == 0: segs_edge[0] += [e]*len(sg)
        style = {1: (HIDDEN_COL + (0.32,), 0.5), 2: (HIDDEN_COL + (0.20,), 0.45), 3: (HIDDEN_COL + (0.14,), 0.4)}
        for lv in (3, 2, 1):
            if not state['hidden']: continue
            if segs_all[lv]:
                col, lw = style[lv]
                ax.add_collection3d(FrontLines(segs_all[lv], rank=lv, colors=[col], linewidths=lw))
        if segs_all[0]:                                                              # visible parts: ink by the kind of crease
            inks = [edge_ink(sg_edge, pos, w, faces_all, corners) for sg_edge in segs_edge[0]]
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
        report_text.set_visible(state['report'])
        if 'layout_panels' in state: state['layout_panels']()
        fig.canvas.draw_idle()
    # ---- the motion sampled along t (cached configurations), for the plots of the dihedral angles
    def motion_samples(nmin):
        ts_ = sorted(k for k in cache if cache[k] is not None)
        if len(ts_) < nmin:
            for tt in np.linspace(lo, hi, nmin + 1): configuration(float(tt))
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
        if ev.inaxes is ax: state['press'] = (ev.x, ev.y)
    def on_motion(ev):
        """the occlusion layers and the hidden-line segments are valid only for the view they were computed for: as soon as a drag
        in the 3D axes starts, the picture is replaced by a plain depth-sorted preview (faces and edges in one collection), which
        stays correct under rotation; the full picture is redrawn when the mouse is released."""
        if state.get('press') is None or state.get('rotating') or state.get('preview') is None or ev.inaxes is not ax: return
        if abs(ev.x - state['press'][0]) < 3 and abs(ev.y - state['press'][1]) < 3: return
        state['rotating'] = True
        for c in list(ax.collections): c.remove()
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
            S = subs[f]; e = assignment[f]; th = state['th'][f]
            lines += [r"  central face of a $3 \times 3$ block with " + RELATION_TEXT[ROW_SYSTEMS[f[1]]],
                      r"  signs $(e_1, e_2, e_3, e_4)$ = %s" % (e,),
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
                      r"  (inset: $\theta_1, \dots, \theta_4$ of this block along the motion, marker = current $t$)"]
            ts_ = motion_samples(16); curves = block_curves(f, ts_)
            inset.cla(); inset.set_visible(True)
            for i, cv in enumerate(curves): inset.plot(ts_, cv, lw=1.2, color=THETA_COLS[i], label=r"$\theta_%d$" % (i+1))
            for i in range(4): inset.plot([state['t']], [math.degrees(th[i+1])], 'o', ms=4, color=THETA_COLS[i])
            inset.set_xlabel('$t$', fontsize=8, color=INK); inset.set_ylabel('deg', fontsize=7, color=INK); style_axes(inset)
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
        over the picture the wheel redraws after a zoom."""
        H, W = fig.get_figheight(), fig.get_figwidth(); step = 0.25/H*(1 if ev.button == 'down' else -1)     # a quarter inch per notch
        hstep = 0.25/W*(1 if ev.button == 'down' else -1); horizontal = (ev.key == 'shift')
        xf_, yf_ = ev.x/fig.bbox.width, ev.y/fig.bbox.height
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
                    lows = [measure(t).y0/fig.bbox.height for t in (report_text, face_text) if t.get_visible()]
                    if inset.get_visible(): lows.append(inset.get_position().y0 - 0.45/fig.get_figheight())
                    max_off = max(0.0, -min(lows) + 0.02 + offs['panel'])
                    if horizontal: offs_x['panel'] = min(max(0.0, offs_x['panel'] + hstep), overflow_x('panel'))
                    else: offs['panel'] = min(max(0.0, offs['panel'] + step), max_off)
                    layout_panels(); fig.canvas.draw_idle(); return
        if ev.inaxes is ax: redraw()
    fig.canvas.mpl_connect('scroll_event', on_scroll_ui)
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
        ts_ = motion_samples(40)
        nk, nn = m - 2, n - 2                                                    # blocks: kappa = 1..nk (columns), nu = 1..nn (rows)
        vk, vn = min(nk, 3), min(nn, 3)
        dfig = plt.figure(figsize=(3.7*vk + 0.9, 2.75*vn + 1.6), facecolor='white')
        try: dfig.canvas.manager.set_window_title("dihedral angles of the 3 x 3 blocks")
        except Exception: pass
        def geometry():
            """viewport and plot sizes in figure fractions for the CURRENT window size (the plots keep their size in inches, 3.7 x 2.75)."""
            W, Hf = dfig.get_figwidth(), dfig.get_figheight()
            L0, R0 = 0.55/W, (W - 0.45)/W                                        # viewport (room for the scrollbars, always reserved)
            B0, T0 = 0.75/Hf, (Hf - 1.18)/Hf
            gx, gy = 0.55/W, 0.62/Hf                                             # gaps between the plots
            tw, th_ = 3.7/W - gx, 2.75/Hf - gy                                   # size of one plot: fixed in inches
            return L0, R0, B0, T0, gx, gy, tw, th_
        L0, R0, B0, T0, gx, gy, tw, th_ = geometry()
        shown_blocks = sorted(subs) if len(subs) <= 100 else sorted(kn for kn in subs if abs(kn[0] - BASE_KN[0]) <= 3 and abs(kn[1] - BASE_KN[1]) <= 3)
        tiles = {}; marks = []; leg_holder = []
        for kn in shown_blocks:
            a_ = dfig.add_axes([0, 0, tw, th_]); a_.set_zorder(0)
            for i, cv in enumerate(block_curves(kn, ts_)): a_.plot(ts_, cv, lw=1.3, color=THETA_COLS[i], label=r"$\theta_%d$" % (i+1))
            a_.set_title(r"%s   (signs $e$ = %s)" % (Fn(kn), assignment[kn]), fontsize=9.5, color=INK)
            vl = a_.axvline(state['t'], color='#8fa9c9', lw=0.9, ls='--')
            dots = [a_.plot([state['t']], [math.degrees(state['th'][kn][i])], 'o', ms=4, color=THETA_COLS[i-1])[0] for i in range(1, 5)]
            style_axes(a_); a_.set_xlabel('$t$', fontsize=9, color=INK); a_.set_ylabel('deg', fontsize=8, color=INK); a_.tick_params(labelsize=7)
            tiles[kn] = a_; marks.append((kn, vl, dots))
        # masks above and below the viewport so that partly scrolled plots do not run into the title, the legend and the scrollbars
        masks = []
        def place_masks(L0, R0, B0, T0, gx, gy):
            for p_ in masks: p_.remove()
            masks.clear()
            for rect in ([0, T0 + 0.35*gy, 1, 1 - T0 - 0.35*gy], [0, 0, 1, B0 - 0.45*gy], [0, 0, L0 - 0.55*gx, 1], [R0 + 0.3*gx, 0, 1 - R0 - 0.3*gx, 1]):
                p_ = FancyBboxPatch((rect[0], rect[1]), rect[2], rect[3], boxstyle="square,pad=0", transform=dfig.transFigure,
                                    facecolor='white', edgecolor='none', zorder=1)
                dfig.add_artist(p_); masks.append(p_)
        place_masks(L0, R0, B0, T0, gx, gy)
        dfig.text(0.5, 0.985, r"dihedral angles $\theta_1, \dots, \theta_4$ along the motion, with respect to the central face $F_{\kappa\nu}$" "\n"
                  r"of the corresponding $3 \times 3$ block, $1 \leq \kappa \leq %d$, $1 \leq \nu \leq %d$" % (nk, nn) + ("" if len(subs) <= 100 else "  (large net: only the blocks near the base block)"),
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
            if ev.key == 'shift' or (limits['oy'] == 0 and limits['ox'] > 0):
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
        try: dfig.show()
        except Exception: pass
        plt.figure(fig.number)                                                     # the main figure stays the current one
    def update_dihedrals():
        dfig = state['dfig']
        if dfig is None: return
        if not plt.fignum_exists(dfig.number):          # closed by hand on a backend without close events
            state['dfig'] = None; b_dih.set_on(False); return
        for kn, vl, dots in state['dmarks']:
            vl.set_xdata([state['t'], state['t']])
            for i, d in enumerate(dots): d.set_data([state['t']], [math.degrees(state['th'][kn][i+1])])
        dfig.canvas.draw_idle()
    def close_dihedrals():
        dfig = state['dfig']; state['dfig'] = None
        if dfig is not None and plt.fignum_exists(dfig.number): plt.close(dfig)
    # ---------------------------------------------------------------- controls: the right column
    X0, XW, H = 0.862, 0.125, 0.030
    ax_sl = fig.add_axes([0.909, 0.545, 0.030, 0.375]); ax_sl.set_facecolor('none')
    sl = make_slider(ax_sl, lo, hi, t0); reg('col', ax_sl)
    def on_change(val):
        cfg = configuration(float(val))
        if cfg is not None:
            state['t'] = float(val); state['pos'] = cfg[0]; state['th'] = cfg[3]; redraw()
            if state['selected'] is not None: show_face(state['selected'])
            update_dihedrals()
    sl.on_changed(on_change)
    # while the slider is dragged only the self-intersection test of medium-size nets is postponed to the release (the picture style is unchanged)
    def sl_press(ev):
        if ev.inaxes is ax_sl: state['dragging'] = True
    def sl_release(ev):
        if state.get('dragging'):
            state['dragging'] = False; redraw()
            if state['selected'] is not None: show_face(state['selected'])
    fig.canvas.mpl_connect('button_press_event', sl_press); fig.canvas.mpl_connect('button_release_event', sl_release)
    def half(y, side): return [X0 if side == 0 else X0 + XW - 0.058, y, 0.058, H]
    step = (hi - lo)/40
    b_minus = FancyButton(fig, half(0.482, 0), '−', fontsize=11); b_plus = FancyButton(fig, half(0.482, 1), '+', fontsize=11); reg('col', b_minus.ax, b_plus.ax)
    b_minus.on_clicked(lambda ev: sl.set_val(max(lo, sl.val - step))); b_plus.on_clicked(lambda ev: sl.set_val(min(hi, sl.val + step)))
    rows = [0.418, 0.378, 0.338, 0.298, 0.258]
    b_vert = FancyButton(fig, half(rows[0], 0), 'Vertices', toggle=True); b_face = FancyButton(fig, half(rows[0], 1), 'Faces', toggle=True)
    b_edge = FancyButton(fig, half(rows[1], 0), 'Lengths', toggle=True); b_angl = FancyButton(fig, half(rows[1], 1), 'Angles', toggle=True)
    b_hid = FancyButton(fig, half(rows[2], 0), 'Hidden', toggle=True, on=True); b_shad = FancyButton(fig, half(rows[2], 1), 'Shadow', toggle=True, on=SHOW_SHADOW)
    b_rep = FancyButton(fig, half(rows[3], 0), 'Report', toggle=True, on=True); b_reset = FancyButton(fig, half(rows[3], 1), 'Reset')
    b_dih = FancyButton(fig, half(rows[4], 0), 'Dihedrals', toggle=True); b_frz = FancyButton(fig, half(rows[4], 1), 'Freeze', toggle=True)
    reg('col', *[b.ax for b in (b_vert, b_face, b_edge, b_angl, b_hid, b_shad, b_rep, b_reset, b_dih, b_frz)])
    reg('col', fig.text(X0 + XW/2, 0.125, 'save', ha='center', va='bottom', fontsize=8, color='#6b7d99'))
    def third(i): return [X0 + i*(0.0395 + 0.0033), 0.088, 0.0395, H]
    b_svg = FancyButton(fig, third(0), 'SVG'); b_png = FancyButton(fig, third(1), 'PNG'); b_obj = FancyButton(fig, third(2), 'OBJ'); reg('col', b_svg.ax, b_png.ax, b_obj.ax)
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
        ax.view_init(elev=30, azim=-55); state['drawn'] = False; state.pop('L_world', None); state.pop('Lsh_world', None); redraw()
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
    b_par = FancyButton(fig, half(0.212, 0), 'Params', toggle=True); reg('col', b_par.ax)
    b_col = FancyButton(fig, half(0.212, 1), 'Colours', toggle=True); reg('col', b_col.ax)
    # light: four options. eye = a flashlight in the viewer's hand; top = the sun at the top of the screen (fixed to the screen);
    # studio = the original setup; world = the sun at noon fixed in space
    reg('col', fig.text(X0 + XW/2, 0.187, 'light', ha='center', va='bottom', fontsize=8, color='#6b7d99'))
    LIGHT_MODES = [('eye', 'Eye'), ('top', 'Top'), ('studio', 'Studio'), ('world', 'World')]
    gap = 0.0025; weights = [0.9, 0.9, 1.25, 1.05]                              # widths in proportion to the label lengths
    unit_w = (XW - 3*gap)/sum(weights)
    b_light = {}; xq = X0
    for (key, label), wt in zip(LIGHT_MODES, weights):
        b_light[key] = FancyButton(fig, [xq, 0.150, wt*unit_w, H], label, toggle=True, on=(LIGHT_FRAME == key), fontsize=8); xq += wt*unit_w + gap
    reg('col', *[b.ax for b in b_light.values()])
    def set_light(frame):
        def f(ev):
            state['light_frame'] = frame; state.pop('L_world', None)
            for key, b in b_light.items(): b.set_on(key == frame)
            redraw()
        return f
    for key, b in b_light.items(): b.on_clicked(set_light(key))
    dlg = {'axes': [], 'dyn': [], 'rowbtns': [], 'sgnbtns': [], 'open': False, 'sgnsel': tuple(BASE_SIGNS)}
    veil = fig.add_axes([0, 0, 1, 1], zorder=30); veil.set_axis_off(); veil.set_navigate(False)
    veil.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0", transform=veil.transAxes, facecolor='white', alpha=0.86, edgecolor='none'))
    veil.set_visible(False)
    PX, PY, PW, PH = 0.20, 0.10, 0.60, 0.80                                     # the dialog frame
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
    def spinbox(x, y, w, val, key, step, lo_, hi_, fmt, after):
        """a text box with up/down arrows; 'after' is called when the value changes."""
        axb = fig.add_axes([x, y - 0.014, w, 0.028], zorder=32); axb.set_visible(False)
        tb = TextBox(axb, '', initial=fmt % val, textalignment='center', color='#ffffff', hovercolor='#eef3fa')
        tb.text_disp.set_fontsize(9.5); dlg['axes'].append(axb); dlg[key] = tb; reg('dlg', axb); tb.set_active(False)
        dlg.setdefault('textboxes', []).append(tb)
        def bump(d):
            def f(ev):
                try: v = float(tb.text) + d*step
                except ValueError: return
                v = min(max(v, lo_), hi_); tb.set_val(fmt % v)                  # set_val fires on_submit, which calls 'after'
            return f
        up = dbutton(x + w + 0.004, y + 0.0075, 0.02, '▲', h=0.014, fontsize=6, rounding=0.3)
        dn = dbutton(x + w + 0.004, y - 0.0075, 0.02, '▼', h=0.014, fontsize=6, rounding=0.3)
        up.on_clicked(bump(+1)); dn.on_clicked(bump(-1)); tb.on_submit(lambda txt: after())
        return tb
    dtext(PX + PW/2, PY + PH - 0.04, 'Parameters of the net', 12, ha='center', fontweight='bold')
    # --- angles
    y = PY + PH - 0.095
    dtext(LX, y, r'flat angles at the vertex $A_1$ of the base block (degrees):  $\alpha_1$, $\beta_1$, $\gamma_1$, $\delta_1$')
    def on_angles(): refresh_signs()
    for j, a in enumerate(BASE_ANGLES_DEG): spinbox(LX + j*0.135, y - 0.045, 0.095, a, 'ang%d' % j, 1.0, 1.0, 179.0, '%g', on_angles)
    # --- size and base
    y -= 0.105
    dtext(LX, y, 'faces  M (columns), N (rows)')
    def on_size(): build_row_buttons(); refresh_signs()
    spinbox(LX + 0.30, y, 0.06, m, 'M', 1, 3, 60, '%d', on_size); spinbox(LX + 0.40, y, 0.06, n, 'N', 1, 3, 60, '%d', on_size)
    y -= 0.055
    dtext(LX, y, r'central face $F_{\kappa\nu}$ of the base block:  $\kappa$, $\nu$')
    spinbox(LX + 0.30, y, 0.06, BASE_KN[0], 'kap', 1, 1, 58, '%d', lambda: None); spinbox(LX + 0.40, y, 0.06, BASE_KN[1], 'nu', 1, 1, 58, '%d', lambda: refresh_signs())
    # --- row types
    y -= 0.06
    dtext(LX, y, r'type of each row of blocks, $\nu = 1$ (bottom) $\ldots$ $N-2$ (top), any combination; click: a $\to$ b $\to$ c $\to$ d')
    ROWS_Y = y - 0.045
    legend_txt = dtext(LX, ROWS_Y - 0.085, "", 7.4, va='top', linespacing=1.4, color='#4a5b78')
    legend_txt.set_text("in every block, vertices 2 and 3 carry the complements to pi of the flat angles at vertices 1 and 4; the type fixes vertex 4 from vertex 1:\n"
                        r"a: $(\alpha_4,\beta_4,\gamma_4,\delta_4)=(\pi-\alpha_1,\ \pi-\beta_1,\ \pi-\gamma_1,\ \pi-\delta_1)$   b: $(\beta_1,\ \alpha_1,\ \delta_1,\ \gamma_1)$   "
                        r"c: $(\alpha_1,\ \beta_1,\ \pi-\gamma_1,\ \pi-\delta_1)$   d: $(\beta_1,\ \alpha_1,\ \pi-\delta_1,\ \pi-\gamma_1)$")
    dlg['row_off'] = 0                                                          # first row shown in the strip of row-type buttons
    dlg['rowtypes'] = dict(ROW_SYSTEMS)                                          # the chosen type of every row (also of the rows not on screen)
    ROWS_PER_LINE, ROW_LINES = 9, 2
    def build_row_buttons():
        for b in dlg['rowbtns']: b.ax.remove(); groups['dlg'].remove(b.ax)
        for t_ in dlg['dyn']:
            a_ = t_.ax if hasattr(t_, 'ax') else t_
            a_.remove(); groups['dlg'].remove(a_)
        dlg['rowbtns'], dlg['dyn'] = [], []
        try: nrows = max(1, min(int(dlg['N'].text) - 2, 60))
        except ValueError: nrows = n - 2
        base_type = dlg['rowtypes'].get(BASE_KN[1], 'a')
        for r in range(1, nrows + 1): dlg['rowtypes'].setdefault(r, base_type)
        cap = ROWS_PER_LINE*ROW_LINES
        dlg['row_off'] = max(0, min(dlg['row_off'], nrows - cap))
        shown = range(dlg['row_off'] + 1, min(nrows, dlg['row_off'] + cap) + 1)
        for k, r in enumerate(shown):
            col, line = k % ROWS_PER_LINE, k // ROWS_PER_LINE
            xx, yy = LX + col*0.0595, ROWS_Y - line*0.048
            t_ = fig.text(xx + 0.024, yy + 0.024, r'$\nu=%d$' % r, fontsize=7.2, color='#4a5b78', ha='center', va='center', zorder=32, visible=dlg['open'])
            b = FancyButton(fig, [xx, yy - 0.014, 0.048, 0.028], dlg['rowtypes'][r], fontsize=9.5)
            b.ax.set_zorder(32); b.ax.set_visible(dlg['open'])
            def cycle(bb, rr):
                def f(ev):
                    new = 'abcd'[('abcd'.index(bb.text.get_text()) + 1) % 4]; bb.text.set_text(new); dlg['rowtypes'][rr] = new; refresh_signs()
                return f
            b.on_clicked(cycle(b, r)); dlg['rowbtns'].append(b); dlg['dyn'].append(t_); reg('dlg', b.ax, t_)
        if nrows > cap:                                                          # more rows than fit: arrows scroll the strip
            xr = LX + ROWS_PER_LINE*0.0595 + 0.01
            up = FancyButton(fig, [xr, ROWS_Y - 0.014, 0.022, 0.028], '▲', fontsize=7, rounding=0.3)
            dn = FancyButton(fig, [xr, ROWS_Y - 0.048 - 0.014, 0.022, 0.028], '▼', fontsize=7, rounding=0.3)
            for bb in (up, dn): bb.ax.set_zorder(32); bb.ax.set_visible(dlg['open']); dlg['dyn'].append(bb); reg('dlg', bb.ax)
            def shift(d):
                def f(ev): dlg['row_off'] = max(0, min(dlg['row_off'] + d*ROWS_PER_LINE, nrows - cap)); build_row_buttons()
                return f
            up.on_clicked(shift(-1)); dn.on_clicked(shift(+1))
        relayout()
    # --- signs
    y = ROWS_Y - 0.15
    dtext(LX, y, r'signs $(e_1, e_2, e_3, e_4)$ of the flexion of the base block; admissible: $e_1 t_1 + e_2 t_2 + e_3 t_3 + e_4 t_4 \in \Lambda$')
    SIGNS_Y = y - 0.045
    sgn_note = dtext(LX, SIGNS_Y - 0.024, "", 7.6, color='#4a5b78', va='top', linespacing=1.4)
    _sgn_set = sgn_note.set_text
    sgn_note.set_text = lambda txt: _sgn_set(wrap_math(txt, 118))
    def refresh_signs():
        key = tuple(dlg['ang%d' % j].text for j in range(4)) + (dlg['M'].text, dlg['N'].text, dlg['kap'].text, dlg['nu'].text) + tuple(sorted(dlg['rowtypes'].items()))
        if dlg.get('sgn_key') == key: return
        dlg['sgn_key'] = key
        par_msg.set_visible(False)                                                # a stale "cannot apply" message goes with the change
        for b in dlg['sgnbtns']: b.ax.remove(); groups['dlg'].remove(b.ax)
        dlg['sgnbtns'] = []
        sgn_note.set_color('#4a5b78')
        try:
            angs = tuple(float(dlg['ang%d' % j].text) for j in range(4))
            problem = angle_problem(angs)                                         # the hypotheses on the base angles, in words
            if problem: raise ValueError(problem)
            nu = int(dlg['nu'].text); rtype = dlg['rowtypes'].get(nu, ROW_SYSTEMS[BASE_KN[1]])
            Bt = Block(1, 1, block_from_vertex1(tuple(math.radians(a) for a in angs), rtype)); pats = Bt.witnesses()
            sgn_note.set_text(r"base row of type %s,  $M$ = %.4f:  %d admissible pattern%s ($e_4$ is fixed by the other three signs)" % (rtype, Bt.M, len(pats), "" if len(pats) == 1 else "s"))
        except Exception as ex:
            pats = []; sgn_note.set_text(str(ex)); sgn_note.set_color('#b00020')
        # which admissible patterns extend to a flexible net with the rows and the size currently entered (trial synchronization)
        feas = {}
        try:
            mm, nn_ = int(dlg['M'].text), int(dlg['N'].text); kk, nu = int(dlg['kap'].text), int(dlg['nu'].text)
            rows_new = {r: dlg['rowtypes'][r] for r in range(1, nn_ - 1) if r in dlg['rowtypes']}
            if nu not in rows_new: rows_new[nu] = 'a'
            rows_ok = check_parameters(mm, nn_, (kk, nu), rows_new, verbose=False)
            sub_t = assemble_net(angs, mm, nn_, (kk, nu), rows_ok); subs_t = {kn: Block(kn[0], kn[1], q) for kn, q in sub_t.items()}
            lo_t, hi_t = admissible_t_range(subs_t[(kk, nu)])
            ang_t = vertex_face_angles(sub_t); t_try = lo_t + 0.15*(hi_t - lo_t)
            for pat in pats:                                                      # synchronization AND construction in space
                th_t = None
                for e0_ in (1, -1):
                    th_t, _ = synchronize(subs_t, (kk, nu), pat, e0_, t_try, max_nodes=6*len(subs_t))
                    if th_t is not None: break
                if th_t is None: feas[pat] = 'nosync'; continue
                try: build_net(ang_t, th_t, mm, nn_, (kk, nu), BASE_EDGE); feas[pat] = True
                except AssertionError: feas[pat] = 'nobuild'
        except Exception as ex:
            feas = {pat: True for pat in pats}; sgn_note.set_text(sgn_note.get_text() + "\ncould not check the construction: %s" % (ex.args[0] if ex.args else ex,))
        good = [p for p in pats if feas.get(p, True) is True]
        if dlg['sgnsel'] not in good and good: dlg['sgnsel'] = good[0]
        dlg['can_apply'] = bool(good)
        set_apply_enabled(bool(good))
        reasons = set(v for v in feas.values() if v is not True)
        why = " / ".join(r for r in ("do not extend to a flexible net with these row types" if 'nosync' in reasons else "",
                                    "give a net that the construction cannot build in space" if 'nobuild' in reasons else "") if r)
        if pats and not good:
            sgn_note.set_text(sgn_note.get_text() + "\nno sign pattern works here (all patterns " + why + "): change a row type or the angles")
        elif pats and len(good) < len(pats):
            sgn_note.set_text(sgn_note.get_text() + "\ngreyed patterns " + why)
        for j, pat in enumerate(pats):
            ok_ = feas.get(pat, True) is True
            b = FancyButton(fig, [LX + j*0.135, SIGNS_Y - 0.015, 0.125, 0.03], "(%+d, %+d, %+d, %+d)" % pat, toggle=True, on=(pat == dlg['sgnsel'] and ok_), fontsize=8.5)
            b.ax.set_zorder(32); b.ax.set_visible(dlg['open'])
            if not ok_: b.text.set_color('#b5bcc8'); b.patch.set_edgecolor('#dde3ec')
            def pick(bb, pp, okk):
                def f(ev):
                    if not okk: bb.set_on(False); return
                    dlg['sgnsel'] = pp; par_msg.set_visible(False)
                    for b2 in dlg['sgnbtns']: b2.set_on(b2 is bb)
                return f
            b.on_clicked(pick(b, pat, ok_)); dlg['sgnbtns'].append(b); reg('dlg', b.ax)
        relayout()
    # --- branch
    y = SIGNS_Y - 0.085
    dtext(LX, y, r'branch $e_0$ of the flexion')
    e0_btns = [dbutton(LX + 0.30, y, 0.06, '−1', toggle=True, on=(E0 == -1)), dbutton(LX + 0.40, y, 0.06, '+1', toggle=True, on=(E0 == 1))]
    def pick_e0(k):
        def f(ev):
            par_msg.set_visible(False)
            for j, b in enumerate(e0_btns): b.set_on(j == k)
        return f
    for k, b in enumerate(e0_btns): b.on_clicked(pick_e0(k))
    # --- apply / cancel
    b_apply = dbutton(PX + PW/2 - 0.11, PY + 0.03, 0.10, 'Apply'); b_cancel = dbutton(PX + PW/2 + 0.01, PY + 0.03, 0.10, 'Cancel')
    def set_apply_enabled(on):
        dlg['can_apply'] = on
        if on: b_apply.COLORS = FancyButton.COLORS; b_apply.text.set_color('#2c4a74'); b_apply.patch.set_edgecolor('#b9c7da'); b_apply.patch.set_facecolor('#f7f9fc')
        else: b_apply.text.set_color('#b5bcc8'); b_apply.patch.set_edgecolor('#dde3ec'); b_apply.patch.set_facecolor('#f4f6f9')
    par_msg = fig.text(PX + PW/2, PY + 0.062, "", fontsize=8.5, color='#b00020', ha='center', va='bottom', zorder=32, visible=False, linespacing=1.3)
    reg('dlg', par_msg)
    _msg_set = par_msg.set_text
    par_msg.set_text = lambda txt: _msg_set(wrap_math(txt, 100))
    build_row_buttons(); refresh_signs()
    def set_dialog(on):
        dlg['open'] = on
        for tb in dlg.get('textboxes', []): tb.set_active(on)                    # hidden text boxes must not react to clicks (they grab the mouse)
        if not on and fig.canvas.mouse_grabber is not None:
            try: fig.canvas.release_mouse(fig.canvas.mouse_grabber)
            except Exception: pass
        ax.set_visible(not on)                                                    # the 3D scene is not redrawn while the dialog is open (speed)
        veil.set_visible(on); frame.set_visible(on)
        for a_ in dlg['axes']: a_.set_visible(on)
        for b in dlg['rowbtns'] + dlg['sgnbtns']: b.ax.set_visible(on)
        for t_ in dlg['dyn']: (t_.ax.set_visible(on) if hasattr(t_, 'ax') else t_.set_visible(on))
        if not on: par_msg.set_visible(False)
        offs['dlg'] = 0.0; offs_x['dlg'] = 0.0; relayout()
        b_par.set_on(on); fig.canvas.draw_idle()
    b_par.on_clicked(lambda ev: set_dialog(b_par.on))

    # ---- colours dialog (button Colours): named themes; Advanced = pick the colour of each entry from a palette (no typing)
    import matplotlib.colors as mcolors
    ENTRIES = [('LIGHT_COL', 'peak highlight'), ('SHADOW_COL', 'deepest face tone'), ('COOL_DEEP', 'rim (grazing faces)'), ('WARM_BOUNCE', 'warm bounce'),
               ('COOL_BOUNCE', 'cool bounce'), ('EDGE_COL', 'silhouette ink'), ('EDGE_SOFT', 'interior creases'), ('HIDDEN_COL', 'hidden lines'),
               ('GROUND_COL', 'shadow ink'), ('PEN_LIGHT', 'penetration, light'), ('PEN_DARK', 'penetration, dark')]
    NUMBERS = [('TINT_MIX', 'tint mix', 0.05), ('SPEC_K', 'specular', 0.04), ('GROUND_A', 'shadow opacity', 0.02)]
    cdl = {'axes': [], 'adv_axes': [], 'buttons': [], 'open': False, 'advanced': False, 'vals': dict(COLOR_DEFAULTS), 'theme': 'Paper', 'sel': 'EDGE_COL',
           'swatch': {}, 'pal': {}, 'num_text': {}}
    CX, CY, CW, CH = 0.25, 0.09, 0.50, 0.82
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
        theme_btns[name] = cbutton(CX + 0.03 + c_*0.0735, CY + CH - 0.125 - r_*0.038, 0.068, name, toggle=True, on=(name == 'Paper'), fontsize=8.5)
    def refresh_swatches():
        for key, a_ in cdl['swatch'].items():
            a_.set_facecolor(mcolors.to_hex(tuple(float(v) for v in cdl['vals'][key])))
            for sp in a_.spines.values(): sp.set_edgecolor('#2c4a74' if key == cdl['sel'] else '#b9c7da'); sp.set_linewidth(2.0 if key == cdl['sel'] else 0.8)
        for key, t_ in cdl['num_text'].items(): t_.set_text("%.2f" % cdl['vals'][key])
        if 'hex_box' in cdl:
            cdl['hex_box'].eventson = False; cdl['hex_box'].set_val(mcolors.to_hex(tuple(float(v) for v in cdl['vals'][cdl['sel']])).upper()); cdl['hex_box'].eventson = True
        fig.canvas.draw_idle()
    def pick_theme(name):
        def f(ev):
            cdl['theme'] = name; cdl['vals'] = dict(THEMES[name])
            for nm, b in theme_btns.items(): b.set_on(nm == name)
            refresh_swatches()
        return f
    for name, b in theme_btns.items(): b.on_clicked(pick_theme(name))
    # --- advanced part: entries with swatches (click to select), a palette (click to assign), numbers with - / +
    b_adv = cbutton(CX + 0.03, CY + CH - 0.215, 0.14, 'Advanced ...', toggle=True, fontsize=9)
    ctext(CX + 0.03, CY + CH - 0.255, 'click an entry, then a colour of the wheel (or type a code)', 8, adv=True, color='#4a5b78')
    y = CY + CH - 0.29
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
    WX, WY, WD = CX + 0.29, CY + CH - 0.56, 0.17
    wheel_ax = fig.add_axes([WX, WY, WD, WD*fig.get_figwidth()/fig.get_figheight()], zorder=32); wheel_ax.set_axis_off(); wheel_ax.set_navigate(False)
    wheel_ax.set_visible(False); cdl['adv_axes'].append(wheel_ax); reg('dlg', wheel_ax)
    NW = 160; yy, xx = np.mgrid[0:NW, 0:NW]; xr, yr = (xx - NW/2 + 0.5)/(NW/2), (yy - NW/2 + 0.5)/(NW/2)
    rad = np.sqrt(xr**2 + yr**2); hue = (np.arctan2(yr, xr)/(2*PI)) % 1.0
    cdl['wheel_light'] = 0.6
    def wheel_image(light):
        img = np.ones((NW, NW, 4))
        h_, s_ = hue.ravel(), np.clip(rad, 0, 1).ravel()
        rgb = np.array([colorsys.hls_to_rgb(hh, light, ss) for hh, ss in zip(h_, s_)]).reshape(NW, NW, 3)
        img[..., :3] = rgb; img[..., 3] = (rad <= 1.0)
        return img
    wheel_im = wheel_ax.imshow(wheel_image(0.6), extent=(-1, 1, -1, 1), origin='lower', interpolation='bilinear')
    ctext(WX, WY + WD*fig.get_figwidth()/fig.get_figheight() + 0.02, 'colour wheel', 8.5, adv=True)
    ctext(WX, WY - 0.03, 'lightness', 8.5, adv=True)
    lax = fig.add_axes([WX, WY - 0.055, WD, 0.014], zorder=32); lax.set_visible(False); lax.set_navigate(False)
    cdl['adv_axes'].append(lax); reg('dlg', lax)
    light_sl = Slider(lax, '', 0.05, 0.95, valinit=0.6, color='#8fb6e3'); light_sl.valtext.set_visible(False)
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
        cdl['hex_box'].set_active(on and cdl['open'])
        b_adv.set_on(on); fig.canvas.draw_idle()
    b_adv.on_clicked(lambda ev: show_advanced(b_adv.on))
    def set_cdialog(on):
        cdl['open'] = on
        ax.set_visible(not on); veil.set_visible(on); cframe.set_visible(on)
        for a_ in cdl['axes']: a_.set_visible(on)
        for a_ in cdl['adv_axes']: a_.set_visible(on and cdl['advanced'])
        cdl['hex_box'].set_active(on and cdl['advanced'])
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
        for t_ in (report_text, face_text): t_.set_color(ink); t_.get_bbox_patch().set_facecolor(vals.get('FACE_COLOR', '#FFFFFF') if ink != '#2c4a74' else PANEL_FC)
        title_text.set_color(ink)
    def capply(ev):
        try: apply_colors(cdl['vals']); apply_canvas(cdl['vals'])
        except Exception as ex:
            cmsg.set_text("cannot apply: %s" % (ex,)); cmsg.set_visible(True); fig.canvas.draw_idle(); return
        set_cdialog(False); state['drawn'] = False; redraw()
    b_capply.on_clicked(capply); b_ccancel.on_clicked(lambda ev: set_cdialog(False))
    def col_click(ev):
        if dlg['open']: set_dialog(False)
        set_cdialog(b_col.on)
    b_col.on_clicked(col_click)
    b_cancel.on_clicked(lambda ev: set_dialog(False))
    _apply_style = b_apply._style
    def _apply_style_guard():
        _apply_style()
        if not dlg.get('can_apply', True): b_apply.text.set_color('#b5bcc8'); b_apply.patch.set_edgecolor('#dde3ec'); b_apply.patch.set_facecolor('#f4f6f9')
    b_apply._style = _apply_style_guard
    def apply_params(ev):
        global BASE_ANGLES_DEG, M_FACES, N_FACES, BASE_KN, ROW_SYSTEMS, BASE_SIGNS, E0
        if not dlg.get('can_apply', True):
            par_msg.set_text("cannot apply: with these row types no sign pattern extends to a flexible net"); par_msg.set_visible(True); fig.canvas.draw_idle(); return
        try:
            angs = tuple(float(dlg['ang%d' % j].text) for j in range(4))
            assert all(0 < a < 180 for a in angs), "the four angles must lie in (0, 180) degrees"
            problem = angle_problem(angs); assert problem is None, problem
            mm, nn_ = int(dlg['M'].text), int(dlg['N'].text)
            kk, nu = int(dlg['kap'].text), int(dlg['nu'].text)
            sg = tuple(dlg['sgnsel']); assert len(sg) == 4, "select one of the admissible sign patterns"
            e0_ = -1 if e0_btns[0].on else 1
            rows_new = {r: dlg['rowtypes'][r] for r in range(1, nn_ - 1) if r in dlg['rowtypes']}
            if nu not in rows_new: rows_new[nu] = 'a'
            rows_ok = check_parameters(mm, nn_, (kk, nu), rows_new)        # trial assembly and synchronization before touching the window
            sub_t = assemble_net(angs, mm, nn_, (kk, nu), rows_ok)
            subs_t = {kn: Block(kn[0], kn[1], q) for kn, q in sub_t.items()}
            assert sg in subs_t[(kk, nu)].witnesses(), "the selected sign pattern is not admissible for the base block"
            lo_t, hi_t = admissible_t_range(subs_t[(kk, nu)])
            th_t, _ = synchronize(subs_t, (kk, nu), sg, e0_, lo_t + 0.15*(hi_t - lo_t))
            assert th_t is not None, "no consistent choice of the signs for the other blocks with this base pattern"
            try: build_net(vertex_face_angles(sub_t), th_t, mm, nn_, (kk, nu), BASE_EDGE)
            except AssertionError as ex: raise AssertionError("the construction in space fails for this net (%s)" % (ex.args[0] if ex.args else ex,))
        except Exception as ex:
            par_msg.set_text("cannot apply: %s" % (ex.args[0] if ex.args else ex,)); par_msg.set_visible(True); fig.canvas.draw_idle(); return
        BASE_ANGLES_DEG, M_FACES, N_FACES, BASE_KN, ROW_SYSTEMS, BASE_SIGNS, E0 = angs, mm, nn_, (kk, nu), rows_ok, sg, e0_
        close_dihedrals(); plt.close(fig)
        state['rebuild'] = True
    b_apply.on_clicked(apply_params)
    # ---- the small buttons on the panels: (i) opens the explanations, (x) closes the panel
    def small(label, toggle=False, italic=False):
        b = FancyButton(fig, [0.4, 0.9, 0.022, 0.026], label, toggle=toggle, fontsize=8.5, rounding=0.3, italic=italic); b.ax.set_zorder(20)
        b.ax.set_visible(False); return b
    b_info = small('i', toggle=True, italic=True); b_x_rep, b_x_face = small('×'), small('×')
    def refresh_panels():
        report_text.set_text(report_string()); report_text.set_visible(state['report']); layout_panels(); fig.canvas.draw_idle()
    def measure(text):
        """window extent of a text with the best renderer at hand, never triggering a draw (a draw from inside a draw event freezes
        the interface): the renderer of the last draw event, else the canvas renderer."""
        r = state.get('renderer')
        try: return text.get_window_extent(r) if r is not None else text.get_window_extent()
        except Exception:
            try: return text.get_window_extent(fig.canvas.get_renderer())
            except Exception: return Bbox.from_extents(0, 0, 1, 1)
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
        for k, b in enumerate(btns):
            b.ax.set_position([x1 - bw - k*(bw + 0.05/W), y1 - bh - 0.005/H, bw, bh]); b.ax.set_visible(True)
    def layout_panels():
        """(x) and (i) in the top-right corners of the panels (over the short first lines only). The face panel sits under the
        report box (or to its right while the explanations are unrolled), and its plot under the face panel, with gaps and a plot
        size fixed in inches, so that nothing overlaps whatever the window size; the mouse wheel scrolls the whole left group."""
        W, H = fig.get_figwidth(), fig.get_figheight()
        xf, top = 0.012 - offs_x['panel'], 0.968 + offs['panel']
        report_text.set_position((xf, top))
        if report_text.get_visible():
            x1, y1 = corner_of(report_text)
            place_small([b_x_rep, b_info], report_text)
            if state['unrolled'] > 0: xf = x1 + 0.012                              # explanations unrolled: the face panel goes to the right (x1 includes the scroll)
            else: top = bottom_of(report_text)[1] - 0.25/H                         # otherwise under the report box
        else:
            b_x_rep.ax.set_visible(False); b_info.ax.set_visible(False)
        face_text.set_position((xf, top))
        if face_text.get_visible():
            place_small([b_x_face], face_text)
            x0, y0 = bottom_of(face_text)                                         # the plot goes right under the panel, fixed size in inches
            w, h = 2.3/W, 1.9/H
            inset.set_position([xf + 0.55/W, y0 - 0.55/H - h, w, h])
        else:
            b_x_face.ax.set_visible(False)
    state['layout_panels'] = layout_panels
    def after_draw(ev):
        """the (x)/(i) buttons and the plot are placed from the measured text boxes; after a resize the boxes are only known once the
        figure has been drawn, so their placement is corrected here (a second draw is requested only if something moved)."""
        if state.get('in_after_draw'): return                                   # never re-enter
        state['in_after_draw'] = True
        try:
            state['renderer'] = getattr(ev, 'renderer', None)
            arts = (inset, b_x_rep.ax, b_info.ax, b_x_face.ax)
            before = [tuple(a.get_position().bounds) for a in arts]
            layout_panels()
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
    def fname(): return "gqs_net_%dx%d_t%.3f" % (m, n, state['t'])
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
        with open(fn + ".obj", "w") as fh:
            idx = {}
            for k, (vv, p) in enumerate(sorted(state['pos'].items())):
                idx[vv] = k + 1; fh.write("v %.10f %.10f %.10f\n" % tuple(p))
            for a in range(m):
                for b in range(n):
                    fh.write("f %d %d %d %d\n" % tuple(idx[x] for x in [(a, b), (a+1, b), (a+1, b+1), (a, b+1)]))
        with open(fn + "_report.txt", "w") as fh:
            fh.write(verify(state['pos'], state['th'], state['t'], tex=False) + "\n\nEdge lengths:\n")
            for e in edges_all: P, Q = sorted(e); fh.write("  V%d%d-V%d%d: %.10f\n" % (P + Q + (edge_length(state['pos'], e),)))
            fh.write("\nFlat angles at inner vertices (deg), measured / prescribed:\n")
            for vv in sorted(ang):
                for f in sorted(ang[vv]): fh.write("  V%d%d in F%d%d: %.6f / %.6f\n" % (vv + f + (math.degrees(face_angle(state['pos'], vv, f)), math.degrees(ang[vv][f]))))
            fh.write("\nDihedral angles of the 3 x 3 blocks (deg):\n")
            for kn in sorted(state['th']): fh.write("  F%d%d: %s\n" % (kn + (str([round(math.degrees(state['th'][kn][i]), 6) for i in range(1, 5)]),)))
    b_svg.on_clicked(save_svg); b_png.on_clicked(save_png); b_obj.on_clicked(save_obj)
    fig._widgets = state['widgets'] = [sl, b_minus, b_plus, b_vert, b_face, b_edge, b_angl, b_hid, b_shad, b_rep, b_reset, b_dih, b_frz, b_par, b_col, *b_light.values(), *cdl['buttons'],
                                       b_svg, b_png, b_obj, b_info, b_x_rep, b_x_face]   # keep the widgets alive
    if STARTUP_ERROR[0] is not None:                                            # parameters of the file failed: show why, in the dialog
        set_dialog(True)
        par_msg.set_text("the parameters at the top of the file gave no net: " + STARTUP_ERROR[0].split("Change ROW_SYSTEMS")[0].split("Change the base")[0].replace("\n", " ").strip()); par_msg.set_visible(True)
        STARTUP_ERROR[0] = None
    if "--frames" in sys.argv:          # headless: save a few frames and exit (optional: --view ELEV,AZIM)
        if "--view" in sys.argv:
            el, az = map(float, sys.argv[sys.argv.index("--view") + 1].split(",")); ax.view_init(elev=el, azim=az); state['drawn'] = False
        for k, t in enumerate(np.linspace(lo, hi, 6)):
            cfg = configuration(float(t))
            if cfg is None: continue
            state['t'] = float(t); state['pos'] = cfg[0]; state['th'] = cfg[3]; state['info'] = (k == 1); state['unrolled'] = len(INFO_LINES) if k == 1 else 0
            state['labels'] = {'vertices': k == 0, 'faces': k == 0, 'edges': k == 0, 'angles': False}; redraw()
            fig.savefig("frame_%02d.png" % k, dpi=100)
        return
    # run the event loop as long as the main window exists: closing a secondary window (Dihedrals) must not end the program,
    # which plt.show() alone can do on some backends (macOS) when a window created during the loop is closed
    try:
        plt.show(block=False)
    except TypeError:
        plt.show()
    while plt.fignum_exists(fig.number) and 'agg' not in matplotlib.get_backend().lower().replace('macosx', ''):   # not for non-GUI backends
        try:
            fig.canvas.flush_events(); fig.canvas.start_event_loop(0.05)          # always on the main canvas (never on a closed one)
        except Exception: break
    return state.get('rebuild', False)

if __name__ == "__main__":
    while main(): pass
