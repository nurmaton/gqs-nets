# CalculiX (ccx 2.23) shell models of quad nets with offset hinge lines, rails, a spindle and point latches; snow loads on a
# cylindrical roof (EN 1991-1-3, 5.3.5); stress recovery of a steel/PU sandwich; a mechanism gate before every analysis.
# Requires numpy and scipy, and ccx 2.23 (the executable named by the environment variable CCX, else 'ccx' on the PATH).
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.
"""
USAGE
    import shellnet as S
    sec = S.panel_section()                      # 80 mm steel/PU sandwich as an equivalent shell with core shear
    M = S.NetModel(pos, m, n, k=12, section=sec, hinge_offset='outer', outer_sign=s, up=(0, 0, 1),
                   latches=[(edge, 0.5)])        # point latches only for the variant with a locked hinge
        pos: {(i, j): xyz} (metres), the OUTER skin of the net; outer_sign s = +1 if the corner-order normal
        (P11 - P00) x (P01 - P10) points outward (away from the panels), else -1.
    foot1 = S.lower_vertices(M, [(i, 0) for i in range(m + 1)])         # the lower zigzag vertices of a foot
    both feet anchored:  M.add_support(('vertices', foot1), [1, 2, 3], 'foot1'); the same for foot 2
    foot 2 on rails:     M.add_support(('vertices', foot2), [d_along, d_up], 'foot2')    # free across the hall
                         M.add_spindle(('vertex', va), ('vertex', vb), E=210e9, d=0.04, L=span)
    out = S.analyse_hall(M, workdir, span, rise, along=(0, 1, 0), s_k=1000.0, tag='name')
        -> mechanism gate (rigid-panel SVD and FE frequencies), snow (l_s, h/b, mu_3, totals), SLS = G + S and
           ULS = 1.35 G + 1.5 S for S1, S2+ (mu_3 on +x), S2-: displacement maxima and ratios to span/200, face von Mises,
           face compression against sigma_w, core shear, the same maxima outside r = 0.05 .. 0.2 m of every net vertex,
           hinge-line forces, reactions, foot thrust, spindle force, the capacity s_k,max per limit and per radius, and
           (buckle=True) *BUCKLE factors under ULS.
    Pieces: check_mechanisms, gravity_forces, snow_cylindrical, run_static (-> StaticResult, .combo), evaluate,
            capacity / capacity_by_radius, run_buckle, buckle_mode_info, hinge_line_resultant, spindle_forces.

NET, FACES, EDGES
    A net is given by its vertices pos[(i, j)], 0 <= i <= m, 0 <= j <= n; face F_ij has the corners (i,j), (i+1,j),
    (i+1,j+1), (i,j+1).  ('h', i, j) joins (i,j)-(i+1,j), shared by F_{i,j-1} and F_{i,j}; ('v', i, j) joins (i,j)-(i,j+1),
    shared by F_{i-1,j} and F_{i,j}.  An interior edge is a 'hinge' (default: the translations of the coincident nodes are
    tied by *EQUATION), a 'lock' (the coincident nodes merged: a rigid joint) or 'free'.  Every face is meshed alone into
    k x k S8R elements (k >= 2 for nets with hinges).  Face normal n_f = (P11 - P00) x (P01 - P10), normalised.

OFFSET HINGES (hinge_offset='outer')
    pos is the outer skin; the material lies on the inner side.  The shell nodes lie on pos, and *SHELL SECTION, OFFSET
    = outer_sign * e / t moves the material inside (e = 40 mm, half the panel thickness).  The equivalent sandwich shell is
    t_eq = 137.6 mm thick; only its stiffness about the panel mid-surface matters.  With an offset alone, CalculiX links a
    shell node to the mean of its expanded nodes (the mid-surface), so a tie would hinge each face about its own mid-surface
    line.  Hence every tied node is made a "knot" (a rigid body of its expanded nodes, reference point on the outer surface):
    each face side along a hinge, and every supported, spindle or add_knots node, gets a dummy B32 beam (E = 1e-6 E) on its
    existing nodes, which adds no stiffness.  A node merged between faces folded by MERGE_BEAM_MAX_FOLD = 20 deg or more is a
    knot by itself and gets no beam.  *EQUATION, *BOUNDARY and *CLOAD on a knot act at the reference point; the printed
    displacement of a knot node is the mean of its expanded nodes.

SUPPORTS, RAILS, SPINDLE, LATCH
    add_support(target, dofs, label): dofs in global axes (1, 2, 3 = x, y, z); rails fix only the along and vertical dofs.
    add_spindle(target_a, target_b, E, d, L): SPRINGA (k = E pi d^2/4 / L) between two plain nodes, each tied to its target
    (('vertex', v), or ('vertices', [...]) through the mean of their masters).  ccx evaluates the SPRINGA force with the
    deformed length even in a linear step: StaticResult uses the linear force k (u_b - u_a).e, and run_static runs one ccx
    job per load case when there is a spindle.  latches=[(edge, s)]: the coincident node pair nearest to the fraction s of the
    hinge edge is joined rigidly ('rottie': the two knots keep their own faces and normals; their rotations are tied by
    *EQUATION).  add_dof_tie(target, dofs): optional stiff carriage (the dofs of the targets move together).

MECHANISM GATE
    rigid_panel_mechanisms(model): every face a rigid body; hinges, locks, latches, supports and the spindle as velocity
    constraints; zero singular values (s < 1e-9 s_max) = mechanisms.  check_mechanisms(workdir) also runs *FREQUENCY (zero =
    |lambda| < 100 x the rigid-mode noise of a free single face).  run_static and run_buckle refuse unless both counts are 0.

LOADS
    up (unit vector): gravity and snow act along -up.  gravity_forces: consistent rho g t int N dA (12.57 kg/m2 for the
    panel).  snow_cylindrical(model, s_k, case, along): vertical rays on a 1 cm plan grid hit the topmost face only; cells
    whose topmost face has its outer normal pointing down get no snow; the cell load mu s_k dA acts at the hit point and is
    distributed with the S8R shape functions.  Smooth roof: the upper concave hull of the ray-cast top surface over the hall
    length, closed at the lowest net point, Gaussian-smoothed over 0.15 m; beta(x) = its slope; l_s = the plan interval around
    the crown with beta <= 60 deg (mu = 0 outside); h = rise of the smooth roof, b = plan width; mu_3 = min(0.2 + 10 h/b, 2.0).
    'S1' undrifted: mu = 0.8 on l_s.  'S2' drifted ('triangles'): two triangles of base l_s/2, 0 at the ends and the middle
    of l_s, peaks 0.5 mu_3 at l_s/4 and mu_3 at 3 l_s/4; drift=+1 puts mu_3 on the +x side.

SANDWICH STRESSES
    sandwich_equivalent(...) with G_c (kappa = 1).  face_and_core: from the expanded-node stresses (frd, mapped by position)
    at the two surfaces of the equivalent shell: N = (s_top + s_bot)/2 t_eq, M = (s_top - s_bot)/2 t_eq^2/6, face stress
    N/(2 t_f) +- M/(d t_f); core shear tau_c = Q/d.  sigma_w = 0.5 (E_f E_c G_c)^(1/3) = 71.6 MPa.
    Panel corners, point supports and point latches are stress singularities: the maxima are also reported outside a radius
    r = 0.05 .. 0.2 m of every net vertex ('away_from_vertices').

CALCULIX NOTES (ccx 2.23)
    numeric fields longer than 20 characters are truncated (every number goes through _num, 13 digits); a spring constant
    without a decimal point is read as a dof number (written with %e); *STATIC on a mechanism gives no error (hence the
    gate); *BUCKLE accuracy 1e-7 (the default 0.01 can return a higher first factor and skip clustered modes); ccx *BUCKLE
    returns the factors nearest to 1, not the lowest ones: scale the load to reach factors below about 0.5.

Units: SI.
"""
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np

CCX = os.environ.get('CCX', 'ccx')
G_ACC = 9.81
KAPPA_C3D20R = 1.0       # effective transverse-shear factor of the expanded S8R shell (calibrated against sandwich theory)
MERGE_BEAM_MAX_FOLD = 20.0   # deg: ccx makes a node shared by two folded shells a knot by itself
                             # exactly from a 20.0 deg fold (measured: 19.99 no knot, 20.00 knot); below that a dummy beam
                             # is needed for a SUPPORTED/tied merged node (lock 17 % too soft without it), at and above
                             # 20 deg a beam breaks the knot (lock 3.2x too soft) -> beams on lock nodes iff fold < 20


# ---------------------------------------------------------------- sections
@dataclass
class Section:
    t: float
    E: float
    nu: float
    rho: float
    G_t: float = None
    E3: float = None
    info: dict = field(default_factory=dict)

    @property
    def orthotropic(self): return self.G_t is not None

    @property
    def D(self): return self.E*self.t**3/(12*(1 - self.nu**2))

    @property
    def mass_per_area(self): return self.rho*self.t

    @property
    def e_ref(self):
        """distance from the outer surface of the REAL panel to its mid-surface"""
        return self.info.get('h_total', self.t)/2


def isotropic(E, nu, rho, t):
    return Section(t=t, E=E, nu=nu, rho=rho, info={'kind': 'isotropic', 'h_total': t})


def sandwich_equivalent(E_f, nu_f, rho_f, t_f, c, rho_c=0.0, G_c=None, kappa=None, E3=None, E_c=None, tau_c_max=None,
                        f_y=None, gamma_M=None):
    """equivalent homogeneous shell of a symmetric sandwich (faces t_f, core c): same A, D, mass/area and (with G_c) core
    shear stiffness S = G_c d^2/c."""
    d = c + t_f
    A = 2*E_f*t_f/(1 - nu_f**2)
    D = E_f/(1 - nu_f**2)*(t_f*d**2/2 + t_f**3/6)
    t_eq = math.sqrt(3*d**2 + t_f**2)
    E_eq = 2*E_f*t_f/t_eq
    rho_eq = (2*rho_f*t_f + rho_c*c)/t_eq
    info = dict(kind='sandwich_equivalent', E_f=E_f, nu_f=nu_f, rho_f=rho_f, t_f=t_f, c=c, rho_c=rho_c, G_c=G_c, d=d,
                A=A, D=D, t_eq=t_eq, E_eq=E_eq, rho_eq=rho_eq, mass_per_area=2*rho_f*t_f + rho_c*c, h_total=c + 2*t_f,
                E_c=E_c, tau_c_max=tau_c_max, f_y=f_y, gamma_M=gamma_M)
    if E_c is not None and G_c is not None:
        info['sigma_w'] = 0.5*(E_f*E_c*G_c)**(1/3)
    G_t = None
    if G_c is not None:
        kappa = KAPPA_C3D20R if kappa is None else kappa
        S = G_c*d**2/c
        G_t = S/(kappa*t_eq)
        info.update(S=S, kappa=kappa, G_t=G_t)
    return Section(t=t_eq, E=E_eq, nu=nu_f, rho=rho_eq, G_t=G_t, E3=(E3 if E3 is not None else E_eq), info=info)


def panel_section():
    """the 80 mm sandwich panel: steel faces 0.6 mm (210 GPa, 0.3, 7850, f_y 280 MPa, gamma_M 1.1), PU core 78.8 mm
    (E_c 4 MPa, G_c 3.5 MPa, 40 kg/m3, core shear strength 0.10 MPa).  12.57 kg/m2, sigma_w = 71.6 MPa."""
    return sandwich_equivalent(210e9, 0.3, 7850.0, 0.6e-3, 78.8e-3, rho_c=40.0, G_c=3.5e6, E_c=4.0e6, tau_c_max=0.10e6,
                               f_y=280e6, gamma_M=1.1)


def sandwich_face_stress(sig_top_eq, sig_bot_eq, sec):
    """face stresses from the surface stresses of the equivalent shell (any in-plane component)"""
    i = sec.info
    st, sb = np.asarray(sig_top_eq, float), np.asarray(sig_bot_eq, float)
    N = (st + sb)/2*i['t_eq']
    M = (st - sb)/2*i['t_eq']**2/6
    return N/(2*i['t_f']) + M/(i['d']*i['t_f']), N/(2*i['t_f']) - M/(i['d']*i['t_f'])


# ---------------------------------------------------------------- small helpers
class _UF:
    def __init__(self): self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        r = x
        while self.p[r] != r: r = self.p[r]
        while self.p[x] != r: self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra: ra, rb = rb, ra
            self.p[rb] = ra


def _shape(elem, xi, eta):
    """shape functions in CalculiX node order (corners (-1,-1),(1,-1),(1,1),(-1,1), then S8R mid-sides 12,23,34,41);
    xi, eta scalars or arrays (result shape (nn,) + xi.shape)"""
    xi, eta = np.asarray(xi, float), np.asarray(eta, float)
    if elem == 'S4':
        return 0.25*np.array([(1 - xi)*(1 - eta), (1 + xi)*(1 - eta), (1 + xi)*(1 + eta), (1 - xi)*(1 + eta)])
    c = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    N = [0.25*(1 + xi*a)*(1 + eta*b)*(xi*a + eta*b - 1) for a, b in c]
    N += [0.5*(1 - xi**2)*(1 - eta), 0.5*(1 + xi)*(1 - eta**2), 0.5*(1 - xi**2)*(1 + eta), 0.5*(1 - xi)*(1 - eta**2)]
    return np.array(N)


_GP3 = [(-math.sqrt(0.6), 5/9), (0.0, 8/9), (math.sqrt(0.6), 5/9)]


def _elem_load_integrals(elem, X):
    """int N_i dA over a flat element with corner coordinates X[0..3] (bilinear geometry)"""
    out = np.zeros(4 if elem == 'S4' else 8)
    for xi, wx in _GP3:
        for eta, wy in _GP3:
            dxi = 0.25*(-(1 - eta)*X[0] + (1 - eta)*X[1] + (1 + eta)*X[2] - (1 + eta)*X[3])
            deta = 0.25*(-(1 - xi)*X[0] - (1 + xi)*X[1] + (1 + xi)*X[2] + (1 - xi)*X[3])
            out += wx*wy*np.linalg.norm(np.cross(dxi, deta))*_shape(elem, xi, eta)
    return out


def en1991_mu1(alpha_deg):
    """EN 1991-1-3 Table 5.2 mu_1 (not used for curved roofs: see snow_cylindrical)"""
    a = abs(alpha_deg)
    return 0.8 if a <= 30 else (0.8*(60 - a)/30 if a < 60 else 0.0)


def _skew(x): return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def _num(x, sig=13):
    """number for a CalculiX data line (ccx 2.23 truncates fields longer than 20 characters silently)"""
    x = float(x)
    if abs(x) < 1e-14: x = 0.0
    out = ('%%.%dg' % sig) % x
    assert len(out) <= 20, out
    return out


def _chunks(ids, per=12):
    return [', '.join(map(str, ids[q:q + per])) + ',' for q in range(0, len(ids), per)]


# ---------------------------------------------------------------- the model
class NetModel:
    def __init__(self, pos, m, n, k=4, elem='S8R', section=None, connections=None, default='hinge', lock_mode='merge',
                 tol=1e-7, name='net', up=(0.0, 0.0, 1.0), hinge_offset=None, outer_sign=+1, latches=(),
                 knot_beam=True, latch_mode='rottie'):
        """hinge_offset: None (reference surface = mid-surface) or 'outer' (pos = outer skin, material
        inside, knots at tied nodes).  outer_sign: +1 if the corner-order normal n_f points outward.  up: unit vector.
        latches: [(edge, s)] point latches (merge the coincident node pair nearest to fraction s of the edge)."""
        assert elem in ('S8R', 'S4')
        assert lock_mode == 'merge', "only lock_mode 'merge' is supported"
        assert hinge_offset in (None, 'outer')
        assert outer_sign in (1, -1)
        assert latch_mode in ('rottie', 'merge')
        assert knot_beam or latch_mode == 'merge' or not latches, "latch_mode 'rottie' needs knots (knot_beam=True)"
        self.latch_mode = latch_mode
        self.m, self.n, self.elem, self.lock_mode, self.name = m, n, elem, lock_mode, name
        self.ka, self.kb = (k, k) if isinstance(k, int) else tuple(k)
        self.pos = {key: np.asarray(v, float) for key, v in pos.items()}
        self.section = section if section is not None else isotropic(210e9, 0.3, 7850.0, 0.01)
        self.up = np.asarray(up, float)
        assert abs(np.linalg.norm(self.up) - 1) < 1e-9, "up must be a unit vector"
        self.hinge_offset, self.outer_sign, self.knot_beam = hinge_offset, outer_sign, knot_beam
        self.conn = {e: default for e in self.interior_edges()}
        if connections:
            for e, v in connections.items():
                assert e in self.conn, "not an interior edge: %r" % (e,)
                assert v in ('hinge', 'lock', 'free'), v
                self.conn[e] = v
        if any(c in ('hinge', 'free') for c in self.conn.values()):
            assert min(self.ka, self.kb) >= 2, "k >= 2 is required for nets with hinges or slits (k = 1 gives " \
                                               "spurious modes)"
        self.latches = []
        for lt in latches:
            e, s = lt[0], lt[1]
            length = float(lt[2]) if len(lt) > 2 else 0.0
            assert self.conn.get(e) == 'hinge', "a latch must sit on a hinge edge: %r" % (e,)
            self.latches.append((e, float(s), length))
        self.tol = tol
        self.supports = []           # (nodes, dofs, label)
        self.springs = []            # dict(a=master, b=master, k=..., label=..., E, A, L)
        self.mech = None             # result of check_mechanisms
        self._build()

    # ---- offsets
    @property
    def offset_fraction(self):
        """the *SHELL SECTION OFFSET (units of t): reference surface = outer skin, e_ref above the mid-surface"""
        if self.hinge_offset is None: return 0.0
        return self.outer_sign*self.section.e_ref/self.section.t

    def mid_point(self, node, f):
        """mid-surface point of the fiber at node (reference point on pos) for face f"""
        return self.xyz[node] - self.offset_fraction*self.section.t*self.face_normal(f)

    # ---- topology
    def faces(self): return [(i, j) for i in range(self.m) for j in range(self.n)]

    def edge_faces(self, e):
        t, i, j = e
        if t == 'h': cand = [((i, j - 1), 'top'), ((i, j), 'bottom')]
        else: cand = [((i - 1, j), 'right'), ((i, j), 'left')]
        return [(f, s) for f, s in cand if 0 <= f[0] < self.m and 0 <= f[1] < self.n]

    def all_edges(self):
        return [('h', i, j) for i in range(self.m) for j in range(self.n + 1)] + \
               [('v', i, j) for i in range(self.m + 1) for j in range(self.n)]

    def interior_edges(self): return [e for e in self.all_edges() if len(self.edge_faces(e)) == 2]

    def boundary_edges(self): return [e for e in self.all_edges() if len(self.edge_faces(e)) == 1]

    def edge_ends(self, e):
        t, i, j = e
        return ((i, j), (i + 1, j)) if t == 'h' else ((i, j), (i, j + 1))

    def face_normal(self, f):
        i, j = f
        P = [self.pos[(i, j)], self.pos[(i + 1, j)], self.pos[(i + 1, j + 1)], self.pos[(i, j + 1)]]
        nv = np.cross(P[2] - P[0], P[3] - P[1])
        return nv/np.linalg.norm(nv)

    def face_center(self, f):
        i, j = f
        return np.mean([self.pos[(i, j)], self.pos[(i + 1, j)], self.pos[(i + 1, j + 1)], self.pos[(i, j + 1)]], 0)

    # ---- mesh
    def _build(self):
        S8 = self.elem == 'S8R'
        Na, Nb = (2*self.ka, 2*self.kb) if S8 else (self.ka, self.kb)
        raw_xyz, raw_face = [None], [None]
        self.face_grid, self.raw_elems, self.side_raw = {}, [], {}
        self.face_elem = {}          # (f, ea, eb) -> index into self.elems
        self.planarity = 0.0
        for f in self.faces():
            i, j = f
            P00, P10, P11, P01 = self.pos[(i, j)], self.pos[(i + 1, j)], self.pos[(i + 1, j + 1)], self.pos[(i, j + 1)]
            nrm = self.face_normal(f)
            C = (P00 + P10 + P11 + P01)/4
            self.planarity = max(self.planarity, max(abs(np.dot(P - C, nrm)) for P in (P00, P10, P11, P01)))
            g = {}
            for a in range(Na + 1):
                for b in range(Nb + 1):
                    if S8 and a % 2 == 1 and b % 2 == 1: continue
                    s, u = a/Na, b/Nb
                    X = (1 - s)*(1 - u)*P00 + s*(1 - u)*P10 + s*u*P11 + (1 - s)*u*P01
                    raw_xyz.append(X); raw_face.append(f); g[(a, b)] = len(raw_xyz) - 1
            self.face_grid[f] = g
            st = 2 if S8 else 1
            for ea in range(0, Na, st):
                for eb in range(0, Nb, st):
                    if S8:
                        loc = [(ea, eb), (ea + 2, eb), (ea + 2, eb + 2), (ea, eb + 2),
                               (ea + 1, eb), (ea + 2, eb + 1), (ea + 1, eb + 2), (ea, eb + 1)]
                    else:
                        loc = [(ea, eb), (ea + 1, eb), (ea + 1, eb + 1), (ea, eb + 1)]
                    self.face_elem[(f, ea//st, eb//st)] = len(self.raw_elems)
                    self.raw_elems.append((f, [g[x] for x in loc]))
            self.side_raw[(f, 'bottom')] = [g[(a, 0)] for a in range(Na + 1)]
            self.side_raw[(f, 'top')] = [g[(a, Nb)] for a in range(Na + 1)]
            self.side_raw[(f, 'left')] = [g[(0, b)] for b in range(Nb + 1)]
            self.side_raw[(f, 'right')] = [g[(Na, b)] for b in range(Nb + 1)]
        raw_xyz = np.array([np.zeros(3)] + raw_xyz[1:])
        self.edge_pairs = {}
        worst = 0.0
        for e in self.interior_edges():
            (f1, s1), (f2, s2) = self.edge_faces(e)
            L1, L2 = self.side_raw[(f1, s1)], self.side_raw[(f2, s2)]
            assert len(L1) == len(L2)
            for a, b in zip(L1, L2): worst = max(worst, np.linalg.norm(raw_xyz[a] - raw_xyz[b]))
            self.edge_pairs[e] = list(zip(L1, L2))
        self.coincidence_error = worst
        assert worst < self.tol, "nodes on shared edges do not coincide (%.3g)" % worst
        uf = _UF()
        for r in range(1, len(raw_xyz)): uf.find(r)
        for e, c in self.conn.items():
            if c == 'lock':
                for a, b in self.edge_pairs[e]: uf.union(a, b)
        self.latch_raw = []
        for e, s, length in self.latches:
            pairs = self.edge_pairs[e]
            npair = len(pairs)
            a_, b_ = self.edge_ends(e)
            Le = np.linalg.norm(self.pos[b_] - self.pos[a_])
            q0 = int(round(s*(npair - 1)))
            qs = [q for q in range(npair) if abs(q/(npair - 1) - q0/(npair - 1))*Le <= length/2 + 1e-12] or [q0]
            assert min(qs) > 0 and max(qs) < npair - 1, "a latch must not reach a vertex of the edge"
            for q in qs:
                if self.latch_mode == 'merge': uf.union(*pairs[q])
                self.latch_raw.append((e, q, pairs[q]))
        roots = sorted({uf.find(r) for r in range(1, len(raw_xyz))})
        newid = {r: q + 1 for q, r in enumerate(roots)}
        self.canon = {r: newid[uf.find(r)] for r in range(1, len(raw_xyz))}
        self.xyz = {newid[r]: raw_xyz[r] for r in roots}
        self.node_faces = {}
        for r in range(1, len(raw_xyz)): self.node_faces.setdefault(self.canon[r], set()).add(raw_face[r])
        self.elems = [(f, [self.canon[r] for r in con]) for f, con in self.raw_elems]
        self.latch_nodes = [self.canon[p[0]] for e, q, p in self.latch_raw]
        # 'rottie' latch = the two coincident (hinge-tied) knot nodes keep their own faces and normals and
        # get their ROTATIONS tied by *EQUATION (dofs 4-6 of the knots); 'merge' = the old merged node
        self.latch_pairs = [(self.canon[p[0]], self.canon[p[1]]) for e, q, p in self.latch_raw]
        self.latch_all_nodes = sorted({x for pr in self.latch_pairs for x in pr})
        ut = _UF()
        for nd in self.xyz: ut.find(nd)
        for e, c in self.conn.items():
            if c == 'free': continue
            for a, b in self.edge_pairs[e]:
                ut.union(self.canon[a], self.canon[b])
        self.tmaster = {nd: ut.find(nd) for nd in self.xyz}
        self.vertex_nodes = {}
        for f in self.faces():
            i, j = f
            g = self.face_grid[f]
            for (a, b), v in (((0, 0), (i, j)), ((Na, 0), (i + 1, j)), ((Na, Nb), (i + 1, j + 1)), ((0, Nb), (i, j + 1))):
                self.vertex_nodes.setdefault(v, set()).add(self.canon[g[(a, b)]])
        self.Na, self.Nb = Na, Nb
        self.n_shell_nodes = max(self.xyz)

    # ---- node selections
    def vertex_master(self, v):
        return min(self.tmaster[x] for x in self.vertex_nodes[v])

    def face_nodes(self, f): return sorted({self.canon[r] for r in self.face_grid[f].values()})

    def edge_nodes(self, e):
        out = []
        for f, s in self.edge_faces(e): out += [self.canon[r] for r in self.side_raw[(f, s)]]
        return sorted(set(out))

    def side_nodes(self, f, side): return [self.canon[r] for r in self.side_raw[(f, side)]]

    def _resolve(self, target):
        kind, what = target
        if kind == 'edge': return self.edge_nodes(what)
        if kind == 'vertex': return sorted(self.vertex_nodes[what])
        if kind == 'vertices': return sorted({x for v in what for x in self.vertex_nodes[v]})
        if kind == 'nodes': return list(what)
        raise ValueError(kind)

    def group(self, master):
        """all nodes (shell and spindle end nodes) whose translations are tied to master"""
        if not hasattr(self, '_groups') or self._groups_n != len(self.tmaster):
            self._groups = {}
            for nd, mm in self.tmaster.items(): self._groups.setdefault(mm, []).append(nd)
            self._groups_n = len(self.tmaster)
        return self._groups.get(master, [master])

    # ---- boundary conditions, spindle
    def add_support(self, target, dofs, label=None):
        """dofs: global translations 1, 2, 3; rotations 4-6 (a clamp; with offsets the node is a knot, so they act on its rigid fiber)"""
        dofs = tuple(sorted(dofs))
        self.supports.append((self._resolve(target), dofs, label or str(target)))
        self.mech = None

    def add_knots(self, target):
        """make the nodes of target knots (loads/ties there act at the outer reference point), no constraint"""
        self.extra_knots = getattr(self, 'extra_knots', set()) | set(self._resolve(target))

    def add_spindle(self, target_a, target_b, E=210e9, d=0.04, L=None, label='spindle'):
        """axial spring (SPRINGA, k = E (pi d^2/4) / L) between two plain nodes.  Each end is tied to its target:
        ('vertex', v) or ('nodes', [n]) -> rigidly to that translation master; ('vertices', [v1, v2, ...]) -> to the MEAN
        of the masters by a distributing *EQUATION (u_end = sum u_i / n: the end force is shared equally, a foot plate or
        rail carriage spanning these vertices).  L = distance of the two ends unless given.
        Single-vertex attachment concentrates the whole spindle force at one node: *BUCKLE then finds a local,
        mesh-dependent mode there (its factor falls with k); spread it over a few foot vertices."""
        def ends(target):
            if target[0] == 'vertices' and len(target[1]) > 1:
                ms = sorted({self.vertex_master(v) for v in target[1]})
            else:
                ms = [min(self.tmaster[x] for x in self._resolve(target))]
            return [(m_, 1.0/len(ms)) for m_ in ms]
        pa, pb = ends(target_a), ends(target_b)
        xa = sum(w_*self.xyz[m_] for m_, w_ in pa); xb = sum(w_*self.xyz[m_] for m_, w_ in pb)
        dist = float(np.linalg.norm(xb - xa))
        L = dist if L is None else float(L)
        A = math.pi*d**2/4
        self.springs.append(dict(a=pa[0][0], b=pb[0][0], pa=pa, pb=pb, xa=xa, xb=xb, k=E*A/L, E=E, A=A, L=L, dist=dist,
                                 label=label))
        self.mech = None

    def add_dof_tie(self, target, dofs, label='tie'):
        """tie global translation dofs of all translation masters of target to the first one (a stiff foot beam / rail
        carriage in that direction).  Optional extra assumption, e.g. dofs [1] on the foot-2 vertices: the rails stay
        free across but move together; without it *BUCKLE finds a mesh-dependent local
        mode at the free-across foot edge (N_cr 0.74 -> 0.43 -> 0.30 of Engesser for k = 4, 8, 12), with it 0.974."""
        ms = sorted({self.tmaster[x] for x in self._resolve(target)})
        self.dof_ties = getattr(self, 'dof_ties', []) + [(ms, tuple(sorted(dofs)), label)]
        self.mech = None

    def _distributed_rf(self, RF):
        """{master: force} that distributing spindle ends pass to their masters (not in any RF print)"""
        out = {}
        for sp in getattr(self, 'springs', []):
            for key, node in (('pa', 'na'), ('pb', 'nb')):
                if len(sp[key]) > 1 and sp.get(node) in RF:
                    for m_, w_ in sp[key]:
                        out[m_] = out.get(m_, np.zeros(3)) + w_*np.asarray(RF[sp[node]], float)
        return out

    # ---- loads (dict {node: vector}; nodes are mapped to translation masters when written)
    def total_area(self, projected=False):
        A = 0.0
        for f, con in self.elems:
            a = _elem_load_integrals(self.elem, [self.xyz[c] for c in con[:4]]).sum()
            A += a*(abs(np.dot(self.face_normal(f), self.up)) if projected else 1.0)
        return A

    # ---- knots (offset ties)
    def _knot_beams(self):
        """dummy 1-D elements forcing knots: both sides of every hinge edge (incl. latched edges), plus one element side
        at every supported / spindle / add_knots node not covered.  Nodes shared by FOLDED faces (merged by locks or
        latches, normals >= MERGE_BEAM_MAX_FOLD apart) are NEVER put in a beam: tests
        showed that such a node in a beam stops acting as a joint (point latch lambda 0.33 instead of 3.6e3; a locked
        20-30 deg fold becomes a mechanism), while for folds <= 15 deg the beam changes nothing (ratio 1.00000); a
        merged node at a fold >= ~20 deg is a knot by itself (ccx), below that ccx averages the normals.
        A 3-node segment that loses such a node becomes a 2-node B31.  Returns {face: [segment, ...]}"""
        if not self.knot_beam: return {}          # knots also for hinge_offset=None (centric)
        multi = set()
        for nd, fs in self.node_faces.items():
            if len(fs) < 2: continue
            ns = [self.face_normal(f) for f in fs]
            ang = max(math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b)))))) for a in ns for b in ns)
            if ang >= MERGE_BEAM_MAX_FOLD: multi.add(nd)
        if self.latch_mode == 'merge':
            multi.update(self.latch_nodes)        # a merged latch node is never put in a beam
        self.folded_merged_nodes = sorted(multi)
        out, covered = {}, set()
        st = 2 if self.elem == 'S8R' else 1

        def add(f, seg):
            seg = tuple(x for x in seg if x not in multi)
            if len(seg) < 2: return
            out.setdefault(f, []).append(seg); covered.update(seg)
        for e, c in self.conn.items():
            if c != 'hinge': continue
            for f, s in self.edge_faces(e):
                side = self.side_nodes(f, s)
                for a in range(0, len(side) - 1, st):
                    add(f, side[a:a + st + 1])
        need = set(getattr(self, 'extra_knots', set()))
        for nodes, dofs, lab in self.supports: need.update(nodes)
        for sp in self.springs:
            for m_, w_ in sp['pa'] + sp['pb']: need.update(self.group(m_))
        need = {x for x in need if x <= self.n_shell_nodes and x not in multi}
        self.unknotted = []          # supported interior face nodes: no side to hang a beam on -> constraint at mid-surface
        for nd in sorted(need - covered):
            if nd in covered: continue
            done = False
            for f in sorted(self.node_faces[nd]):
                for s in ('bottom', 'top', 'left', 'right'):
                    side = self.side_nodes(f, s)
                    if nd in side:
                        q = side.index(nd)
                        a = min((q//st)*st, len(side) - 1 - st)
                        add(f, side[a:a + st + 1]); done = nd in covered
                        if done: break
                if done: break
            if not done: self.unknotted.append(nd)
        if self.latch_mode == 'rottie' and self.latch_pairs:
            miss = [x for x in self.latch_all_nodes if x not in covered]
            assert not miss, "rottie latch nodes must be knots (in a dummy beam): %s" % miss
        return out

    # ---- input file
    def write_inp(self, path, analysis='static', cases=None, nmodes=10, freq_shift=None, buckle_accuracy=None,
                  title=None, print_rf=True):
        """analysis 'static' (cases = ordered dict name -> forces, one *STEP each), 'frequency', 'buckle' (cases = one
        forces dict, or {name: forces} with one entry)"""
        sec = self.section
        L = []
        w = L.append
        w('*HEADING')
        w((title or 'shellnet %s: %d x %d faces, %s, k = %dx%d, offset %s' % (
            self.name, self.m, self.n, self.elem, self.ka, self.kb, self.hinge_offset))[:120])
        # spindle plain nodes
        nid = self.n_shell_nodes
        self.spindle_nodes = []
        for sp in self.springs:
            sp['na'], sp['nb'] = nid + 1, nid + 2
            nid += 2
            self.spindle_nodes += [sp['na'], sp['nb']]
        self.tmaster_all = dict(self.tmaster)
        for sp in self.springs:
            if len(sp['pa']) == 1: self.tmaster_all[sp['na']] = sp['pa'][0][0]
            if len(sp['pb']) == 1: self.tmaster_all[sp['nb']] = sp['pb'][0][0]
        w('*NODE, NSET=NALL')
        for nd in sorted(self.xyz):
            x = self.xyz[nd]
            w('%d, %s, %s, %s' % (nd, _num(x[0]), _num(x[1]), _num(x[2])))
        for sp in self.springs:
            for q, x in ((sp['na'], sp['xa']), (sp['nb'], sp['xb'])):
                w('%d, %s, %s, %s' % (q, _num(x[0]), _num(x[1]), _num(x[2])))
        w('*ELEMENT, TYPE=%s, ELSET=EALL' % self.elem)
        self.elem_ids = {}
        for q, (f, con) in enumerate(self.elems, 1):
            w('%d, ' % q + ', '.join(map(str, con)))
            self.elem_ids.setdefault(f, []).append(q)
        eid = len(self.elems)
        for f, ids in self.elem_ids.items():
            w('*ELSET, ELSET=F_%d_%d' % f)
            L.extend(_chunks(ids))
        beams = self._knot_beams()
        self.n_knot_beams = sum(len(v) for v in beams.values())
        self.knot_nodes = sorted({x for v in beams.values() for seg in v for x in seg} | set(self.latch_nodes))
        self.beam_sets = []
        for f, segs in sorted(beams.items()):
            for nn, btype, pre in ((3, 'B32', 'KB'), (2, 'B31', 'KC')):
                ss = [sg for sg in segs if len(sg) == nn]
                if not ss: continue
                w('*ELEMENT, TYPE=%s, ELSET=%s_%d_%d' % ((btype, pre) + f))
                self.beam_sets.append(('%s_%d_%d' % ((pre,) + f), f))
                for seg in ss:
                    eid += 1
                    w('%d, ' % eid + ', '.join(map(str, seg)))
        for q, sp in enumerate(self.springs):
            eid += 1
            sp['elem'] = eid
            w('*ELEMENT, TYPE=SPRINGA, ELSET=SPR%d' % q)
            w('%d, %d, %d' % (eid, sp['na'], sp['nb']))
        verts = sorted({self.vertex_master(v) for v in self.vertex_nodes})
        w('*NSET, NSET=NVERT')
        L.extend(_chunks(verts))
        if self.spindle_nodes:
            w('*NSET, NSET=NSPIN')
            L.extend(_chunks(self.spindle_nodes))
        # materials
        w('*MATERIAL, NAME=SHELLMAT')
        if sec.orthotropic:
            G12 = sec.E/(2*(1 + sec.nu))
            w('*ELASTIC, TYPE=ENGINEERING CONSTANTS')
            w('%.10g, %.10g, %.10g, %.10g, 0., 0., %.10g, %.10g,' % (sec.E, sec.E, sec.E3, sec.nu, G12, sec.G_t))
            w('%.10g' % sec.G_t)
        else:
            w('*ELASTIC')
            w('%.10g, %.10g' % (sec.E, sec.nu))
        w('*DENSITY')
        w('%.10g' % sec.rho)
        off = self.offset_fraction
        offs = (', OFFSET=%.10g' % off) if off != 0.0 else ''
        if sec.orthotropic:
            for f in self.faces():
                i, j = f
                a = self.pos[(i + 1, j)] - self.pos[(i, j)]
                b = self.pos[(i, j + 1)] - self.pos[(i, j)]
                w('*ORIENTATION, NAME=OR_%d_%d, SYSTEM=RECTANGULAR' % f)
                w(', '.join(_num(x) for x in tuple(a) + tuple(b)))
                w('*SHELL SECTION, ELSET=F_%d_%d, MATERIAL=SHELLMAT, ORIENTATION=OR_%d_%d%s' % (f + f + (offs,)))
                w('%.10g' % sec.t)
        else:
            w('*SHELL SECTION, ELSET=EALL, MATERIAL=SHELLMAT%s' % offs)
            w('%.10g' % sec.t)
        if beams:
            w('*MATERIAL, NAME=KNOTMAT')
            w('*ELASTIC')
            w('%.6e, 0.3' % (1e-6*sec.E))
            w('*DENSITY')
            w('%.6e' % (1e-9*sec.rho))
            # beam cross-section: small against the element size
            le = min(np.linalg.norm(self.xyz[s[-1]] - self.xyz[s[0]]) for segs in beams.values() for s in segs)
            bdim = max(1e-4, 0.05*le)
            for name, f in self.beam_sets:
                nf = self.face_normal(f)
                w('*BEAM SECTION, ELSET=%s, MATERIAL=KNOTMAT, SECTION=RECT' % name)
                w('%.6g, %.6g' % (bdim, bdim))
                w(', '.join(_num(x) for x in nf))
        for q, sp in enumerate(self.springs):
            w('*SPRING, ELSET=SPR%d' % q)
            w('')
            w('%.12e' % sp['k'])
        # equations: translations to masters (shell groups and spindle end nodes)
        eqs = []
        for nd in sorted(self.tmaster_all):
            mm = self.tmaster_all[nd]
            if mm != nd: eqs += [(nd, d, mm) for d in (1, 2, 3)]
        dist_eqs = []
        for sp in self.springs:
            for key, node in (('pa', 'na'), ('pb', 'nb')):
                if len(sp[key]) > 1:
                    for d in (1, 2, 3):
                        dist_eqs.append([(sp[node], d, 1.0)] + [(m_, d, -w_) for m_, w_ in sp[key]])
        for ms, dofs, lab in getattr(self, 'dof_ties', []):
            for m_ in ms[1:]:
                for d in dofs:
                    dist_eqs.append([(m_, d, 1.0), (ms[0], d, -1.0)])
        if self.latch_mode == 'rottie':
            for a_, b_ in self.latch_pairs:
                for d in (4, 5, 6):
                    dist_eqs.append([(b_, d, 1.0), (a_, d, -1.0)])
        if eqs or dist_eqs:
            w('*EQUATION')
            for s, d, mm in eqs:
                w('2')
                w('%d, %d, 1., %d, %d, -1.' % (s, d, mm, d))
            for terms in dist_eqs:
                w('%d' % len(terms))
                for q0 in range(0, len(terms), 4):
                    w(', '.join('%d, %d, %s' % (n_, d_, _num(c_) if '.' in _num(c_) or 'e' in _num(c_) else _num(c_) + '.')
                                for n_, d_, c_ in terms[q0:q0 + 4]))
        self.n_equations = len(eqs) + len(dist_eqs)
        # supports
        self.n_spc, self.spc = 0, []
        done = set()
        for nodes, dofs, lab in self.supports:
            for nd in nodes:
                for d in dofs:
                    done.add((self.tmaster[nd] if d <= 3 else nd, d))
        tied_dep = {(m_, d) for ms, dofs, lab in getattr(self, 'dof_ties', []) for m_ in ms[1:] for d in dofs}
        assert not (tied_dep & done), "a tied dof is also constrained (SPC): %s" % sorted(tied_dep & done)[:3]
        if getattr(self, 'dof_ties', None) and any(len(sp['pa']) > 1 or len(sp['pb']) > 1 for sp in self.springs):
            raise ValueError("use either a spread spindle ('vertices') or add_dof_tie with a single-vertex spindle, not both")
        if done:
            self.spc = sorted(done)
            sup_masters = sorted({nd for nd, d in done})
            allg = sorted({x for mm in sup_masters for x in self._group_all(mm)})
            w('*NSET, NSET=NSUP')
            L.extend(_chunks(sup_masters))
            w('*NSET, NSET=NSUPALL')
            L.extend(_chunks(allg))
            w('*BOUNDARY')
            for nd, d in self.spc: w('%d, %d, %d' % (nd, d, d))
            self.n_spc = len(self.spc)
        # steps
        def outputs(static=True):
            if static:
                w('*NODE FILE')
                w('U')
                w('*EL FILE')
                w('S')
                w('*NODE PRINT, NSET=NALL')
                w('U')
                if print_rf:
                    w('*NODE PRINT, NSET=NALL')
                    w('RF')
            else:
                w('*NODE FILE, OUTPUT=2D')
                w('U')

        def cload(F):
            Fm = self.to_masters(F)
            w('*CLOAD, OP=NEW')
            nz = 0
            for nd in sorted(Fm):
                for d in range(3):
                    if Fm[nd][d] != 0.0:
                        w('%d, %d, %s' % (nd, d + 1, _num(Fm[nd][d]))); nz += 1
            if nz == 0:
                w('%d, 1, 0.' % min(self.xyz))
        self.step_names = []
        if analysis == 'static':
            assert cases, "static needs load cases"
            for name, F in cases.items():
                w('*STEP')
                w('*STATIC')
                cload(F)
                outputs(True)
                w('*END STEP')
                self.step_names.append(name)
        elif analysis == 'frequency':
            w('*STEP')
            w('*FREQUENCY')
            w('%d' % nmodes if freq_shift is None else '%d, %.10g' % (nmodes, freq_shift))
            outputs(False)
            w('*END STEP')
        elif analysis == 'buckle':
            F = cases if not (cases and isinstance(next(iter(cases.values())), dict)) else next(iter(cases.values()))
            w('*STEP')
            w('*BUCKLE')
            w('%d' % nmodes if buckle_accuracy is None else '%d, %g' % (nmodes, buckle_accuracy))
            cload(F)
            outputs(False)
            w('*END STEP')
        else:
            raise ValueError(analysis)
        with open(path, 'w') as fh: fh.write('\n'.join(L) + '\n')
        self.analysis = analysis
        return path

    def _group_all(self, master):
        return [nd for nd, mm in self.tmaster_all.items() if mm == master] if hasattr(self, 'tmaster_all') \
            else self.group(master)

    def to_masters(self, F):
        out = {}
        for nd, v in F.items():
            mm = self.tmaster[nd]
            out[mm] = out.get(mm, np.zeros(3)) + np.asarray(v, float)
        return out

    # ---- runs
    def run(self, workdir, job, analysis='static', threads=3, **kw):
        os.makedirs(workdir, exist_ok=True)
        self.write_inp(os.path.join(workdir, job + '.inp'), analysis=analysis, **kw)
        return run_ccx(workdir, job, threads=threads)

    def check_mechanisms(self, workdir, job='mech', nmodes=6, threads=3, factor=100.0, noise=None, tol=1e-9):
        """rigid-panel SVD (primary) + FE *FREQUENCY (confirmation).  Sets self.mech; returns it."""
        sv = rigid_panel_mechanisms(self, tol=tol)
        out = dict(svd=sv)
        if noise is None:
            noise = noise_floor(self, workdir, threads=threads)
        r = self.run(workdir, job, 'frequency', nmodes=nmodes, threads=threads)
        assert not r['errors'], r['errors'][:3]
        lam = [e[1] for e in r['dat']['eigen']]
        thr = factor*noise
        n_zero = int(sum(1 for x in lam if abs(x) < thr))
        n_neg = int(sum(1 for x in lam if x < -thr))
        out.update(fe_lambda=lam, fe_f_Hz=[e[3] for e in r['dat']['eigen']], noise=noise, threshold=thr,
                   fe_zero=n_zero, fe_negative=n_neg, lambda1_over_threshold=(lam[0]/thr if lam else None),
                   last_mode_above_threshold=bool(lam and lam[-1] > thr), freq_result=r)
        out['ok'] = (sv['n_zero'] == 0 and n_zero == 0 and n_neg == 0)
        self.mech = out
        return out

    def _gate(self):
        if self.mech is None:
            raise RuntimeError('mechanism check not done for this model/supports (call check_mechanisms first)')
        if not self.mech['ok']:
            raise RuntimeError('model has a mechanism: SVD zero modes %d, FE zero modes %d, FE negative %d -> static and '
                               'buckling results refused' % (self.mech['svd']['n_zero'], self.mech['fe_zero'],
                                                             self.mech['fe_negative']))

    def run_static(self, workdir, job, cases, threads=3, force=False, separate=None):
        """linear static load cases; refuses unless check_mechanisms passed (force=True only for benchmarks of known-rigid
        systems).  separate (default: True when the model has a spindle): one ccx job per case.  Reason: ccx
        evaluates the SPRINGA force with the deformed length and carries that tiny nonlinear residual into the next *STEP
        of a multi-step deck (reaction sums then off by 1e-4); without springs one deck with one *STEP per case is exact."""
        if not force: self._gate()
        os.makedirs(workdir, exist_ok=True)
        if separate is None: separate = bool(self.springs)
        if not separate:
            self.write_inp(os.path.join(workdir, job + '.inp'), analysis='static', cases=cases)
            r = run_ccx(workdir, job, threads=threads)
            assert not r['errors'], r['errors'][:5]
            return StaticResult(self, [(list(cases.keys()), r)], {k: dict(v) for k, v in cases.items()})
        parts = []
        for q, (nm, F) in enumerate(cases.items()):
            jb = '%s_c%d' % (job, q)
            self.write_inp(os.path.join(workdir, jb + '.inp'), analysis='static', cases={nm: F})
            r = run_ccx(workdir, jb, threads=threads)
            assert not r['errors'], r['errors'][:5]
            parts.append(([nm], r))
        return StaticResult(self, parts, {k: dict(v) for k, v in cases.items()})

    def run_buckle(self, workdir, job, forces, nmodes=4, threads=3, force=False, accuracy=1e-7):
        """accuracy (2nd *BUCKLE data field, ccx default 0.01) = 1e-7.  With the default, ccx returned a
        higher lowest factor (1 mode: +0.3..14 %) and skipped clustered modes; with 1e-7 one mode is already exact."""
        if not force: self._gate()
        r = self.run(workdir, job, 'buckle', cases=forces, nmodes=nmodes, threads=threads, buckle_accuracy=accuracy)
        assert not r['errors'], r['errors'][:5]
        return [b[1] for b in r['dat']['buckling']], r


# ---------------------------------------------------------------- static results (linear, combinable)
class StaticResult:
    """per load case: U (printed .dat displacement: mean of the expanded nodes for shell nodes, own displacement for
    spindle nodes), RF, applied forces; and the expanded-node stresses mapped to (node, face, side).
    parts: [([case names in step order], ccx result), ...]"""

    def __init__(self, model, parts, cases):
        self.model, self.cases = model, cases
        self.names = [nm for names, r in parts for nm in names]
        self.raw = parts[0][1] if len(parts) == 1 else [r for names, r in parts]
        self.U, self.RF, self.S = {}, {}, {}
        self.smap = None
        for names, r in parts:
            disp = [b for b in r['dat']['blocks'] if b['name'].startswith('displacements') and b['set'] == 'NALL']
            frc = [b for b in r['dat']['blocks'] if b['name'].startswith('forces') and b['set'] == 'NALL']
            assert len(disp) == len(names), (len(disp), len(names))
            if r['frd'] is not None and self.smap is None:
                self.smap = expanded_map(model, r['frd']['nodes'])
            st = [b for b in r['frd']['blocks'] if b['name'] == 'STRESS'] if r['frd'] is not None else []
            for q, nm in enumerate(names):
                self.U[nm] = {k: v[:3] for k, v in disp[q]['data'].items() if k != 'total'}
                if frc: self.RF[nm] = {k: v[:3] for k, v in frc[q]['data'].items() if k != 'total'}
                # ccx evaluates the SPRINGA internal force from the DEFORMED length even in a linear step (
                # printed RF = k (dL_lin + |du_perp|^2 / 2L), 0.06 % off for a 61 um sideways motion), while the solve is
                # linear: the RF of the spindle end nodes is replaced by the linear force k (u_b - u_a).e
                for sp in model.springs:
                    e = (sp['xb'] - sp['xa'])/np.linalg.norm(sp['xb'] - sp['xa'])
                    N = sp['k']*float(np.dot(self.U[nm][sp['nb']] - self.U[nm][sp['na']], e))
                    if frc:
                        self.RF[nm][sp['na']] = -N*e
                        self.RF[nm][sp['nb']] = N*e
                if st:
                    assert len(st) == len(names), (len(st), len(names))
                    D = st[q]['data']
                    self.S[nm] = {key: np.asarray(D[fid][:6]) for key, fid in self.smap['map'].items() if fid in D}

    def combo(self, coef):
        """linear combination {case: factor} -> dict with U, RF, S, F (applied, by node before master mapping)"""
        out = {'U': {}, 'RF': {}, 'S': {}, 'F': {}}
        for nm, c in coef.items():
            if c == 0: continue
            for key in ('U', 'RF', 'S'):
                src = getattr(self, key).get(nm, {})
                dst = out[key]
                for k, v in src.items():
                    dst[k] = dst.get(k, 0.0) + c*np.asarray(v, float)
            for k, v in self.cases[nm].items():
                out['F'][k] = out['F'].get(k, 0.0) + c*np.asarray(v, float)
        return out


def expanded_map(model, frd_nodes, rel_tol=2e-5):
    """map (shell node, face, side) -> frd node id of the expanded shell (side +1 = +n_f surface, 0 = mid, -1 = -n_f),
    by POSITION (knots and dummy beams change the numbering): expected positions for OFFSET o:
    top = x + n t (0.5 - o), mid = x - n t o, bottom = x - n t (0.5 + o)."""
    from scipy.spatial import cKDTree
    ids = np.array([k for k in frd_nodes if k > model.n_shell_nodes + len(getattr(model, 'spindle_nodes', []))])
    X = np.array([frd_nodes[k] for k in ids])
    tree = cKDTree(X)
    t, o = model.section.t, model.offset_fraction
    keys, P = [], []
    for nd, fs in model.node_faces.items():
        x = model.xyz[nd]
        for f in fs:
            n = model.face_normal(f)
            for side, z in ((1, t*(0.5 - o)), (0, -t*o), (-1, -t*(0.5 + o))):
                keys.append((nd, f, side)); P.append(x + n*z)
    P = np.array(P)
    d, idx = tree.query(P, k=2)
    # frd coordinates carry 6 significant digits (E12.5): tolerance relative to the model size; the next candidate must be
    # clearly farther (dummy-beam nodes sit +-bdim/2 from the reference point)
    scale = max(1.0, float(np.abs(X).max()))
    tol = rel_tol*scale
    amb = int(np.sum((d[:, 0] < tol) & (d[:, 1] < 3*tol)))
    d, idx = d[:, 0], idx[:, 0]
    ok = d < tol
    mp = {keys[q]: int(ids[idx[q]]) for q in range(len(keys)) if ok[q]}
    return {'map': mp, 'missing': int((~ok).sum()), 'n': len(keys), 'ambiguous': amb, 'tol': tol,
            'max_dist_ok': float(d[ok].max()) if ok.any() else None}


# ---------------------------------------------------------------- running and parsing
def run_ccx(workdir, job, threads=3, timeout=7200):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), CCX_NPROC_EQUATION_SOLVER=str(threads))
    t0 = time.time()
    p = subprocess.run([CCX, '-i', job], cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout)
    log = p.stdout + p.stderr
    with open(os.path.join(workdir, job + '.log'), 'w') as fh: fh.write(log)
    res = {'returncode': p.returncode, 'seconds': time.time() - t0, 'log': log,
           'errors': [l for l in log.splitlines() if '*ERROR' in l],
           'warnings': [l for l in log.splitlines() if '*WARNING' in l]}
    dat, frd = os.path.join(workdir, job + '.dat'), os.path.join(workdir, job + '.frd')
    res['dat'] = parse_dat(dat) if os.path.exists(dat) else None
    res['frd'] = parse_frd(frd) if os.path.exists(frd) else None
    return res


_FLOAT = re.compile(r'[-+]?\d*\.\d+(?:[EeDd][-+]?\d+)?|[-+]?\d+(?:[EeDd][-+]?\d+)')


def parse_dat(path):
    out = {'blocks': [], 'eigen': [], 'buckling': []}
    lines = open(path).read().splitlines()
    q = 0
    while q < len(lines):
        l = lines[q]
        s = l.strip()
        if s.startswith('E I G E N V A L U E   O U T P U T'):
            q += 1
            while q < len(lines) and not re.match(r'^\s*\d+\s+[-\d.]', lines[q]): q += 1
            while q < len(lines) and re.match(r'^\s*\d+\s+[-\d.]', lines[q]):
                v = lines[q].split()
                out['eigen'].append((int(v[0]), float(v[1]), float(v[2]), float(v[3])))
                q += 1
            continue
        if s.startswith('B U C K L I N G   F A C T O R   O U T P U T'):
            q += 1
            while q < len(lines) and not re.match(r'^\s*\d+\s+[-\d.]', lines[q]): q += 1
            while q < len(lines) and re.match(r'^\s*\d+\s+[-\d.]', lines[q]):
                v = lines[q].split()
                out['buckling'].append((int(v[0]), float(v[1])))
                q += 1
            continue
        mt = re.match(r'^\s*(.+?) for set (\S+) and time\s+(\S+)', l)
        if mt:
            blk = {'name': mt.group(1).strip(), 'set': mt.group(2), 'time': float(mt.group(3)), 'data': {}}
            q += 1
            while q < len(lines) and not lines[q].strip(): q += 1
            while q < len(lines) and lines[q].strip():
                v = lines[q].split()
                try:
                    if v[0].lower() == 'total' or not v[0].isdigit():
                        blk['data'].setdefault('total', []).append([float(x) for x in v if _FLOAT.fullmatch(x)])
                    elif 'stresses' in blk['name'] or 'strain' in blk['name']:
                        blk['data'][(int(v[0]), int(v[1]))] = np.array([float(x) for x in v[2:8]])
                    else:
                        blk['data'][int(v[0])] = np.array([float(x) for x in v[1:]])
                except ValueError:
                    pass
                q += 1
            out['blocks'].append(blk)
            continue
        q += 1
    return out


def _vals(s):
    return [float(x.replace('D', 'E')) for x in re.findall(r'[-+]?\d\.\d+[EeDd][-+]\d+', s)]


def parse_frd(path):
    """ASCII .frd: node coordinates and result blocks [{name, value, mode, comps, data{node: array}}]"""
    nodes, blocks = {}, []
    mode = None
    with open(path) as fh:
        lines = fh.read().splitlines()
    q = 0
    while q < len(lines):
        l = lines[q]
        if l.startswith('    2C'):
            q += 1
            while not lines[q].startswith(' -3'):
                if lines[q].startswith(' -1'):
                    nodes[int(lines[q][3:13])] = np.array(_vals(lines[q][13:]))
                q += 1
        elif l.startswith('    1PMODE'):
            mode = int(l.split()[1])
        elif l.startswith('  100C'):
            hdr = l
            try: value = float(hdr[12:24])
            except ValueError: value = float('nan')
            q += 1
            name = lines[q][5:13].strip() if lines[q].startswith(' -4') else '?'
            comps = []
            data = {}
            q += 1
            while q < len(lines) and not lines[q].startswith(' -3'):
                r = lines[q]
                if r.startswith(' -5'): comps.append(r[5:13].strip())
                elif r.startswith(' -1'):
                    nid = int(r[3:13]); data[nid] = np.array(_vals(r[13:]))
                elif r.startswith(' -2'):
                    data[nid] = np.concatenate([data[nid], _vals(r[13:])])
                q += 1
            blocks.append({'name': name, 'value': value, 'mode': mode, 'comps': comps, 'data': data})
            mode = None
        q += 1
    return {'nodes': nodes, 'blocks': blocks}


def von_mises(s):
    s = np.asarray(s)
    sxx, syy, szz, sxy, syz, szx = s[..., 0], s[..., 1], s[..., 2], s[..., 3], s[..., 4], s[..., 5]
    return np.sqrt(0.5*((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2) + 3*(sxy**2 + syz**2 + szx**2))


def tensor(s):
    """frd stress vector (SXX, SYY, SZZ, SXY, SYZ, SZX) -> 3x3 (arrays: (..., 3, 3))"""
    s = np.asarray(s, float)
    T = np.empty(s.shape[:-1] + (3, 3))
    T[..., 0, 0], T[..., 1, 1], T[..., 2, 2] = s[..., 0], s[..., 1], s[..., 2]
    T[..., 0, 1] = T[..., 1, 0] = s[..., 3]
    T[..., 1, 2] = T[..., 2, 1] = s[..., 4]
    T[..., 0, 2] = T[..., 2, 0] = s[..., 5]
    return T


def node_displacements(res, step=0):
    """mid-surface displacements of all original nodes from the .dat (step = index of the static step)"""
    bl = [b for b in res['dat']['blocks'] if b['name'].startswith('displacements') and b['set'] == 'NALL']
    return {k: v for k, v in bl[step]['data'].items() if k != 'total'} if bl else None


def total_reaction(model, RF, F):
    """sum of the reactions = sum over every supported (master, dof) of RF over the master's WHOLE tied group
    (shell dependents and spindle end nodes) minus the loads applied in that group.  RF, F: {node: vec}"""
    R = np.zeros(3)
    Fm = model.to_masters(F) if F else {}
    extra = model._distributed_rf(RF)
    for nd, d in model.spc:
        if d > 3: continue
        grp = model._group_all(nd)
        R[d - 1] += sum(RF[x][d - 1] for x in grp if x in RF) - (Fm[nd][d - 1] if nd in Fm else 0.0) + \
            (extra[nd][d - 1] if nd in extra else 0.0)
    return R


def reactions_by_label(model, RF, F, exclude_spindle=False):
    """{label: {'nodes': [masters], 'R': {master: vec (constrained dofs only)}, 'sum': vec}}.  exclude_spindle (checked
    copy): leave out the spindle end nodes (and spread-end shares): the force of the SHELLS on the support = the arch's
    thrust on the foot (in V2/V3 the label sum incl. the spindle is ~0 across, because the spindle end sits on foot 1)."""
    Fm = model.to_masters(F) if F else {}
    extra = {} if exclude_spindle else model._distributed_rf(RF)
    spn = set(getattr(model, 'spindle_nodes', [])) if exclude_spindle else set()
    out = {}
    seen = set()
    for nodes, dofs, lab in model.supports:
        o = out.setdefault(lab, {'R': {}, 'sum': np.zeros(3)})
        for nd in nodes:
            mm = model.tmaster[nd]
            for d in dofs:
                if d > 3 or (mm, d) in seen: continue
                seen.add((mm, d))
                val = sum(RF[x][d - 1] for x in model._group_all(mm) if x in RF and x not in spn) - \
                    (Fm[mm][d - 1] if mm in Fm else 0.0) + (extra[mm][d - 1] if mm in extra else 0.0)
                o['R'].setdefault(mm, np.zeros(3))[d - 1] += val
                o['sum'][d - 1] += val
    return out


def spindle_forces(model, U):
    """axial force (tension > 0) of every spindle from the displacements of its plain end nodes"""
    out = {}
    for sp in model.springs:
        e = sp['xb'] - sp['xa']
        e = e/np.linalg.norm(e)
        out[sp['label']] = float(sp['k']*np.dot(np.asarray(U[sp['nb']]) - np.asarray(U[sp['na']]), e))
    return out


# ---------------------------------------------------------------- loads
def gravity_forces(model, factor=1.0):
    """self-weight along -up: consistent rho g t int N dA per element node (mass per area = rho t of the section)"""
    F = {}
    q = model.section.rho*G_ACC*model.section.t*factor
    for f, con in model.elems:
        X = [model.xyz[c] for c in con[:4]]
        for nd, w in zip(con, _elem_load_integrals(model.elem, X)):
            F[nd] = F.get(nd, 0.0) - q*w*model.up
    return F


def point_forces(model, target, vec):
    """total force vec at a target, split equally over its translation masters"""
    ms = sorted({model.tmaster[x] for x in model._resolve(target)})
    return {x: np.asarray(vec, float)/len(ms) for x in ms}


def line_forces(model, e, vec_per_length, face=None):
    """uniform line load (N/m) along edge e on the side of one face, consistent nodal forces"""
    f, s = model.edge_faces(e)[0] if face is None else [x for x in model.edge_faces(e) if x[0] == face][0]
    nodes = model.side_nodes(f, s)
    v = np.asarray(vec_per_length, float)
    F = {}
    step = 2 if model.elem == 'S8R' else 1
    for s0 in range(0, len(nodes) - 1, step):
        seg = nodes[s0:s0 + step + 1]
        Lg = np.linalg.norm(model.xyz[seg[-1]] - model.xyz[seg[0]])
        ws = (1/6, 2/3, 1/6) if step == 2 else (0.5, 0.5)
        for nd, w in zip(seg, ws): F[nd] = F.get(nd, 0.0) + v*Lg*w
    return F


def add_forces(*Fs, factors=None):
    out = {}
    factors = factors or [1.0]*len(Fs)
    for F, c in zip(Fs, factors):
        for k, v in F.items(): out[k] = out.get(k, 0.0) + c*np.asarray(v, float)
    return out


def force_total(F):
    return np.sum([np.asarray(v, float) for v in F.values()], axis=0) if F else np.zeros(3)


def plan_frame(up, along):
    """orthonormal frame ex (across), ey (along the hall, projected perpendicular to up), ez = up"""
    ez = np.asarray(up, float)/np.linalg.norm(up)
    ey = np.asarray(along, float) - np.dot(along, ez)*ez
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, ez)
    return ex, ey, ez


def raycast_top(model, frame, grid=0.01, faces=None):
    """topmost face above every plan cell centre.  Returns dict x, y (cell centres), face index (-1 = no hit), z, s, u
    (face bilinear parameters of the hit point).  Vertical rays; each face is planar and convex."""
    ex, ey, ez = frame
    faces = faces or model.faces()
    Q = {v: np.array([np.dot(p, ex), np.dot(p, ey), np.dot(p, ez)]) for v, p in model.pos.items()}
    allp = np.array(list(Q.values()))
    x0, x1 = allp[:, 0].min(), allp[:, 0].max()
    y0, y1 = allp[:, 1].min(), allp[:, 1].max()
    nx, ny = int(math.ceil((x1 - x0)/grid)), int(math.ceil((y1 - y0)/grid))
    xs = x0 + (np.arange(nx) + 0.5)*grid
    ys = y0 + (np.arange(ny) + 0.5)*grid
    zt = np.full((nx, ny), -np.inf)
    fid = np.full((nx, ny), -1, int)
    S = np.zeros((nx, ny)); Uu = np.zeros((nx, ny))
    for q, f in enumerate(faces):
        i, j = f
        C = [Q[(i, j)], Q[(i + 1, j)], Q[(i + 1, j + 1)], Q[(i, j + 1)]]
        nrm = np.cross(C[2] - C[0], C[3] - C[1]); nrm /= np.linalg.norm(nrm)
        if abs(nrm[2]) < 1e-9: continue
        cx = [c[0] for c in C]; cy = [c[1] for c in C]
        ia = np.searchsorted(xs, min(cx) - 1e-12); ib = np.searchsorted(xs, max(cx) + 1e-12)
        ja = np.searchsorted(ys, min(cy) - 1e-12); jb = np.searchsorted(ys, max(cy) + 1e-12)
        if ia >= ib or ja >= jb: continue
        X, Y = np.meshgrid(xs[ia:ib], ys[ja:jb], indexing='ij')
        cen = np.mean(C, 0)
        Z = cen[2] - (nrm[0]*(X - cen[0]) + nrm[1]*(Y - cen[1]))/nrm[2]
        P = np.stack([X, Y, Z], -1)
        # in-plane 2-D coordinates and inverse bilinear map (Newton)
        e1 = C[1] - C[0]; e1 /= np.linalg.norm(e1); e2 = np.cross(nrm, e1)
        c2 = [np.array([np.dot(c - C[0], e1), np.dot(c - C[0], e2)]) for c in C]
        p2 = np.stack([np.einsum('...k,k->...', P - C[0], e1), np.einsum('...k,k->...', P - C[0], e2)], -1)
        s = np.full(X.shape, 0.5); u = np.full(X.shape, 0.5)
        for _ in range(30):
            Fv = ((1 - s)*(1 - u))[..., None]*c2[0] + (s*(1 - u))[..., None]*c2[1] + (s*u)[..., None]*c2[2] + \
                 ((1 - s)*u)[..., None]*c2[3] - p2
            ds_ = (-(1 - u))[..., None]*c2[0] + (1 - u)[..., None]*c2[1] + u[..., None]*c2[2] - u[..., None]*c2[3]
            du_ = (-(1 - s))[..., None]*c2[0] - s[..., None]*c2[1] + s[..., None]*c2[2] + (1 - s)[..., None]*c2[3]
            det = ds_[..., 0]*du_[..., 1] - ds_[..., 1]*du_[..., 0]
            dS = (Fv[..., 0]*du_[..., 1] - Fv[..., 1]*du_[..., 0])/det
            dU = (ds_[..., 0]*Fv[..., 1] - ds_[..., 1]*Fv[..., 0])/det
            s -= dS; u -= dU
            if max(np.abs(dS).max(), np.abs(dU).max()) < 1e-13: break
        eps = 1e-10
        inside = (s >= -eps) & (s <= 1 + eps) & (u >= -eps) & (u <= 1 + eps)
        blk = zt[ia:ib, ja:jb]
        better = inside & (Z > blk + 1e-12)
        blk[better] = Z[better]
        fid[ia:ib, ja:jb][better] = q
        S[ia:ib, ja:jb][better] = np.clip(s[better], 0, 1)
        Uu[ia:ib, ja:jb][better] = np.clip(u[better], 0, 1)
    return dict(x=xs, y=ys, z=zt, face=fid, s=S, u=Uu, faces=faces, grid=grid, frame=frame)


def _upper_hull(px, pz):
    pts = sorted(zip(px, pz))
    hull = []
    for p in pts:
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0])*(p[1] - hull[-2][1]) - \
                (hull[-1][1] - hull[-2][1])*(p[0] - hull[-2][0]) >= 0:
            hull.pop()
        hull.append(p)
    return np.array(hull)


def smooth_envelope(rc, base_z=None, smooth_len=0.15, ds=0.005):
    """smooth envelope of the ray-cast top surface: profile max_y z_top(x), its upper concave hull (incl. the two foot
    points at base_z), arc-length resampling and Gaussian smoothing (sigma = smooth_len).  Returns x(tau), z(tau),
    beta(tau) (deg), and the 60-degree interval [x_L, x_R] around the crown, h, b."""
    from scipy.ndimage import gaussian_filter1d
    z = rc['z']
    hit = np.isfinite(z)
    prof = np.where(hit.any(1), np.where(hit, z, -np.inf).max(1), np.nan)
    ok = np.isfinite(prof)
    px, pz = rc['x'][ok], prof[ok]
    if base_z is None: base_z = float(pz.min())
    g = rc['grid']
    pxs = np.concatenate([[px[0] - g/2], px, [px[-1] + g/2]])
    pzs = np.concatenate([[base_z], pz, [base_z]])
    H = _upper_hull(pxs, pzs)
    seg = np.linalg.norm(np.diff(H, axis=0), axis=1)
    tau = np.concatenate([[0], np.cumsum(seg)])
    tt = np.arange(0, tau[-1] + ds/2, ds)
    xr, zr = np.interp(tt, tau, H[:, 0]), np.interp(tt, tau, H[:, 1])
    sig = max(smooth_len/ds, 1e-9)
    xs_, zs_ = gaussian_filter1d(xr, sig, mode='nearest'), gaussian_filter1d(zr, sig, mode='nearest')
    dx, dz = np.gradient(xs_), np.gradient(zs_)
    beta = np.degrees(np.arctan2(np.abs(dz), np.abs(dx)))
    return dict(x=xs_, z=zs_, beta=beta, hull=H, profile_x=px, profile_z=pz, base_z=base_z,
                h=float(pz.max() - base_z), b=float(pxs[-1] - pxs[0]))


def snow_interval(env, beta_max=60.0):
    """plan interval [x_L, x_R] around the crown (max z) where the envelope slope beta <= beta_max"""
    x, z, beta = env['x'], env['z'], env['beta']
    c = int(np.argmax(z))
    l = c
    while l > 0 and beta[l - 1] <= beta_max: l -= 1
    r = c
    while r < len(x) - 1 and beta[r + 1] <= beta_max: r += 1

    def cross(a, b):     # interpolate the beta = beta_max crossing between samples a (in) and b (out)
        if beta[b] == beta[a]: return x[a]
        w = (beta_max - beta[a])/(beta[b] - beta[a])
        return x[a] + w*(x[b] - x[a])
    xL = cross(l, l - 1) if l > 0 else x[0]
    xR = cross(r, r + 1) if r < len(x) - 1 else x[-1]
    inner = beta[l:r + 1]
    return float(min(xL, xR)), float(max(xL, xR)), bool((inner <= beta_max + 1e-9).all())


def mu_profile(x, case, xL, xR, mu1=0.8, mu3=2.0, drift=+1, reading='triangles'):
    """EN 1991-1-3 5.3.5 / Figure 5.6 shape coefficient at plan positions x (array)"""
    x = np.asarray(x, float)
    mu = np.zeros_like(x)
    inn = (x >= xL) & (x <= xR)
    if case == 'S1':
        mu[inn] = mu1
        return mu
    if case == 'uniform':
        mu[:] = 1.0
        return mu
    assert case == 'S2'
    ls = xR - xL
    xc = 0.5*(xL + xR)
    # coordinate measured from the low-load end toward the high-load end
    sx = (x - xL) if drift > 0 else (xR - x)
    if reading == 'triangles':
        half = ls/2
        a = np.clip(sx, 0, ls)
        left = a <= half
        tri_l = 0.5*mu3*(1 - np.abs(a - half/2)/(half/2))
        tri_r = mu3*(1 - np.abs(a - 1.5*half)/(half/2))
        mu = np.where(left, tri_l, tri_r)
        mu = np.clip(mu, 0, None)
    elif reading == 'halves':
        mu = np.where(sx <= ls/2, 0.5*mu3, mu3)
    else:
        raise ValueError(reading)
    mu[~inn] = 0.0
    return mu


def snow_cylindrical(model, s_k=1.0, case='S1', along=(0.0, 1.0, 0.0), drift=+1, reading='triangles', grid=0.01,
                     smooth_len=0.15, beta_max=60.0, mu1=0.8, mu3_max=2.0, base_z=None, rc=None, env=None,
                     up_facing_only=True):
    """EN 1991-1-3 5.3.5 cylindrical-roof snow on the exposed (topmost) roof, as consistent nodal forces along -up.
    case 'S1' (mu 0.8) | 'S2' (drifted, see mu_profile / module docstring) | 'uniform' (mu = 1 everywhere, for tests).
    Returns dict(forces, total (vector), plan_total (sum of mu s_k dA), info...)"""
    frame = plan_frame(model.up, along)
    if rc is None: rc = raycast_top(model, frame, grid=grid)
    if base_z is None:      # foot level = lowest point of the net along up
        base_z = float(min(np.dot(p, frame[2]) for p in model.pos.values()))
    if env is None: env = smooth_envelope(rc, base_z=base_z, smooth_len=smooth_len)
    xL, xR, single = snow_interval(env, beta_max)
    hb = env['h']/env['b']
    mu3 = min(0.2 + 10*hb, mu3_max)
    hit = rc['face'] >= 0
    I, J = np.nonzero(hit)
    # cell-averaged mu of every grid column (32 sub-samples across the cell): exact integrals of the step (S1, 'halves')
    # and kinked ('triangles') profiles up to the plan coverage of the grid
    g = rc['grid']
    sub = (np.arange(32) + 0.5)/32 - 0.5
    mucol = mu_profile((rc['x'][:, None] + g*sub[None, :]).ravel(), case, xL, xR, mu1=mu1, mu3=mu3, drift=drift,
                       reading=reading).reshape(len(rc['x']), 32).mean(1)
    mu = mucol[I]
    sel = mu > 0
    # up-facing only: a topmost face whose OUTER normal points down (open-end zigzag faces: the upper side is the slab's
    # inner side) gets no snow when up_facing_only; the dropped load is reported
    fq_all = rc['face'][I, J]
    outer_up = np.array([np.dot(model.outer_sign*model.face_normal(f_), model.up) for f_ in rc['faces']])
    down_cells = outer_up[fq_all] < 0
    dropped = float((mu*s_k*rc['grid']**2)[sel & down_cells].sum())
    if up_facing_only: sel = sel & ~down_cells
    I, J, mu = I[sel], J[sel], mu[sel]
    dA = rc['grid']**2
    load = mu*s_k*dA                          # N (s_k in N/m^2)
    fq = rc['face'][I, J]; s = rc['s'][I, J]; u = rc['u'][I, J]
    ka, kb = model.ka, model.kb
    ea = np.minimum((s*ka).astype(int), ka - 1); eb = np.minimum((u*kb).astype(int), kb - 1)
    xi = 2*(s*ka - ea) - 1; eta = 2*(u*kb - eb) - 1
    Nsh = _shape(model.elem, xi, eta)        # (nn, npts)
    F = {}
    faces = rc['faces']
    # vectorised accumulation per element
    keys = np.stack([fq, ea, eb], 1)
    order = np.lexsort((eb, ea, fq))
    keys, load_o, N_o = keys[order], load[order], Nsh[:, order]
    brk = np.nonzero(np.any(np.diff(keys, axis=0) != 0, axis=1))[0] + 1
    starts = np.concatenate([[0], brk]); ends = np.concatenate([brk, [len(load_o)]])
    down = -np.asarray(model.up, float)
    per_face = {}
    for a, b in zip(starts, ends):
        fqi, eai, ebi = keys[a]
        f = faces[fqi]
        con = model.elems[model.face_elem[(f, int(eai), int(ebi))]][1]
        fn = (N_o[:, a:b]*load_o[a:b]).sum(1)
        for nd, w_ in zip(con, fn):
            F[nd] = F.get(nd, 0.0) + w_*down
        per_face[f] = per_face.get(f, 0.0) + float(load_o[a:b].sum())
    plan_total = float(load.sum())
    # orientation check: faces receiving load whose OUTER normal points down (should not exist on the exposed roof)
    bad = []
    for f, val in per_face.items():
        n_out = model.outer_sign*model.face_normal(f)
        if np.dot(n_out, model.up) < -1e-9 and val > 0: bad.append((f, val))
    ex, ey, ez = frame
    Lhall = rc['y'][-1] - rc['y'][0] + rc['grid']
    return dict(forces=F, total=force_total(F), plan_total=plan_total, case=case, reading=reading, drift=drift,
                dropped_on_outer_down_faces=dropped, up_facing_only=up_facing_only,
                xL=xL, xR=xR, l_s=xR - xL, single_interval=single, h=env['h'], b=env['b'], h_over_b=hb, mu3=mu3,
                per_face=per_face, loaded_faces_outer_normal_down=bad, hall_length_grid=Lhall, env=env, rc=rc,
                frame=frame)


# ---------------------------------------------------------------- mechanism checks
def rigid_panel_mechanisms(model, tol=1e-9, with_supports=True):
    """rigid-panel SVD: unknowns (v_f, w_f) per face; rows: hinges (equal velocity at both edge ends), locks and
    latches (+ equal w), supports (e_d . velocity = 0 at each supported node for each of its faces), spindles (rigid bar).
    Positions scaled by the model size.  Returns n_zero (incl. unconstrained rigid motions), the smallest singular values
    (relative to the largest) and the gap s_(n_zero+1) / max(zero s)."""
    faces = model.faces()
    idx = {f: q for q, f in enumerate(faces)}
    allp = np.array(list(model.pos.values()))
    c0 = allp.mean(0)
    Ls = max(np.ptp(allp, 0).max(), 1e-12)
    P = lambda x: (np.asarray(x, float) - c0)/Ls
    ncol = 6*len(faces)
    rows = []

    def vel_row(f, x):
        R = np.zeros((3, ncol))
        R[:, 6*idx[f]:6*idx[f] + 3] = np.eye(3)
        R[:, 6*idx[f] + 3:6*idx[f] + 6] = -_skew(P(x))
        return R
    latched = {lt[0] for lt in model.latches}
    for e, c in model.conn.items():
        if c == 'free': continue
        (f1, _), (f2, _) = model.edge_faces(e)
        for v in model.edge_ends(e):
            rows.append(vel_row(f1, model.pos[v]) - vel_row(f2, model.pos[v]))
        if c == 'lock' or e in latched:
            R = np.zeros((3, ncol))
            R[:, 6*idx[f1] + 3:6*idx[f1] + 6] = np.eye(3); R[:, 6*idx[f2] + 3:6*idx[f2] + 6] = -np.eye(3)
            rows.append(R)
    if with_supports:
        seen = set()
        for nodes, dofs, lab in model.supports:
            for nd in nodes:
                for d in dofs:
                    if d > 3: continue
                    for f in model.node_faces[nd]:
                        key = (nd, d, f)
                        if key in seen: continue
                        seen.add(key)
                        rows.append(vel_row(f, model.xyz[nd])[d - 1:d])
        for sp in model.springs:
            e = (sp['xb'] - sp['xa'])/np.linalg.norm(sp['xb'] - sp['xa'])
            row = np.zeros((1, ncol))
            for key, sgn in (('pb', 1.0), ('pa', -1.0)):
                for m_, w_ in sp[key]:
                    f_ = sorted(model.node_faces[m_])[0]
                    row += sgn*w_*(e @ vel_row(f_, model.xyz[m_]))[None, :]
            rows.append(row)
        for ms, dofs, lab in getattr(model, 'dof_ties', []):
            for m_ in ms[1:]:
                for d in dofs:
                    f1, f0 = sorted(model.node_faces[m_])[0], sorted(model.node_faces[ms[0]])[0]
                    rows.append((vel_row(f1, model.xyz[m_]) - vel_row(f0, model.xyz[ms[0]]))[d - 1:d])
    A = np.vstack(rows) if rows else np.zeros((1, ncol))
    sv = np.linalg.svd(A, compute_uv=False)
    full = np.zeros(ncol); full[:min(len(sv), ncol)] = sv[:ncol]
    full = np.sort(full)
    rel = full/full[-1]
    nz = int(np.sum(rel < tol))
    gap = float(rel[nz]/max(rel[:nz].max(), 1e-300)) if 0 < nz < len(rel) else None
    return dict(n_zero=nz, smallest=[float(x) for x in rel[:max(nz + 3, 6)]], gap=gap, rows=A.shape[0], cols=ncol)


def noise_floor(model, workdir, threads=3):
    """|lambda| noise of rigid modes: free-free *FREQUENCY of one flat square face with the same section, k, element,
    offset and knot beams (on two opposite sides) as model; returns max |lambda_1..6|"""
    fs = model.faces()
    a = np.mean([np.linalg.norm(model.pos[(i + 1, j)] - model.pos[(i, j)]) for i, j in fs])
    pos = {(0, 0): (0, 0, 0), (1, 0): (a, 0, 0), (1, 1): (a, a, 0), (0, 1): (0, a, 0)}
    M = NetModel(pos, 1, 1, k=(model.ka, model.kb), elem=model.elem, section=model.section,
                 hinge_offset=model.hinge_offset, outer_sign=model.outer_sign, knot_beam=model.knot_beam)
    if model.knot_beam:
        M.supports = [(M.side_nodes((0, 0), 'left') + M.side_nodes((0, 0), 'right'), (), 'knots')]  # dofs () = knots only
    r = M.run(workdir, 'noise_ref', 'frequency', nmodes=8, threads=threads)
    assert not r['errors'], r['errors'][:3]
    lam = [e[1] for e in r['dat']['eigen']]
    return float(max(abs(x) for x in lam[:6]))


# ---------------------------------------------------------------- stress recovery and checks
def face_and_core(model, S):
    """sandwich recovery at every (node, face) with complete top/mid/bottom stresses.  S: {(node, face, side): 6-vector}
    (from StaticResult.S or a combination).  Returns dict of arrays: keys, sig_outer (3x3), sig_inner (3x3), vm_outer,
    vm_inner, smin_outer, smin_inner (smallest in-plane principal, < 0 = compression), tau_core (|.|), Q (|.|)"""
    sec = model.section
    info = sec.info
    sandwich = info.get('kind') == 'sandwich_equivalent'
    # S8R corner nodes expand to top/mid/bottom nodes, mid-side nodes only to top/bottom (C3D20R): there the mid-surface
    # stress is taken as the mean of the two surfaces (core shear within 0.1 % of the series solution)
    keys = sorted({(nd, f) for (nd, f, s) in S if (nd, f, 1) in S and (nd, f, -1) in S})
    if not keys: return None
    top = tensor(np.array([S[(nd, f, 1)] for nd, f in keys]))
    bot = tensor(np.array([S[(nd, f, -1)] for nd, f in keys]))
    mid = tensor(np.array([S[(nd, f, 0)] if (nd, f, 0) in S else 0.5*(np.asarray(S[(nd, f, 1)]) + np.asarray(S[(nd, f, -1)]))
                           for nd, f in keys]))
    nrm = np.array([model.face_normal(f) for nd, f in keys])
    Pm = np.eye(3)[None] - nrm[:, :, None]*nrm[:, None, :]
    ip = lambda T: Pm @ T @ Pm
    t_eq = sec.t
    Nt = (ip(top) + ip(bot))/2*t_eq
    Mt = (ip(top) - ip(bot))/2*t_eq**2/6
    if sandwich:
        tf, d = info['t_f'], info['d']
        f_plus = Nt/(2*tf) + Mt/(d*tf)          # face on the +n side
        f_minus = Nt/(2*tf) - Mt/(d*tf)
    else:
        f_plus, f_minus = ip(top), ip(bot)
    trac = np.einsum('kij,kj->ki', mid, nrm)
    tau = trac - np.einsum('ki,ki->k', trac, nrm)[:, None]*nrm
    Q = np.linalg.norm(tau, axis=1)*t_eq
    tau_c = Q/info['d'] if sandwich else np.linalg.norm(tau, axis=1)
    if model.outer_sign > 0: so, si = f_plus, f_minus
    else: so, si = f_minus, f_plus

    def vm(T):
        return np.sqrt(0.5*((T[:, 0, 0] - T[:, 1, 1])**2 + (T[:, 1, 1] - T[:, 2, 2])**2 + (T[:, 2, 2] - T[:, 0, 0])**2)
                       + 3*(T[:, 0, 1]**2 + T[:, 1, 2]**2 + T[:, 0, 2]**2))

    def smin(T):
        return np.linalg.eigvalsh(T)[:, 0] if len(T) else np.zeros(0)
    # in-plane principal: eigenvalues of the projected tensor include a 0 along n -> take min over the in-plane ones
    def smin_ip(T, n):
        ev, evec = np.linalg.eigh(T)
        # drop the eigenvector closest to n
        dots = np.abs(np.einsum('kij,ki->kj', evec, n))
        drop = np.argmax(dots, axis=1)
        ev2 = np.where(np.arange(3)[None, :] == drop[:, None], np.inf, ev)
        return ev2.min(1)
    return dict(keys=keys, sig_outer=so, sig_inner=si, vm_outer=vm(so), vm_inner=vm(si),
                smin_outer=smin_ip(so, nrm), smin_inner=smin_ip(si, nrm), tau_core=tau_c, Q=Q, normals=nrm)


def hinge_forces(model, RF, F):
    """hinge forces on each face side of every hinge edge (incl. latched).  H_a = RF_a - own load share F[a].  Returns
    dict: per_metre (list of (edge, face, node, tension N/m, shear N/m, along, normal)), vertex point forces, and the
    maxima."""
    st = 2 if model.elem == 'S8R' else 1
    rows, vert = [], []
    for e, c in model.conn.items():
        if c != 'hinge': continue
        a_, b_ = model.edge_ends(e)
        t = model.pos[b_] - model.pos[a_]; t /= np.linalg.norm(t)
        for f, s in model.edge_faces(e):
            side = model.side_nodes(f, s)
            nf = model.face_normal(f)
            mvec = np.cross(nf, t)
            if np.dot(mvec, 0.5*(model.pos[a_] + model.pos[b_]) - model.face_center(f)) < 0: mvec = -mvec
            nseg = (len(side) - 1)//st
            for q, nd in enumerate(side):
                H = np.asarray(RF.get(nd, np.zeros(3)), float) - np.asarray(F.get(nd, np.zeros(3)), float)
                # nodes merged by a latch carry both faces: skip (their force is the latch force)
                if nd in getattr(model, 'latch_all_nodes', model.latch_nodes): continue
                if q == 0 or q == len(side) - 1:
                    vert.append((e, f, nd, H)); continue
                # tributary length (consistent S8R / S4 loads)
                if st == 2:
                    seg = q//2
                    Le = np.linalg.norm(model.xyz[side[2*seg + 2]] - model.xyz[side[2*seg]]) if q % 2 else None
                    if q % 2 == 1: w = 2*Le/3
                    else:
                        La = np.linalg.norm(model.xyz[side[q]] - model.xyz[side[q - 2]])
                        Lb = np.linalg.norm(model.xyz[side[q + 2]] - model.xyz[side[q]])
                        w = La/6 + Lb/6
                else:
                    La = np.linalg.norm(model.xyz[side[q]] - model.xyz[side[q - 1]])
                    Lb = np.linalg.norm(model.xyz[side[q + 1]] - model.xyz[side[q]])
                    w = (La + Lb)/2
                ten = float(np.dot(H, mvec))/w
                al = float(np.dot(H, t))/w
                no = float(np.dot(H, nf))/w
                rows.append((e, f, nd, ten, math.hypot(al, no), al, no))
    out = dict(per_metre=rows, vertex=vert)
    if rows:
        out['max_tension_N_per_m'] = max(r[3] for r in rows)
        out['max_compression_N_per_m'] = -min(r[3] for r in rows)
        out['max_shear_N_per_m'] = max(r[4] for r in rows)
    if vert:
        out['max_vertex_force_N'] = max(float(np.linalg.norm(v[3])) for v in vert)
    lat = []
    for nd in model.latch_nodes:
        lat.append((nd, np.asarray(RF.get(nd, np.zeros(3)), float) - np.asarray(F.get(nd, np.zeros(3)), float)))
    out['latch_nodes'] = lat
    return out


VERTEX_RADII = (0.0, 0.05, 0.1, 0.15, 0.2)


def vertex_distance(model, nodes):
    """distance of each node from the nearest net vertex (panel corner)"""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.array(list(model.pos.values())))
    d, _ = tree.query(np.array([model.xyz[nd] for nd in nodes]))
    return d


def hinge_line_resultant(model, RF, F, edges, faces):
    """resultant force applied by a hinge line on the given faces: sum of RF - own load over the UNIQUE nodes of those
    faces' sides along the given edges (merged nodes counted once).  Returns (vector, line length)."""
    nodes, Ltot = set(), 0.0
    for e in edges:
        a_, b_ = model.edge_ends(e)
        Ltot += float(np.linalg.norm(model.pos[b_] - model.pos[a_]))
        for f, s in model.edge_faces(e):
            if f in faces: nodes.update(model.side_nodes(f, s))
    H = np.zeros(3)
    for nd in nodes:
        H += np.asarray(RF.get(nd, np.zeros(3)), float) - np.asarray(F.get(nd, np.zeros(3)), float)
    return H, Ltot


def evaluate(model, res, combo, limits=None, exclude_radius=0.0, exclude_points=()):
    """all checks for one combination {case: factor} of a StaticResult.  exclude_points / exclude_radius: optionally
    report stress maxima also without nodes within the radius of given points (latch, point supports).  Returns dict."""
    C = res.combo(combo)
    sec = model.section
    info = sec.info
    U = C['U']
    nodes = [nd for nd in model.xyz if nd in U]
    Ua = np.array([U[nd] for nd in nodes])
    uv = Ua @ model.up
    out = dict(combo=combo)
    out['u_vert_max'] = float(np.abs(uv).max()); out['u_vert_node'] = int(nodes[int(np.argmax(np.abs(uv)))])
    ut = np.linalg.norm(Ua, axis=1)
    out['u_tot_max'] = float(ut.max()); out['u_tot_node'] = int(nodes[int(np.argmax(ut))])
    Fall = C['F']
    out['applied'] = force_total(Fall)
    out['reaction'] = total_reaction(model, C['RF'], Fall)
    out['reaction_plus_applied_rel'] = float(np.linalg.norm(out['reaction'] + out['applied'])/
                                             max(np.linalg.norm(out['applied']), 1e-300))
    out['reactions_by_label'] = {k: v['sum'] for k, v in reactions_by_label(model, C['RF'], Fall).items()}
    out['spindle'] = spindle_forces(model, U) if model.springs else {}
    fc = face_and_core(model, C['S']) if C['S'] else None
    if fc is not None:
        keys = fc['keys']
        vm = np.maximum(fc['vm_outer'], fc['vm_inner'])
        comp = -np.minimum(fc['smin_outer'], fc['smin_inner'])
        out['face_vm_max'] = float(vm.max()); out['face_vm_at'] = keys[int(np.argmax(vm))]
        out['face_comp_max'] = float(comp.max()); out['face_comp_at'] = keys[int(np.argmax(comp))]
        out['core_tau_max'] = float(fc['tau_core'].max()); out['core_tau_at'] = keys[int(np.argmax(fc['tau_core']))]
        if exclude_points and exclude_radius > 0:
            X = np.array([model.xyz[nd] for nd, f in keys])
            far = np.ones(len(keys), bool)
            for p in exclude_points: far &= np.linalg.norm(X - np.asarray(p), axis=1) > exclude_radius
            out['face_vm_max_excl'] = float(vm[far].max()); out['face_comp_max_excl'] = float(comp[far].max())
            out['core_tau_max_excl'] = float(fc['tau_core'][far].max())
        if 'sigma_w' in info: out['wrinkling_ratio'] = out['face_comp_max']/info['sigma_w']
        out['n_stress_points'] = len(keys)
        out['face_vm_max_xyz'] = model.xyz[out['face_vm_at'][0]].tolist()
        out['face_comp_max_xyz'] = model.xyz[out['face_comp_at'][0]].tolist()
        out['core_tau_max_xyz'] = model.xyz[out['core_tau_at'][0]].tolist()
        # panel corners (net vertices), supports and point latches are stress singularities (point forces): maxima
        # outside a radius r from every net vertex (r in m); report these with the k-convergence
        dv = vertex_distance(model, [nd for nd, f in keys])
        prof = {}
        for r_ in VERTEX_RADII:
            sel = dv > r_
            if sel.any():
                prof['%g' % r_] = dict(vm=float(vm[sel].max()), comp=float(comp[sel].max()), tau=float(fc['tau_core'][sel].max()),
                                       n=int(sel.sum()))
        out['away_from_vertices'] = prof
        out['face_vm_percentiles_50_90_99'] = [float(x) for x in np.percentile(vm, [50, 90, 99])]
    hf = hinge_forces(model, C['RF'], Fall)
    for k_ in ('max_tension_N_per_m', 'max_compression_N_per_m', 'max_shear_N_per_m', 'max_vertex_force_N'):
        if k_ in hf: out['hinge_' + k_] = hf[k_]
    if hf['per_metre']:
        rows = hf['per_metre']
        dvh = vertex_distance(model, [r_[2] for r_ in rows])
        ten = np.array([r_[3] for r_ in rows]); sh = np.array([r_[4] for r_ in rows])
        hp = {}
        for r_ in VERTEX_RADII:
            sel = dvh > r_
            if sel.any():
                hp['%g' % r_] = dict(tension=float(ten[sel].max()), compression=float(-ten[sel].min()), shear=float(sh[sel].max()))
        out['hinge_away_from_vertices'] = hp
    out['latch_forces'] = [(nd, v.tolist()) for nd, v in hf['latch_nodes']]
    return out


def capacity_by_radius(model, res, case_S, radii=VERTEX_RADII, **kw):
    """capacity() with the stress limits evaluated outside a radius r from every net vertex, for each r"""
    fc = face_and_core(model, res.S[kw.get('G', 'G')])
    dv = vertex_distance(model, [nd for nd, f in fc['keys']]) if fc is not None else None
    out = {}
    for r_ in radii:
        out['%g' % r_] = capacity(model, res, case_S, exclude=(dv <= r_) if (dv is not None and r_ > 0) else None, **kw)
    return out


def capacity(model, res, case_S, G='G', span=None, gG=1.35, gQ=1.5, sk_ref=1.0, s_hi=100.0, exclude=None):
    """s_k,max (in units of the reference s_k of the S case, linear) at which each limit is first reached:
    SLS vertical deflection |u_v(G + s S)| = span/200; ULS face von Mises = f_y/gamma_M; ULS face compression = sigma_w;
    ULS core shear = tau_c,max.  Exact per point (quadratic / linear in s; compression by bisection).  exclude: boolean
    mask over the stress keys to ignore (e.g. latch zone).  Returns {limit: s_k,max} and the governing one."""
    info = model.section.info
    out = {}
    # SLS deflection (vertical, linear)
    nodes = [nd for nd in model.xyz if nd in res.U[G]]
    uG = np.array([res.U[G][nd] for nd in nodes]) @ model.up
    uS = np.array([res.U[case_S][nd] for nd in nodes]) @ model.up
    if span is not None:
        lim = span/200
        s = np.full(len(nodes), np.inf)
        for sign in (1, -1):
            with np.errstate(divide='ignore', invalid='ignore'):
                r = (sign*lim - uG)/uS
            s = np.where((r >= 0) & np.isfinite(r), np.minimum(s, r), s)
        s = np.where(np.abs(uG) >= lim, 0.0, s)
        out['SLS_deflection_span/200'] = float(s.min())*sk_ref
    fcG = face_and_core(model, res.S[G]); fcS = face_and_core(model, res.S[case_S])
    if fcG is not None and fcS is not None:
        assert fcG['keys'] == fcS['keys']
        m = np.ones(len(fcG['keys']), bool) if exclude is None else ~np.asarray(exclude)

        def vm2_coeffs(A, B):
            # VM^2 of (A + s B) for 3x3 tensors = a s^2 + b s + c (quadratic form)
            def q(X, Y):
                return 0.5*((X[:, 0, 0] - X[:, 1, 1])*(Y[:, 0, 0] - Y[:, 1, 1]) + (X[:, 1, 1] - X[:, 2, 2])*(Y[:, 1, 1] - Y[:, 2, 2])
                            + (X[:, 2, 2] - X[:, 0, 0])*(Y[:, 2, 2] - Y[:, 0, 0])) + 3*(X[:, 0, 1]*Y[:, 0, 1] + X[:, 1, 2]*Y[:, 1, 2] + X[:, 0, 2]*Y[:, 0, 2])
            return q(B, B), 2*q(A, B), q(A, A)

        def solve_quad(a, b, c, lim):
            disc = b*b - 4*a*(c - lim**2)
            with np.errstate(invalid='ignore', divide='ignore'):
                r = (-b + np.sqrt(np.maximum(disc, 0)))/(2*a)
            r = np.where((disc >= 0) & (a > 0), r, np.inf)
            r = np.where(c >= lim**2, 0.0, r)
            return r
        fyd = info['f_y']/info['gamma_M'] if info.get('f_y') else None
        if fyd:
            best = np.inf
            for side in ('sig_outer', 'sig_inner'):
                A = gG*fcG[side]; B = gQ*fcS[side]
                a, b, c = vm2_coeffs(A, B)
                r = solve_quad(a, b, c, fyd)
                best = min(best, float(r[m].min()))
            out['ULS_face_vm_fy/gM'] = best*sk_ref
        if info.get('sigma_w'):
            sw = info['sigma_w']
            best = np.inf
            for side in ('sig_outer', 'sig_inner'):
                A = gG*fcG[side][m]; B = gQ*fcS[side][m]
                nrm = fcG['normals'][m]

                def comp(s):
                    T = A + s*B
                    ev, evec = np.linalg.eigh(T)
                    dots = np.abs(np.einsum('kij,ki->kj', evec, nrm))
                    drop = np.argmax(dots, axis=1)
                    ev2 = np.where(np.arange(3)[None, :] == drop[:, None], np.inf, ev)
                    return -ev2.min(1)
                svals = np.linspace(0, s_hi, 401)
                prev = comp(0.0)
                found = np.full(len(A), np.inf)
                found[prev >= sw] = 0.0
                for q in range(1, len(svals)):
                    cur = comp(svals[q])
                    newly = (cur >= sw) & ~np.isfinite(found)
                    if newly.any():
                        lo, hi = np.full(newly.sum(), svals[q - 1]), np.full(newly.sum(), svals[q])
                        Asub, Bsub, nsub = A[newly], B[newly], nrm[newly]
                        for _ in range(40):
                            md = 0.5*(lo + hi)
                            T = Asub + md[:, None, None]*Bsub
                            ev, evec = np.linalg.eigh(T)
                            dots = np.abs(np.einsum('kij,ki->kj', evec, nsub))
                            drop = np.argmax(dots, axis=1)
                            ev2 = np.where(np.arange(3)[None, :] == drop[:, None], np.inf, ev)
                            cm = -ev2.min(1)
                            hi = np.where(cm >= sw, md, hi); lo = np.where(cm >= sw, lo, md)
                        found[newly] = hi
                    if np.isfinite(found).any(): break
                best = min(best, float(found.min()))
            out['ULS_face_compression_sigma_w'] = best*sk_ref
        if info.get('tau_c_max'):
            # |Q| is the norm of a vector linear in s: recompute the traction vectors
            tG = _core_vectors(model, res.S[G], fcG['keys'])*gG
            tS = _core_vectors(model, res.S[case_S], fcG['keys'])*gQ
            a = np.einsum('ki,ki->k', tS, tS); b = 2*np.einsum('ki,ki->k', tG, tS); c = np.einsum('ki,ki->k', tG, tG)
            r = solve_quad(a, b, c, info['tau_c_max'])
            out['ULS_core_shear'] = float(r[m].min())*sk_ref
    gov = min(out, key=out.get) if out else None
    return dict(limits=out, governing=gov, s_k_max=(out[gov] if gov else None))


def _core_vectors(model, S, keys):
    """core shear stress vectors tau_c (Q/d) at the stress keys"""
    mid = tensor(np.array([S[(nd, f, 0)] if (nd, f, 0) in S else 0.5*(np.asarray(S[(nd, f, 1)]) + np.asarray(S[(nd, f, -1)]))
                           for nd, f in keys]))
    nrm = np.array([model.face_normal(f) for nd, f in keys])
    trac = np.einsum('kij,kj->ki', mid, nrm)
    tau = trac - np.einsum('ki,ki->k', trac, nrm)[:, None]*nrm
    info = model.section.info
    return tau*model.section.t/(info['d'] if 'd' in info else model.section.t)


# ---------------------------------------------------------------- rigid motions (kept)
def rigid_basis(X):
    X = np.asarray(X, float)
    c = X.mean(0)
    cols = []
    for a in range(3):
        e = np.zeros(3); e[a] = 1
        cols.append(np.tile(e, len(X)))
        cols.append(np.cross(e, X - c).ravel())
    Q, _ = np.linalg.qr(np.array(cols).T)
    return Q


def remove_rigid(X, V):
    Q = rigid_basis(X)
    v = np.asarray(V, float).ravel()
    return (v - Q @ (Q.T @ v)).reshape(-1, 3)


def rigidity_residual(X, V):
    v = np.asarray(V, float).ravel()
    r = remove_rigid(X, V).ravel()
    return np.linalg.norm(r)/max(np.linalg.norm(v), 1e-300)


def fit_rigid(X, V):
    """least-squares rigid velocity field V = a + w x X; returns a, w, relative residual"""
    X, V = np.asarray(X, float), np.asarray(V, float)
    A = np.zeros((3*len(X), 6))
    for q, x in enumerate(X):
        A[3*q:3*q + 3, :3] = np.eye(3)
        A[3*q:3*q + 3, 3:] = -_skew(x)
    sol, *_ = np.linalg.lstsq(A, V.ravel(), rcond=None)
    res = np.linalg.norm(A @ sol - V.ravel())/max(np.linalg.norm(V), 1e-300)
    return sol[:3], sol[3:], float(res)


def mode_shapes(res):
    eig = res['dat']['eigen']
    disp = [b for b in res['frd']['blocks'] if b['name'] == 'DISP']
    return [(eig[q][3] if q < len(eig) else None, eig[q][1] if q < len(eig) else None, disp[q]['data'])
            for q in range(len(disp))]


def add_support_321(model, A=None, B=None, C=None):
    """statically determinate 3-2-1 support (translations only) at three vertices"""
    import itertools
    A = A or (0, 0); B = B or (model.m, 0); C = C or (0, model.n)
    a, b, c = (np.asarray(model.pos[v], float) for v in (A, B, C))
    I = np.eye(3); best = None
    for d1, d2 in itertools.combinations(range(3), 2):
        for d3 in range(3):
            det = abs(np.linalg.det(np.array([np.cross(b - a, I[d1]), np.cross(b - a, I[d2]), np.cross(c - a, I[d3])])))
            if best is None or det > best[0]: best = (det, d1, d2, d3)
    det, d1, d2, d3 = best
    assert det > 1e-9
    model.add_support(('vertex', A), [1, 2, 3], 'A'); model.add_support(('vertex', B), [d1 + 1, d2 + 1], 'B')
    model.add_support(('vertex', C), [d3 + 1], 'C')
    return '%s:123 %s:%d%d %s:%d (det %.3g)' % (A, B, d1 + 1, d2 + 1, C, d3 + 1, det)


def lower_vertices(model, verts, frac=0.5):
    """the lower zigzag vertices of a foot row: those whose height (pos . up) is below the row's mid height"""
    h = {v: float(np.dot(model.pos[v], model.up)) for v in verts}
    lo, hi = min(h.values()), max(h.values())
    return [v for v in verts if h[v] <= lo + frac*(hi - lo)]


# ---------------------------------------------------------------- one-call driver for a hall variant
def analyse_hall(model, workdir, span, rise, along, foot_labels=('foot1', 'foot2'), s_k=1000.0, sls_limit=None,
                 buckle=True, buckle_cases=('S1', 'S2+', 'S2-'), nmodes_buckle=6, grid=0.01, reading='triangles',
                 threads=3, exclude_radius=0.0, tag='hall', buckle_accuracy=1e-7):
    """the snow check of a hall for one support/lock variant (supports, spindle and latch already on the model):
    1 mechanism gate (rigid-panel SVD + FE frequency; returns early if a zero mode remains);
    2 load cases G, S1, S2+ (mu_3 on +x), S2- (mu_3 on -x) at s_k (N/m2), one static run;
    3 SLS = G + S and ULS = 1.35 G + 1.5 S for each snow case: evaluate() (displacements, face von Mises, compression vs
      sigma_w, core shear, hinge forces per metre, reactions per label, foot thrust per metre, spindle force);
    4 capacity s_k,max per limit (linear); 5 *BUCKLE under ULS (nmodes >= 4: one mode can return a wrong factor);
    6 hand check of the arch thrust H = q L^2/(8 f) (and the 3-hinge value for the snow width l_s) vs the FE foot thrust.
    span, rise: of the hall (m); along: unit vector of the hall length (rails)."""
    t0 = time.time()
    out = dict(tag=tag, k=(model.ka, model.kb), span=span, rise=rise, s_k=s_k)
    mech = model.check_mechanisms(workdir, tag + '_mech', nmodes=6, threads=threads)
    out['mechanism'] = dict(svd_zero=mech['svd']['n_zero'], svd_smallest=mech['svd']['smallest'][:4], fe_zero=mech['fe_zero'],
                            fe_negative=mech['fe_negative'], fe_lambda=mech['fe_lambda'], fe_f_Hz=mech['fe_f_Hz'],
                            noise=mech['noise'], ok=mech['ok'])
    out['f1_Hz'] = mech['fe_f_Hz'][0] if mech['fe_f_Hz'] else None
    if not mech['ok']:
        out['refused'] = 'mechanism: static, buckling and capacity not computed'
        return out
    frame = plan_frame(model.up, along)
    s1 = snow_cylindrical(model, s_k, 'S1', along=along, grid=grid)
    s2p = snow_cylindrical(model, s_k, 'S2', along=along, drift=+1, reading=reading, rc=s1['rc'], env=s1['env'])
    s2m = snow_cylindrical(model, s_k, 'S2', along=along, drift=-1, reading=reading, rc=s1['rc'], env=s1['env'])
    out['snow'] = dict(l_s=s1['l_s'], xL=s1['xL'], xR=s1['xR'], h=s1['h'], b=s1['b'], h_over_b=s1['h_over_b'], mu3=s1['mu3'],
                       single_interval=s1['single_interval'], totals={'S1': float(-np.dot(s1['total'], model.up)),
                                                                     'S2+': float(-np.dot(s2p['total'], model.up)),
                                                                     'S2-': float(-np.dot(s2m['total'], model.up))},
                       loaded_faces_outer_normal_down=len(s1['loaded_faces_outer_normal_down']),
                       load_on_faces_outer_normal_down_N=float(sum(v for f_, v in s1['loaded_faces_outer_normal_down'])),
                       dropped_on_outer_down_faces_N=s1['dropped_on_outer_down_faces'],
                       reading=reading)
    G = gravity_forces(model)
    cases = {'G': G, 'S1': s1['forces'], 'S2+': s2p['forces'], 'S2-': s2m['forces']}
    out['G_total'] = float(-np.dot(force_total(G), model.up))
    R = model.run_static(workdir, tag + '_static', cases, threads=threads)
    out['stress_map'] = dict(missing=R.smap['missing'], n=R.smap['n'], ambiguous=R.smap['ambiguous'])
    sec = model.section.info
    L_hall = None
    ex = frame[0]
    excl = list(model.xyz[nd] for nd in model.latch_nodes) if exclude_radius > 0 else ()
    res = {}
    for S_ in ('S1', 'S2+', 'S2-'):
        for lim, coef in (('SLS', {'G': 1.0, S_: 1.0}), ('ULS', {'G': 1.35, S_: 1.5})):
            ev = evaluate(model, R, coef, exclude_radius=exclude_radius, exclude_points=excl)
            # foot thrust: across component of each foot's reaction, per metre of foot (along-extent of its supports)
            ft = {}
            rbx = {k_: v for k_, v in reactions_by_label(model, R.combo(coef)['RF'], R.combo(coef)['F'],
                                                          exclude_spindle=True).items()} if model.springs else None
            for lab in foot_labels:
                nodes = [nd for nodes_, dofs, l_ in model.supports if l_ == lab for nd in nodes_]
                if not nodes: continue
                ys = [float(np.dot(model.xyz[nd], frame[1])) for nd in nodes]
                Lf = max(ys) - min(ys)
                Rl = ev['reactions_by_label'].get(lab, np.zeros(3))
                ft[lab] = dict(across_total=float(np.dot(Rl, ex)), vertical_total=float(np.dot(Rl, model.up)),
                               along_total=float(np.dot(Rl, frame[1])), foot_length=Lf,
                               across_per_m=float(np.dot(Rl, ex))/Lf if Lf > 0 else None)
                if rbx is not None and lab in rbx:      # thrust of the shells on the foot (spindle left out)
                    Rs = rbx[lab]['sum']
                    pv = [float(np.dot(v_, ex)) for v_ in rbx[lab]['R'].values()]
                    ft[lab].update(across_total_shells=float(np.dot(Rs, ex)),
                                   across_per_m_shells=float(np.dot(Rs, ex))/Lf if Lf > 0 else None,
                                   across_per_vertex_shells_min_max=(min(pv), max(pv)) if pv else None)
            ev['foot_thrust'] = ft
            if sls_limit is None: sls_limit = span/200
            if lim == 'SLS':
                ev['u_vert_ratio'] = ev['u_vert_max']/sls_limit; ev['u_tot_ratio'] = ev['u_tot_max']/sls_limit
            else:
                if sec.get('f_y'): ev['vm_ratio_fyd'] = ev.get('face_vm_max', 0)/(sec['f_y']/sec['gamma_M'])
                if sec.get('tau_c_max'): ev['core_ratio'] = ev.get('core_tau_max', 0)/sec['tau_c_max']
            for k_ in ('latch_forces',): ev.pop(k_, None)
            res['%s_%s' % (lim, S_)] = ev
    out['combos'] = res
    cap = {}
    for S_ in ('S1', 'S2+', 'S2-'):
        cap[S_] = capacity_by_radius(model, R, S_, span=span, sk_ref=s_k/1000.0)
    out['capacity_kN_per_m2_by_vertex_radius'] = cap
    # hand check of the arch thrust (uniform case = S1 over l_s; and the full-span formula)
    q1 = 0.8*s_k
    ls = s1['l_s']
    out['thrust_hand'] = dict(q=q1, H_full_span_qL2_8f=q1*span**2/(8*rise), H_3hinge_over_ls=q1*ls*(2*span - ls)/(8*rise),
                              G_H_qL2_8f=sec.get('mass_per_area', model.section.rho*model.section.t)*G_ACC*span**2/(8*rise))
    # FE thrust of S1 alone: total across reaction of each foot, per metre of the loaded hall length
    # L_load = S1 total / (0.8 s_k l_s) (the snow strip length), and per metre of the support extent
    fe = {}
    C = R.combo({'S1': 1.0})
    rb = reactions_by_label(model, C['RF'], C['F'])
    rbs = reactions_by_label(model, C['RF'], C['F'], exclude_spindle=True)     # shells only
    L_load = out['snow']['totals']['S1']/(0.8*s_k*ls)
    fe['L_load'] = L_load
    for lab in foot_labels:
        if lab in rb:
            nodes = [nd for nodes_, dofs, l_ in model.supports if l_ == lab for nd in nodes_]
            ys = [float(np.dot(model.xyz[nd], frame[1])) for nd in nodes]
            Htot = float(np.dot(rb[lab]['sum'], ex))
            Hsh = float(np.dot(rbs[lab]['sum'], ex))
            fe[lab] = dict(H_total=Htot, per_m_loaded_length=Htot/L_load, per_m_support_extent=Htot/(max(ys) - min(ys)),
                           H_total_shells=Hsh, per_m_loaded_length_shells=Hsh/L_load)
    if model.springs:
        fe['spindle_S1'] = spindle_forces(model, C['U'])
    out['thrust_FE_S1'] = fe
    if buckle:
        bk = {}
        for S_ in buckle_cases:
            F = add_forces(G, cases[S_], factors=[1.35, 1.5])
            fac, rb_ = model.run_buckle(workdir, '%s_buckle_%s' % (tag, S_.replace('+', 'p').replace('-', 'm')), F,
                                        nmodes=nmodes_buckle, threads=threads, accuracy=buckle_accuracy)
            bk[S_] = dict(factors=fac, mode1=buckle_mode_info(model, rb_))
        out['buckle_ULS'] = bk
    out['seconds'] = time.time() - t0
    return out


def buckle_mode_info(model, r, mode=0):
    """where the buckling mode sits: node of the largest displacement (position), and the fraction of shell nodes with
    |u| > 30 % of the maximum (small = local mode, e.g. at a point support or the spindle)"""
    D = [b for b in r['frd']['blocks'] if b['name'] == 'DISP']
    if not D: return None
    d = D[mode]['data']
    ids = [k for k in model.xyz if k in d]
    if not ids: return None
    U = np.array([np.linalg.norm(d[k][:3]) for k in ids])
    j = int(np.argmax(U))
    x = model.xyz[ids[j]]
    vk = list(model.pos.keys())
    dv = np.linalg.norm(np.array([model.pos[v] for v in vk]) - x, axis=1)
    q = int(np.argmin(dv))
    v = vk[q]
    on_boundary = v[0] in (0, model.m) or v[1] in (0, model.n)
    X = np.array([model.xyz[k] for k in ids])
    far = np.linalg.norm(X - x, axis=1) > 0.5
    return dict(max_at=x.tolist(), participation_30pc=float(np.mean(U > 0.3*U.max())), nearest_vertex=v,
                nearest_vertex_dist=float(dv[q]), vertex_on_net_boundary=bool(on_boundary),
                mean_amplitude_beyond_0p5m=float(U[far].mean()/U.max()) if far.any() else None)
