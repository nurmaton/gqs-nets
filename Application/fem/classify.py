# Buckling modes of a CalculiX *BUCKLE run: reader of the mode shapes in the .frd file and classification of every mode.
# Requires numpy and scipy.
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.
"""
Reader: streams the ASCII .frd and keeps every DISP block with its value.  ccx writes the static preload state first (value
0.0); block q (q >= 1) is buckling mode q.  The mapping is checked by order and by value (1e-5 relative).

Classification (rule A).  A mode u (translations at the shell nodes) is LOCAL if
    A30 < A30_LOCAL = 0.02   (A30: area of the region {|u| >= 0.3 |u|max} over the net area), or
    R_far < RFAR_LOCAL = 0.5 (R_far: the largest |u| farther than 0.5 m from the peak, over the peak);
it is then named by the place of its largest |u|: 'local-spindle' (within R_LOC = 0.3 m of a spindle end),
'local-support' (of a point support), 'local-vertex' (of another net vertex, a panel corner), 'local-other'.
Otherwise the mode is 'broad'.  Further measures are kept as evidence: fractions of the displacement norm
sum_i w_i |u_i|^2 (w_i: tributary area) and of the deformation norm sum_e int |grad_s u|^2 dA near the supports, the
spindle ends, the latch and the single net vertex that holds the most; part30, the fraction of nodes with |u| > 0.3 max.
A second rule (rule B, in fe.py) counts the face centres that move at least 30 % of the peak.
"""
import numpy as np
from scipy.spatial import cKDTree

R_LOC = 0.3
THR = 0.5
ALPHA = 0.3
A30_LOCAL = 0.02
RFAR_LOCAL = 0.5


def read_frd_disp(path):
    """-> (values, blocks) with blocks = list of (ids array, U array (n, 3)) for every DISP block, in file order"""
    vals, blocks = [], []
    cur_val, in_disp, ids, U = None, False, None, None
    with open(path) as fh:
        for line in fh:
            if line.startswith('  100CL'):
                cur_val = float(line[12:24]); in_disp = None; continue
            if in_disp is None and line.startswith(' -4'):
                in_disp = line[5:13].strip() == 'DISP'
                if in_disp: ids, U = [], []
                continue
            if in_disp:
                if line.startswith(' -1'):
                    ids.append(int(line[3:13])); U.append((float(line[13:25]), float(line[25:37]), float(line[37:49])))
                elif line.startswith(' -3'):
                    vals.append(cur_val); blocks.append((np.array(ids), np.array(U))); in_disp = False
    return vals, blocks


def map_blocks(vals, factors, tol=1e-5):
    """block index of every mode: by order (block q = mode q, block 0 = static preload with value 0) and by value"""
    assert abs(vals[0]) < 1e-12, 'block 0 is not the static preload (value %r)' % vals[0]
    assert len(vals) == len(factors) + 1, (len(vals), len(factors))
    out = []
    for q, lam in enumerate(factors):
        bv = [i for i, v in enumerate(vals) if i > 0 and abs(v - lam) <= tol*max(1.0, abs(lam))]
        assert (q + 1) in bv, ('order/value mapping disagree', q + 1, lam, bv, vals[q + 1])
        out.append(q + 1)
    return out


class Classifier:
    def __init__(self, model, r_loc=R_LOC):
        M = self.M = model
        self.ids = np.array(sorted(M.xyz))                      # shell nodes (spindle plain nodes are not in M.xyz)
        self.idx = {nd: q for q, nd in enumerate(self.ids)}
        self.X = np.array([M.xyz[nd] for nd in self.ids])
        w = np.zeros(len(self.ids))
        for f, con in M.elems:
            P = np.array([M.xyz[c] for c in con[:4]])
            A = 0.5*np.linalg.norm(np.cross(P[2] - P[0], P[3] - P[1]))
            for c in con: w[self.idx[c]] += A/len(con)
        self.w = w
        self.area = float(w.sum())
        self.vk = sorted(M.pos.keys())
        self.VP = np.array([M.pos[v] for v in self.vk])
        sup_v = sorted({v for nodes, dofs, lab in M.supports for v in self.vk
                        if any(nd in M.vertex_nodes[v] for nd in nodes)})
        self.sup_v = sup_v
        self.supP = np.array([M.pos[v] for v in sup_v])
        self.spinP = np.array([x for sp in M.springs for x in (sp['xa'], sp['xb'])]) if M.springs else np.zeros((0, 3))
        self.spin_v = [self.vk[int(np.argmin(np.linalg.norm(self.VP - x, axis=1)))] for x in self.spinP]
        self.latchP = np.array([M.xyz[nd] for nd in getattr(M, 'latch_nodes', [])]) if getattr(M, 'latch_nodes', []) \
            else np.zeros((0, 3))
        tree = cKDTree(self.X)
        self.r = r_loc

        def mask(P):
            mk = np.zeros(len(self.X), bool)
            for lst in (tree.query_ball_point(P, r_loc) if len(P) else []): mk[lst] = True
            return mk
        self.m_sup, self.m_spin, self.m_latch = mask(self.supP), mask(self.spinP), mask(self.latchP)
        self.vnb = tree.query_ball_point(self.VP, r_loc)           # nodes within r of every net vertex
        self.tree = tree
        # deformation norm: S8R element data (corner geometry is bilinear, mid-side nodes on straight edges)
        con = np.array([[self.idx[c] for c in cn] for f, cn in M.elems])
        assert con.shape[1] == 8, 'S8R expected'
        self.con = con
        Xc = self.X[con[:, :4]]                                            # (E, 4, 3)
        g = 1/np.sqrt(3.0)
        self.gp = []
        for xi in (-g, g):
            for eta in (-g, g):
                c4 = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
                dN = np.zeros((8, 2))
                for q, (a, b) in enumerate(c4):
                    dN[q, 0] = 0.25*a*(1 + eta*b)*(2*xi*a + eta*b)
                    dN[q, 1] = 0.25*b*(1 + xi*a)*(xi*a + 2*eta*b)
                dN[4] = (-xi*(1 - eta), -0.5*(1 - xi**2))
                dN[5] = (0.5*(1 - eta**2), -eta*(1 + xi))
                dN[6] = (-xi*(1 + eta), 0.5*(1 - xi**2))
                dN[7] = (-0.5*(1 - eta**2), -eta*(1 - xi))
                dG = np.array([[0.25*-(1 - eta), 0.25*-(1 - xi)], [0.25*(1 - eta), 0.25*-(1 + xi)],
                               [0.25*(1 + eta), 0.25*(1 + xi)], [0.25*-(1 + eta), 0.25*(1 - xi)]])   # bilinear geometry
                J = np.einsum('eci,ck->eik', Xc, dG)                       # (E, 3, 2)
                JtJ = np.einsum('eik,eil->ekl', J, J)
                detA = np.sqrt(np.linalg.det(JtJ))
                P = np.einsum('ekl,eil->eki', np.linalg.inv(JtJ), J)         # (E, 2, 3): grad_s = dU_dxi @ P
                self.gp.append((dN, P, detA))
        self.cent = Xc.mean(1)
        ctree = cKDTree(self.cent)

        def emask(P):
            mk = np.zeros(len(self.cent), bool)
            for lst in (ctree.query_ball_point(P, r_loc) if len(P) else []): mk[lst] = True
            return mk
        self.e_sup, self.e_spin, self.e_latch = emask(self.supP), emask(self.spinP), emask(self.latchP)
        self.evnb = ctree.query_ball_point(self.VP, r_loc)
        self.elem_area = sum(detA for dN, P, detA in self.gp)              # 2x2 Gauss weights are 1

    def deformation(self, A3):
        """element deformation energies int |grad_s u|^2 dA for nodal displacements A3 (n, 3)"""
        Ue = A3[self.con]                                                  # (E, 8, 3)
        out = np.zeros(len(self.con))
        for dN, P, detA in self.gp:
            dU = np.einsum('eci,ck->eik', Ue, dN)                          # (E, 3, 2)
            Gs = np.einsum('eik,ekj->eij', dU, P)                          # (E, 3, 3)
            out += detA*np.einsum('eij,eij->e', Gs, Gs)
        return out

    def mode(self, ids, U):
        pos = np.array([self.idx.get(int(i), -1) for i in ids])
        keep = pos >= 0
        A3 = np.zeros((len(self.ids), 3)); A3[pos[keep]] = U[keep]
        A = np.linalg.norm(A3, axis=1)
        e = self.w*A**2
        ge = self.deformation(A3); GE = ge.sum()
        g_v = np.array([ge[lst].sum()/GE for lst in self.evnb]); qg = int(np.argmax(g_v))
        E = e.sum()
        j = int(np.argmax(A)); xm = self.X[j]
        dv = np.linalg.norm(self.VP - xm, axis=1); qv = int(np.argmin(dv))
        f_v = np.array([e[lst].sum()/E for lst in self.vnb])
        qmax = int(np.argmax(f_v))
        near_peak = self.tree.query_ball_point(xm, self.r)
        far05 = np.linalg.norm(self.X - xm, axis=1) > 0.5
        d_sup = float(np.linalg.norm(self.supP - xm, axis=1).min())
        d_spin = float(np.linalg.norm(self.spinP - xm, axis=1).min()) if len(self.spinP) else None
        d_latch = float(np.linalg.norm(self.latchP - xm, axis=1).min()) if len(self.latchP) else None
        r = dict(f_sup=float(e[self.m_sup].sum()/E), f_spin=float(e[self.m_spin].sum()/E) if len(self.spinP) else 0.0,
                 f_latch=float(e[self.m_latch].sum()/E) if len(self.latchP) else 0.0,
                 f_peak=float(e[near_peak].sum()/E), f_vtx=float(f_v[qmax]), vtx=self.vk[qmax],
                 vtx_is_support=self.vk[qmax] in self.sup_v,
                 peak_xyz=xm.round(4).tolist(), peak_vertex=self.vk[qv], peak_vertex_dist=float(dv[qv]),
                 peak_dist_support=d_sup, peak_dist_spindle_end=d_spin, peak_dist_latch=d_latch,
                 part30=float(np.mean(A > 0.3*A.max())), f_beyond05=float(e[far05].sum()/E),
                 n_missing=int((~keep).sum()),
                 g_sup=float(ge[self.e_sup].sum()/GE), g_spin=float(ge[self.e_spin].sum()/GE) if len(self.spinP) else 0.0,
                 g_latch=float(ge[self.e_latch].sum()/GE) if len(self.latchP) else 0.0,
                 g_vtx=float(g_v[qg]), g_vtx_at=self.vk[qg], g_vtx_is_support=self.vk[qg] in self.sup_v)

        # peak region: nodes with |u| >= ALPHA max (the large-amplitude part of the mode), area weighted
        Pk = A >= ALPHA*A.max(); wP = self.w*Pk; WP = wP.sum()
        a_v = np.array([wP[lst].sum()/WP for lst in self.vnb]); qa = int(np.argmax(a_v))
        r.update(a_sup=float(wP[self.m_sup].sum()/WP), a_spin=float(wP[self.m_spin].sum()/WP) if len(self.spinP) else 0.0,
                 a_latch=float(wP[self.m_latch].sum()/WP) if len(self.latchP) else 0.0,
                 a_vtx=float(a_v[qa]), a_vtx_at=self.vk[qa], peak_area_m2=float(WP),
                 peak_extent_m=float(np.linalg.norm(self.X[Pk] - xm, axis=1).max()))

        def cls(sp, su, vt):
            if sp > THR: return 'local-spindle'
            if su > THR: return 'local-support'
            if vt > THR: return 'local-vertex'
            return 'broad'
        r['class_deform'] = cls(r['g_spin'], r['g_sup'], r['g_vtx'])
        r['class_disp'] = cls(r['f_spin'], r['f_sup'], r['f_vtx'])
        # primary rule
        far = np.linalg.norm(self.X - xm, axis=1) > 0.5
        r['R_far05'] = float(A[far].max()/A.max())               # largest |u| farther than 0.5 m from the peak / peak
        r['A30'] = float(WP/self.area)
        if r['A30'] < A30_LOCAL or r['R_far05'] < RFAR_LOCAL:
            if d_spin is not None and d_spin <= self.r: c = 'local-spindle'
            elif d_sup <= self.r: c = 'local-support'
            elif dv[qv] <= self.r: c = 'local-vertex'
            else: c = 'local-other'
        else: c = 'broad'
        r['class'] = c
        return r, A

    def vertex_amplitudes(self, A):
        """|u| at the net vertices (max over the coincident vertex nodes)"""
        return np.array([max(A[self.idx[nd]] for nd in self.M.vertex_nodes[v]) for v in self.vk])


# ---------------------------------------------------------------- mesh comparison (k = 4, 8, 12)
def sample_ids(M):
    """shell node ids at the face-parametric points (p/8, q/8), p, q = 0..8 not both odd: present in every S8R mesh with
    k a multiple of 4 (k = 4, 8, 12 -> grid index (p, q) * 2k/8), 65 points per face, each face with its own nodes
    (coincident hinge nodes kept apart).  Same order for every k."""
    assert M.Na % 8 == 0 and M.Nb % 8 == 0
    out = []
    for f in M.faces():
        g = M.face_grid[f]
        for p in range(9):
            for q in range(9):
                if p % 2 and q % 2: continue
                out.append(M.canon[g[(p*M.Na//8, q*M.Nb//8)]])
    return np.array(out)


def sample_vector(ids, U, sids):
    row = {int(i): q for q, i in enumerate(ids)}
    return np.concatenate([U[row[int(s)]] for s in sids])


def mac(a, b):
    return float(np.dot(a, b)**2/(np.dot(a, a)*np.dot(b, b)))
