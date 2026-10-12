# The 80 mm panels of a hall along its motion: construction and collision check.
# The motion is built with ../gqs.py at the positions t of data/motion_t.json (deployed to transport). The bare net is the outer
# surface; every face carries a rigid panel 80 mm thick on the inner side (the prism of the face along the inward normal);
# the hinge axes are the outer edges. At every hinge both panels are bevelled by the bisector plane of the hinge at the
# sampled position where the inner angle there is smallest (a fixed bevel in each panel). Then, at every sampled position,
# every two panels whose boxes come within MARGIN are tested by the separating-axis theorem: panels with a common hinge or a
# common vertex may touch (their overlap depth is reported), all others must be apart (their smallest gap is reported, a
# lower bound of the distance, and the exact distance of the closest pairs). Where two panels with only a common vertex
# overlap ((i)), both corners are cut by the plane through that vertex equidistant from the two face planes, at the
# position of their deepest overlap, and the check is repeated with the cut panels.
# Quantities: the number of overlapping pairs (none allowed), the number of corner cuts, the smallest gap between panels
# without a common vertex along the motion, the gap between the feet of the packed tube (iii), and the largest distance
# between the panels built here at the two end positions and the stored panels (data/hall_*_panels.npy).
# usage: python3 checks/check_panels.py [i ii iii]        output: out/check_panels_<hall>.json
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, json, math, time, itertools
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import halls as HL

THICK = HL.SPEC['panel_thickness_m']
MARGIN = 0.30
unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)


def motion(h):
    """the hall vertices (S, ni*nj, 3) in metres at the positions t of data/motion_t.json"""
    with open(os.path.join(HL.DATA, 'motion_t.json')) as fh:
        ts = json.load(fh)[h]
    net = HL.Net(h); s = HL.HALLS[h]['metres_per_unit']
    P = []
    for t in ts:
        V, _ = net.hall_grid(t)
        P.append(V.reshape(-1, 3) * s)
    i0, i1, j0, j1 = HL.window(h)
    return np.array(P), np.array(ts), i1 - i0, j1 - j0


def exact_dist(A, B):
    na, nb = len(A), len(B)

    def fun(x):
        q = x[:na] @ A - x[na:] @ B; return q @ q

    def jac(x):
        q = x[:na] @ A - x[na:] @ B; return np.concatenate([2 * A @ q, -2 * B @ q])
    x0 = np.concatenate([np.ones(na) / na, np.ones(nb) / nb])
    cons = [dict(type='eq', fun=lambda x: x[:na].sum() - 1), dict(type='eq', fun=lambda x: x[na:].sum() - 1)]
    r = minimize(fun, x0, jac=jac, bounds=[(0, 1)] * (na + nb), constraints=cons, method='SLSQP', options=dict(ftol=1e-16, maxiter=500))
    return math.sqrt(max(r.fun, 0.0))


def sat(VA, VB, NA, NB, EA, EB):
    """batched separating-axis test of convex polytopes: > 0 apart (a lower bound of the distance), <= 0 overlapping
    (minus the overlap depth along the best axis)"""
    p = len(VA)
    cr = np.cross(EA[:, :, None, :], EB[:, None, :, :]).reshape(p, -1, 3)
    ln = np.linalg.norm(cr, axis=-1, keepdims=True); ok = ln > 1e-6
    cr = np.where(ok, cr / np.where(ok, ln, 1.0), NA[:, :1, :])
    AX = np.concatenate([NA, NB, cr], 1)
    pa = np.einsum('pki,pvi->pkv', AX, VA); pb = np.einsum('pki,pvi->pkv', AX, VB)
    gap = np.maximum(pb.min(-1) - pa.max(-1), pa.min(-1) - pb.max(-1))
    return gap.max(-1)


class Panels:
    """the panels of the faces of a hall along a sampled motion P (S, (m+1)(n+1), 3)"""

    def __init__(self, P, m, n):
        self.P, self.m, self.n, self.S = P, m, n, len(P)
        vid = lambda i, j: i * (n + 1) + j
        self.faces = [(i, j) for i in range(m) for j in range(n)]; F = len(self.faces)
        fidx = {f: k for k, f in enumerate(self.faces)}
        self.cor = np.array([[vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)] for i, j in self.faces])
        C = P[:, self.cor]
        N = unit(np.cross(C[:, :, 2] - C[:, :, 0], C[:, :, 3] - C[:, :, 1]))
        v0, v1 = self.side_vote(0, N), self.side_vote(self.S - 1, N)
        self.sigma = 1.0 if v0 > 0.5 else -1.0
        self.vote = (v0, v1)
        self.NIN = self.sigma * N                                         # the direction in which the panels grow
        # hinges: face f's edge kf (corner kf -> kf+1) is face g's edge kg
        self.edges = []
        for (i, j) in self.faces:
            f = fidx[(i, j)]
            if i + 1 < m: self.edges.append((f, 1, fidx[(i + 1, j)], 3))
            if j + 1 < n: self.edges.append((f, 2, fidx[(i, j + 1)], 0))
        PHI = np.zeros((self.S, len(self.edges)))
        for e, (f, kf, g, kg) in enumerate(self.edges):
            wf = self.inface_perp(f, kf); wg = self.inface_perp(g, kg)
            ph = np.arctan2(np.linalg.norm(np.cross(wf, wg), axis=-1), (wf * wg).sum(-1))
            toward = (self.NIN[:, f] * wg).sum(-1) > 0
            PHI[:, e] = np.where(toward, ph, 2 * math.pi - ph)
        self.PHI = PHI
        self.phistar = PHI.min(0)                                          # smallest inner angle at each hinge along the motion
        self.bevel = {}
        for e, (f, kf, g, kg) in enumerate(self.edges):
            self.bevel[(f, kf)] = self.phistar[e] / 2; self.bevel[(g, kg)] = self.phistar[e] / 2
        self.share = {}
        vf = {}
        for f in range(F):
            for v in self.cor[f]: vf.setdefault(v, []).append(f)
        for v, fl in vf.items():
            for a, b in itertools.combinations(sorted(fl), 2): self.share[(a, b)] = self.share.get((a, b), 0) + 1

    def side_vote(self, k, N):
        """fraction of the faces whose normal points into the region bounded by the profile and the foot chord"""
        m, n = self.m, self.n
        V = self.P[k].reshape(m + 1, n + 1, 3)
        D = []
        for j in range(1, n):
            for par in (1, 0):
                I = [i for i in range(1, m) if i % 2 == par]
                if len(I) >= 2:
                    Q = V[I, j]; c = Q.mean(0); w = np.linalg.svd(Q - c)[2][0]; D.append(w * np.sign(w @ (D[0] if D else w)))
        d = np.mean(D, 0); d /= np.linalg.norm(d)
        a = np.cross(d, [0, 0, 1.0]); a /= np.linalg.norm(a); b = np.cross(d, a)
        prof = np.array([V[:, j].mean(0) for j in range(n + 1)]); prof2 = np.stack([prof @ a, prof @ b], 1)
        poly = Path(np.vstack([prof2, prof2[:1]]))
        cen = self.P[k][self.cor].mean(1); q = cen + 0.02 * N[k]
        return poly.contains_points(np.stack([q @ a, q @ b], 1)).mean()

    def inface_perp(self, f, k):
        X = self.P
        A = X[:, self.cor[f][k]]; B = X[:, self.cor[f][(k + 1) % 4]]; u = unit(B - A)
        c = X[:, self.cor[f]].mean(1) - A; w = c - (c * u).sum(-1, keepdims=True) * u
        return unit(w)

    def frame(self, k, f):
        """origin and rotation (columns e1, e2, e3 = growth direction) of face f at position k"""
        C = self.P[k][self.cor[f]]; o = C[0]; e1 = unit(C[1] - C[0]); e3 = self.NIN[k, f]
        e3 = unit(e3 - (e3 @ e1) * e1); e2 = np.cross(e3, e1)
        return o, np.stack([e1, e2, e3], -1)

    def build(self, cuts=None):
        """the panels as convex polytopes in the local frame of each face (vertices, face normals, edge directions)"""
        cuts = cuts or {}
        self.LOC = []; self.cut_count = sum(len(v) for v in cuts.values())
        for f in range(len(self.faces)):
            o, R = self.frame(0, f); L = (self.P[0][self.cor[f]] - o) @ R
            assert np.abs(L[:, 2]).max() < 1e-9
            Hs = [(np.array([0, 0, -1.0]), 0.0), (np.array([0, 0, 1.0]), THICK)]
            for k in range(4):
                A, B = L[k], L[(k + 1) % 4]; u = (B - A) / np.linalg.norm(B - A); c = L.mean(0) - A; w = c - (c @ u) * u; w /= np.linalg.norm(w)
                Hs.append((-w, -(w @ A)))
                if (f, k) in self.bevel:
                    psi = self.bevel[(f, k)]
                    if psi < math.pi / 2 - 1e-12:
                        nu = math.sin(psi) * w - math.cos(psi) * np.array([0, 0, 1.0])
                        Hs.append((-nu, -(nu @ A)))
            for k_, a_w, V in cuts.get(f, []):
                o_, R_ = self.frame(k_, f)
                Hs.append((R_.T @ a_w, float(a_w @ (V - o_))))
            Am = np.array([a for a, b in Hs]); bm = np.array([b for a, b in Hs])
            pts = []
            for i1, i2, i3 in itertools.combinations(range(len(Hs)), 3):
                M3 = Am[[i1, i2, i3]]
                if abs(np.linalg.det(M3)) < 1e-12: continue
                x = np.linalg.solve(M3, bm[[i1, i2, i3]])
                if np.all(Am @ x <= bm + 1e-10): pts.append(x)
            pts = np.unique(np.round(np.array(pts), 10), axis=0)
            act = [k for k in range(len(Hs)) if np.sum(np.abs(Am[k] @ pts.T - bm[k]) < 1e-9) >= 3]
            nrm = [Am[k] / np.linalg.norm(Am[k]) for k in act]
            ed = []
            for a_, b_ in itertools.combinations(act, 2):
                on = np.sum((np.abs(Am[a_] @ pts.T - bm[a_]) < 1e-9) & (np.abs(Am[b_] @ pts.T - bm[b_]) < 1e-9))
                if on >= 2:
                    dv = np.cross(Am[a_], Am[b_]); ed.append(dv / np.linalg.norm(dv))
            self.LOC.append(dict(V=pts, N=np.array(nrm), E=np.array(ed)))
        pad = lambda A, k: np.vstack([A, np.repeat(A[-1:], k - len(A), 0)])
        Vm = max(len(x['V']) for x in self.LOC); Nm = max(len(x['N']) for x in self.LOC); Em = max(len(x['E']) for x in self.LOC)
        self.LV = np.array([pad(x['V'], Vm) for x in self.LOC]); self.LN = np.array([pad(x['N'], Nm) for x in self.LOC])
        self.LE = np.array([pad(x['E'], Em) for x in self.LOC])
        self.zmax_min = min(x['V'][:, 2].max() for x in self.LOC)

    def world(self, k):
        F = len(self.faces)
        FR = [self.frame(k, f) for f in range(F)]
        O = np.stack([x[0] for x in FR]); R = np.stack([x[1] for x in FR])
        return (O[:, None, :] + np.einsum('fij,fvj->fvi', R, self.LV), np.einsum('fij,fvj->fvi', R, self.LN),
                np.einsum('fij,fvj->fvi', R, self.LE))

    def check(self, log_every=0):
        res = dict(edge_overlap=(0.0, None), vertex_overlap=(0.0, None), apart_overlap=(0.0, None), apart_gap=(1e9, None))
        PEN, closest, ends, last_gap = [], [], {}, None
        for k in range(self.S):
            WV, WN, WE = self.world(k)
            lo = WV.min(1) - MARGIN / 2; hi = WV.max(1) + MARGIN / 2
            ov = np.all((lo[:, None, :] <= hi[None, :, :]) & (lo[None, :, :] <= hi[:, None, :]), -1)
            I, J = np.nonzero(np.triu(ov, 1))
            g = np.concatenate([sat(WV[I[a:a + 2000]], WV[J[a:a + 2000]], WN[I[a:a + 2000]], WN[J[a:a + 2000]], WE[I[a:a + 2000]],
                                    WE[J[a:a + 2000]]) for a in range(0, len(I), 2000)]) if len(I) else np.zeros(0)
            cls = np.array([self.share.get((int(a), int(b)), 0) for a, b in zip(I, J)])
            for q in np.nonzero(g < -1e-6)[0]: PEN.append((int(k), int(I[q]), int(J[q]), int(cls[q]), float(-g[q])))
            for c_, key in ((2, 'edge_overlap'), (1, 'vertex_overlap'), (0, 'apart_overlap')):
                sel = cls == c_
                if sel.any():
                    pen = float(max(0.0, -g[sel].min())); a_ = int(np.argmin(np.where(sel, g, np.inf)))
                    if pen > res[key][0]: res[key] = (pen, dict(k=k, f=self.faces[I[a_]], g=self.faces[J[a_]]))
            sel = cls == 0
            if sel.any():
                gg = np.where(sel, g, np.inf); a_ = int(np.argmin(gg))
                if gg[a_] < res['apart_gap'][0]: res['apart_gap'] = (float(gg[a_]), dict(k=k, f=self.faces[I[a_]], g=self.faces[J[a_]]))
                for a_ in np.argsort(gg)[:3]: closest.append((float(gg[a_]), k, int(I[a_]), int(J[a_])))
                if k == self.S - 1: last_gap = (float(gg.min()), self.faces[I[int(np.argmin(gg))]], self.faces[J[int(np.argmin(gg))]])
            if k in (0, self.S - 1): ends[k] = WV.copy()
            if log_every and k % log_every == 0:
                print('    position %d/%d: %d pairs tested' % (k, self.S, len(I)), flush=True)
        closest.sort(); ex = []
        for gsat, k, a_, b_ in closest[:6]:
            WV, _, _ = self.world(k)
            ex.append(dict(k=k, f=self.faces[a_], g=self.faces[b_], sat=gsat, exact=exact_dist(WV[a_], WV[b_])))
        res['closest_exact'] = ex; res['PEN'] = PEN; res['ends'] = ends; res['last_gap'] = last_gap
        return res


def corner_cuts(pan, PEN):
    """the cut planes of the panel pairs with a common vertex that overlap: per pair, at the position of deepest overlap"""
    worst = {}
    for k_, a_, b_, c_, d_ in PEN:
        if c_ == 1 and d_ > worst.get((a_, b_), (0, -1))[0]: worst[(a_, b_)] = (d_, k_)
    cuts = {}
    for (a_, b_), (d_, k_) in worst.items():
        V = pan.P[k_, list(set(pan.cor[a_]) & set(pan.cor[b_]))[0]]; na, nb = pan.NIN[k_, a_], pan.NIN[k_, b_]
        for f_, nf, ng in ((a_, na, nb), (b_, nb, na)):
            cuts.setdefault(f_, []).append((k_, nf - ng, V))
    return cuts, worst


def compare_stored(h, pan, ends):
    """largest distance between the panels built here at the two end positions and the stored panels (stored frame)"""
    out = {}
    for st, k in (('deployed', 0), ('transport', pan.S - 1)):
        Vd = HL.load_V(h, st); Pd = HL.load_panels(h, st)
        R, c, res = HL.kabsch(pan.P[k], Vd)
        W = ends[k] @ R.T + c
        worst = 0.0
        for q in range(len(W)):
            A = np.unique(np.round(W[q], 9), axis=0); B = np.unique(np.round(Pd[q], 9), axis=0)
            D = np.linalg.norm(A[:, None] - B[None], axis=-1)
            worst = max(worst, D.min(1).max(), D.min(0).max())
        out[st] = dict(vertex_fit_m=res, panel_hausdorff_m=float(worst))
    return out


def run(h):
    t0 = time.time()
    print('\nhall (%s): building the motion with gqs.py ...' % h, flush=True)
    P, ts, m, n = motion(h)
    print('  %d positions, t = %.6f .. %.6f, %d x %d faces (%.0f s)' % (len(P), ts[0], ts[-1], m, n, time.time() - t0), flush=True)
    pan = Panels(P, m, n)
    print('  inner side: fraction of normals pointing inside %.3f (deployed), %.3f (transport); smallest inner angle at a hinge along '
          'the motion %.2f deg; %d of %d hinges bevelled' % (pan.vote[0], pan.vote[1], math.degrees(pan.phistar.min()),
                                                             int((pan.phistar < math.pi).sum()), len(pan.edges)), flush=True)
    pan.build()
    r0 = pan.check()
    out = dict(hall=h, positions=len(P), t_first=float(ts[0]), t_last=float(ts[-1]), hinges=len(pan.edges),
               bevelled_hinges=int((pan.phistar < math.pi).sum()), smallest_inner_angle_deg=math.degrees(pan.phistar.min()))
    pairs0 = {(p[1], p[2]) for p in r0['PEN']}
    print('  bevelled panels: %d overlapping pairs at %d positions (%d with a common vertex, %d with a common hinge, %d apart) (%.0f s)'
          % (len(pairs0), len({p[0] for p in r0['PEN']}), len({(p[1], p[2]) for p in r0['PEN'] if p[3] == 1}),
             len({(p[1], p[2]) for p in r0['PEN'] if p[3] == 2}), len({(p[1], p[2]) for p in r0['PEN'] if p[3] == 0}), time.time() - t0), flush=True)
    out['bevel_only_overlapping_pairs'] = len(pairs0)
    r = r0; cuts_n = 0
    if any(p[3] == 1 for p in r0['PEN']):
        cuts, worst = corner_cuts(pan, r0['PEN'])
        cuts_n = len(worst)
        verts = {list(set(pan.cor[a]) & set(pan.cor[b]))[0] for a, b in worst}
        pan.build(cuts)
        r = pan.check()
        out['corner_cut_vertices'] = len(verts)
        print('  corners cut at %d panel pairs with a common vertex (%d vertices, %d panels); checked again (%.0f s)'
              % (cuts_n, len(verts), len({f for ab in worst for f in ab}), time.time() - t0), flush=True)
    out['corner_cut_pairs'] = cuts_n
    pairs = {(p[1], p[2]) for p in r['PEN']}
    out['overlapping_pairs'] = len(pairs)
    out['edge_overlap_max_m'] = r['edge_overlap'][0]; out['vertex_overlap_max_m'] = r['vertex_overlap'][0]
    out['smallest_gap_m'] = r['apart_gap'][0]; out['smallest_gap_at'] = r['apart_gap'][1]
    out['smallest_gap_exact_m'] = r['closest_exact'][0]['exact'] if r['closest_exact'] else None
    out['transport_gap_m'] = r['last_gap'][0] if r['last_gap'] else None
    out['transport_gap_faces'] = r['last_gap'][1:] if r['last_gap'] else None
    out['closest_exact'] = r['closest_exact']
    out['thinnest_panel_m'] = pan.zmax_min
    cmp = compare_stored(h, pan, r['ends'])
    out['stored_panels'] = cmp
    i0, _, j0, _ = HL.window(h)
    lab = lambda f: 'F(%d,%d)' % (f[0] + i0, f[1] + j0)
    ga = r['apart_gap'][1]
    print('  final panels: %d overlapping pairs; largest overlap at a common hinge %.1e m, at a common vertex %.1e m (contact); '
          'smallest gap between panels without a common vertex %.4f m (exact %.4f m; %s and %s, position %d, t = %.4f); at the transport '
          'position %.4f m; thinnest panel %.4f m' % (len(pairs), out['edge_overlap_max_m'], out['vertex_overlap_max_m'], out['smallest_gap_m'],
                                                      out['smallest_gap_exact_m'], lab(ga['f']), lab(ga['g']), ga['k'], ts[ga['k']],
                                                      out['transport_gap_m'], pan.zmax_min))
    for st, c in cmp.items():
        print('  %-9s panels built here vs stored: vertices fit to %.1e m, panels agree to %.1e m (Hausdorff)' % (st, c['vertex_fit_m'], c['panel_hausdorff_m']))
    out['seconds'] = time.time() - t0
    return out


if __name__ == '__main__':
    HL.header('80 mm panels along the motion: bevels, corner cuts, collision check')
    which = [a for a in sys.argv[1:] if a in HL.HALL_KEYS] or list(HL.HALL_KEYS)
    for h in which:
        o = run(h)
        vals = {'%s.panel_overlapping_pairs' % h: o['overlapping_pairs'], '%s.corner_cut_vertices' % h: o.get('corner_cut_vertices', 0),
                '%s.smallest_panel_gap_m' % h: o['smallest_gap_m'], '%s.transport_panel_gap_m' % h: o['transport_gap_m'],
                '%s.stored_panels_agree_m' % h: max(c['panel_hausdorff_m'] for c in o['stored_panels'].values()),
                '%s.motion_positions' % h: o['positions']}
        path = HL.write_result('check_panels_%s' % h, dict(values=vals, details=o))
        print('  written %s (%.0f s)' % (os.path.relpath(path, HL.ROOT), o['seconds']))
