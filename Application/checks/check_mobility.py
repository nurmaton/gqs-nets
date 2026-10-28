# Mobility of the halls as systems of rigid panels joined by hinges, at sampled positions of the motion built with ../gqs.py.
# Compatibility matrix: unknowns (omega, v) of every panel; per hinge 5 rows (equal velocity of the two panels at the hinge
# midpoint, 3 rows; the components of omega_a - omega_b normal to the hinge, 2 rows); a locked hinge adds the component along
# the hinge. The dimension of its null space is the number of singular values below 1e-9 times the largest.
# Quantities: the null-space dimension (7 = six rigid motions and one more) at sampled positions, for (iii) away from
# t = 1/sqrt(2); 6 with one hinge locked (several hinges and positions); for (iii) at t = 1/sqrt(2) the null-space dimension
# and the number of hinges with dihedral angle 0 (oriented angle of the two face normals about the hinge); the smallest
# |dihedral angle| at the sampled positions; the period-two translation V_{i+2,j} - V_ij (largest deviation of a single
# difference from the mean, in both positions and at all samples); the length of that translation in metres in both
# positions and its change (the rows of blocks lengthen during folding).
# usage: python3 checks/check_mobility.py [--samples N]     output: out/check_mobility.json
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, math, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import halls as HL

NS = int(sys.argv[sys.argv.index('--samples') + 1]) if '--samples' in sys.argv else 9
REL = 1e-9
T0 = time.time()
HL.header('Mobility of the halls (rigid panels and hinges), period-two translation')


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def perp_basis(e):
    e = e / np.linalg.norm(e)
    a = np.array([1.0, 0, 0]) if abs(e[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(e, a); u /= np.linalg.norm(u)
    return u, np.cross(e, u)


def hinges(P):
    """(panel a, panel b, end p, end q) of every common edge of two panels of the grid P (ni, nj, 3), panels i-major"""
    NI, NJ = P.shape[:2]
    idx = lambda i, j: i * (NJ - 1) + j
    out = []
    for i in range(NI - 1):
        for j in range(NJ - 1):
            if i + 1 < NI - 1: out.append((idx(i, j), idx(i + 1, j), P[i + 1, j], P[i + 1, j + 1]))
            if j + 1 < NJ - 1: out.append((idx(i, j), idx(i, j + 1), P[i, j + 1], P[i + 1, j + 1]))
    return out


def compat(P, lock=None):
    NI, NJ = P.shape[:2]; nf = (NI - 1) * (NJ - 1)
    hh = hinges(P); rows = []
    for k, (a, b, p, q) in enumerate(hh):
        x = 0.5 * (p + q); e = q - p; u, w = perp_basis(e)
        R = np.zeros((5, 6 * nf))
        R[0:3, 6 * a:6 * a + 3] = -skew(x); R[0:3, 6 * a + 3:6 * a + 6] = np.eye(3)
        R[0:3, 6 * b:6 * b + 3] = skew(x); R[0:3, 6 * b + 3:6 * b + 6] = -np.eye(3)
        R[3, 6 * a:6 * a + 3] = u; R[3, 6 * b:6 * b + 3] = -u
        R[4, 6 * a:6 * a + 3] = w; R[4, 6 * b:6 * b + 3] = -w
        rows.append(R)
        if lock is not None and k in lock:
            L = np.zeros((1, 6 * nf)); ee = e / np.linalg.norm(e)
            L[0, 6 * a:6 * a + 3] = ee; L[0, 6 * b:6 * b + 3] = -ee
            rows.append(L)
    return np.vstack(rows), len(hh)


def nullity(P, lock=None):
    A, nh = compat(P, lock)
    s = np.sort(np.linalg.svd(A, compute_uv=False))
    z = int((s < REL * s[-1]).sum())
    return z, s, nh


def dihedral_deg(P):
    """oriented angle between the normals of the two faces at every hinge, about the hinge (0: the two faces are coplanar)"""
    d1 = P[1:, 1:] - P[:-1, :-1]; d2 = P[:-1, 1:] - P[1:, :-1]
    N = np.cross(d1, d2); N /= np.linalg.norm(N, axis=-1, keepdims=True)
    NI, NJ = P.shape[:2]; out = []
    for i in range(1, NI - 1):
        for j in range(NJ - 1):
            e = P[i, j + 1] - P[i, j]; e /= np.linalg.norm(e); n1, n2 = N[i - 1, j], N[i, j]
            out.append(math.degrees(math.atan2(np.dot(np.cross(n1, n2), e), np.dot(n1, n2))))
    for j in range(1, NJ - 1):
        for i in range(NI - 1):
            e = P[i + 1, j] - P[i, j]; e /= np.linalg.norm(e); n1, n2 = N[i, j - 1], N[i, j]
            out.append(math.degrees(math.atan2(np.dot(np.cross(n1, n2), e), np.dot(n1, n2))))
    return np.array(out)


vals, det = {}, {}
T_SPECIAL = 1 / math.sqrt(2)
for h in HL.HALL_KEYS:
    H = HL.HALLS[h]; s = H['metres_per_unit']
    print('\nhall (%s)' % h)
    net = HL.Net(h)
    td, tt = H['t_deployed'], H['t_transport']
    ts = list(np.linspace(td, tt, NS))
    if h == 'iii':
        ts = [t for t in ts if abs(t - T_SPECIAL) > 1e-3]
    nulls, mind, p2 = [], [], []
    for t in ts:
        P, _ = net.hall_grid(float(t))
        z, sv, nh = nullity(P)
        th = dihedral_deg(P)
        T, dev = HL.period_two(P)
        nulls.append(z); mind.append(float(np.abs(th).min())); p2.append(dev * s)
        print('  t = %.5f: null space %d (singular values %s | next %.1e, largest %.1e); smallest |dihedral angle| %.3f deg; '
              'period two: |T| = %.4f m, deviation %.1e m' % (t, z, ' '.join('%.0e' % x for x in sv[:z]), sv[z], sv[-1],
                                                              np.abs(th).min(), np.linalg.norm(T) * s, dev * s))
    vals['%s.nullspace_at_samples' % h] = sorted(set(nulls))
    vals['%s.min_abs_dihedral_deg' % h] = min(mind)
    det['%s.samples_t' % h] = [float(t) for t in ts]
    det['%s.nullspace' % h] = nulls
    # one hinge locked: several hinges, at three positions
    locks = []
    for t in (ts[0], ts[len(ts) // 2], ts[-1]):
        P, _ = net.hall_grid(float(t))
        _, nh = compat(P)
        for k in sorted({0, nh // 4, nh // 2, 3 * nh // 4, nh - 1}):
            z, _, _ = nullity(P, lock={k})
            locks.append(z)
    vals['%s.nullspace_one_hinge_locked' % h] = sorted(set(locks))
    print('  one hinge locked (hinges 0, nh/4, nh/2, 3nh/4, nh - 1; t = %.4f, %.4f, %.4f): null space %s'
          % (ts[0], ts[len(ts) // 2], ts[-1], sorted(set(locks))))
    # period two and the length of the translation in both positions (stored and rebuilt)
    TT = {}
    for st in HL.STATES:
        P, _ = net.hall_grid(H['t_%s' % st])
        T, dev = HL.period_two(P)
        Td, devd = HL.period_two(HL.load_V(h, st))
        TT[st] = float(np.linalg.norm(Td))
        vals['%s.period_two_deviation_%s_m' % (h, st)] = max(dev * s, devd)
        print('  %-9s period two: |T| = %.6f m (rebuilt %.6f m); largest deviation %.1e m (stored), %.1e m (rebuilt)'
              % (st, np.linalg.norm(Td), np.linalg.norm(T) * s, devd, dev * s))
    vals['%s.period_two_deviation_samples_m' % h] = max(p2)
    vals['%s.row_lengthening_pct' % h] = 100 * (TT['transport'] / TT['deployed'] - 1)
    vals['%s.row_period_deployed_m' % h] = TT['deployed']; vals['%s.row_period_transport_m' % h] = TT['transport']
    print('  rows of blocks: the translation by two rows grows from %.4f to %.4f m (%+.2f %%)'
          % (TT['deployed'], TT['transport'], vals['%s.row_lengthening_pct' % h]))
    if h == 'iii':
        P0, _ = net.hall_grid(T_SPECIAL)
        z0, sv0, nh = nullity(P0)
        th0 = dihedral_deg(P0)
        flat = int((np.abs(th0) < 1e-6).sum())
        Pa, _ = net.hall_grid(T_SPECIAL - 1e-3); Pb, _ = net.hall_grid(T_SPECIAL + 1e-3)
        ch = np.sign(dihedral_deg(Pa)) != np.sign(dihedral_deg(Pb))
        vals['iii.nullspace_at_1_over_sqrt2'] = z0
        vals['iii.flat_hinges_at_1_over_sqrt2'] = flat
        vals['iii.hinges'] = nh
        det['iii.special'] = dict(t=T_SPECIAL, singular_values=sv0[:24], max_abs_angle_of_flat_deg=float(np.abs(th0[np.abs(th0) < 1e-6]).max()),
                                  min_abs_angle_of_others_deg=float(np.abs(th0[np.abs(th0) >= 1e-6]).min()),
                                  sign_changes_across=int(ch.sum()), sign_changing_equals_flat=bool(np.array_equal(ch, np.abs(th0) < 1e-6)))
        print('  t = 1/sqrt(2): null space %d (next singular value %.1e); dihedral angle 0 (|theta| < 1e-6 deg, largest %.1e deg) at %d of %d '
              'hinges, the others >= %.4f deg; the same %d hinges change sign between t = 1/sqrt(2) -+ 0.001: %s'
              % (z0, sv0[z0], det['iii.special']['max_abs_angle_of_flat_deg'], flat, nh, det['iii.special']['min_abs_angle_of_others_deg'],
                 int(ch.sum()), det['iii.special']['sign_changing_equals_flat']))
path = HL.write_result('check_mobility', dict(values=vals, details=det, seconds=time.time() - T0))
print('\nwritten %s (%.0f s)' % (os.path.relpath(path, HL.ROOT), time.time() - T0))
