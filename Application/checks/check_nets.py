# Net data of the three halls, rebuilt with ../gqs.py and compared with the stored positions.
# For each hall: the number of rows of blocks (n - 2), the palindrome rho_nu = rho_{n-1-nu}, M = a_1 b_1 c_1 d_1 of the flat
# angles (gqs.py vdata), the numbers of panels and hinges, that every face of the hall is a central face of the net (so every
# hinge is an edge of a central face), the typical panel edge (median edge length; one gqs.py unit in metres), t and
# theta_1 of both positions;
# both positions rebuilt from gqs.py, scaled, and fitted to data/hall_*_V.npy by a rigid motion (largest residual), and the
# standing frame of halls.py compared with the stored one; the faces of the bare net checked for self-intersections
# (gqs.py self_intersections) at sampled positions of the motion between the two positions; the hinge lines against the
# panel corners in the deployed position (the stresses of the snow check are read outside a radius r of the corners): the
# hinge lengths, the largest distance from a point of a hinge line to the nearest panel corner (exact: the largest value
# of the distance to the nearest corner along a segment is reached at an end or where two corners are equally near), and
# the number of hinges with points farther than r = 0.2 and 0.15 m from every corner.
# usage: python3 checks/check_nets.py [--samples N]           output: out/check_nets.json
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, math, time, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import halls as HL
from fractions import Fraction

def hinge_corner_distance(A, B, P):
    """largest distance from a point of the segment [A, B] to the nearest of the points P (A and B among them)"""
    d = B - A; L = float(np.linalg.norm(d))
    # corners that can be nearest somewhere on the segment lie within L / 2 of it
    s_ = np.clip((P - A) @ d / (L * L), 0.0, 1.0)
    C = P[np.linalg.norm(A + s_[:, None] * d - P, axis=1) <= L / 2 + 1e-9]
    cand = [0.0, 1.0]
    for a in range(len(C)):
        for b in range(a + 1, len(C)):
            w = C[b] - C[a]; den = 2.0 * (d @ w)
            if abs(den) > 1e-15:
                s = (C[b] @ C[b] - C[a] @ C[a] - 2.0 * (A @ w)) / den
                if 0.0 < s < 1.0: cand.append(float(s))
    X = A + np.array(cand)[:, None] * d
    return float(np.linalg.norm(X[:, None, :] - C[None], axis=2).min(1).max())


NS = int(sys.argv[sys.argv.index('--samples') + 1]) if '--samples' in sys.argv else 25
T0 = time.time()
HL.header('Net data: rebuild of the halls with gqs.py')
G = HL.gqs()
md5 = HL.gqs_md5()
print('gqs.py: %s, md5 %s %s' % (os.path.abspath(HL.GQS_PATH), md5, '(as used)' if md5 == HL.GQS_MD5 else '(DIFFERS from %s)' % HL.GQS_MD5))
vals, det = {}, {'gqs_md5': md5}
for h in HL.HALL_KEYS:
    H = HL.HALLS[h]
    rows = H['rows']; n = H['n']; m = H['m']
    print('\nhall (%s): angles %s deg, m x n = %d x %d, (kappa_0, nu_0) = %s, rows %s, signs %s, e_0 = %d'
          % (h, H['angles_deg'], m, n, tuple(H['base']), rows, tuple(H['signs']), H['e0']))
    vals['%s.rows_count_is_n_minus_2' % h] = bool(len(rows) == n - 2)
    pal = all(rows[k] == rows[len(rows) - 1 - k] for k in range(len(rows)))
    vals['%s.rows_palindrome' % h] = bool(pal)
    print('  rho_1 .. rho_{n-2}: %d rows (n - 2 = %d); rho_nu = rho_{n-1-nu} for all nu: %s' % (len(rows), n - 2, pal))
    vd = G.vdata(*[math.radians(a) for a in H['angles_deg']])
    vals['%s.M' % h] = vd['M']
    print('  M = a_1 b_1 c_1 d_1 = %.15f (= %s to %.1e)' % (vd['M'], Fraction(vd['M']).limit_denominator(100),
                                                            abs(vd['M'] - float(Fraction(vd['M']).limit_denominator(100)))))
    faces = HL.face_labels(h); hinges = HL.hinge_list(h)
    vals['%s.panels' % h] = len(faces); vals['%s.hinges' % h] = len(hinges)
    central = all(1 <= i <= m - 2 and 1 <= j <= n - 2 for i, j in faces)
    vals['%s.all_faces_central' % h] = bool(central)
    print('  faces F_ij, %d <= i <= %d, %d <= j <= %d: %d panels, %d hinges; all are central faces (1 <= i <= m-2, 1 <= j <= n-2): %s'
          % (H['faces_i'][0], H['faces_i'][1], H['faces_j'][0], H['faces_j'][1], len(faces), len(hinges), central))
    s = H['metres_per_unit']
    net = HL.Net(h)
    for st in HL.STATES:
        t = H['t_%s' % st]
        Vg, worst = net.hall_grid(t)
        Vd = HL.load_V(h, st)
        R, c, res = HL.kabsch(Vg * s, Vd)
        # standing frame of halls.py against the stored frame: same rotation, the stored grid differs by a pure shift
        Vs, _, Rs, _ = HL.to_standing(Vg * s, None, h, st)
        dsh = Vd - Vs; shift_spread = float(np.abs(dsh - dsh.reshape(-1, 3).mean(0)).max())
        vals['%s.%s.rebuild_residual_m' % (h, st)] = res
        vals['%s.%s.standing_frame_residual_m' % (h, st)] = shift_spread
        vals['%s.t_%s' % (h, st)] = t
        vals['%s.theta1_%s_deg' % (h, st)] = HL.theta1_deg(t)
        E = np.concatenate([np.linalg.norm(np.diff(Vd, axis=0), axis=-1).ravel(), np.linalg.norm(np.diff(Vd, axis=1), axis=-1).ravel()])
        if st == 'deployed':
            vals['%s.typical_edge_m' % h] = float(np.median(E))
        det['%s.%s' % (h, st)] = dict(t=t, theta1_deg=HL.theta1_deg(t), build_mismatch_rad=worst, rigid_fit_residual_m=res,
                                      standing_frame_residual_m=shift_spread, edge_m_min_median_max=[E.min(), np.median(E), E.max()])
        print('  %-9s t = %.10f (theta_1 = %.6f deg): build mismatch %.1e rad; rebuilt x %.6f m/unit vs stored: rigid fit residual '
              '%.1e m; standing frame (halls.py) residual %.1e m; panel edges %.3f / %.3f / %.3f m (min / median / max)'
              % (st, t, HL.theta1_deg(t), worst, s, res, shift_spread, E.min(), np.median(E), E.max()))
    # the rounded t of the summary table
    for st, tr in zip(HL.STATES, {'i': (0.08, 0.32), 'ii': (0.7265, 1.4019), 'iii': (0.15, 0.75)}[h]):
        Vg, _ = net.hall_grid(tr)
        _, _, res = HL.kabsch(Vg * s, HL.load_V(h, st))
        det['%s.%s.rounded_t' % (h, st)] = dict(t=tr, deviation_m=res)
        vals['%s.%s.t_rounded_matches' % (h, st)] = bool(round(H['t_%s' % st], len(str(tr).split('.')[1])) == tr)
        print('  %-9s rounded t = %s: the stored position moves by at most %.1e m' % (st, tr, res))
    # hinge lines against the panel corners, deployed position (the radius r of the snow check is measured from the corners)
    Vd = HL.load_V(h, 'deployed'); i0, _, j0, _ = HL.window(h)
    P = Vd.reshape(-1, 3)
    hl, hd = [], []
    for (a, b) in hinges:
        A, B = Vd[a[0] - i0, a[1] - j0], Vd[b[0] - i0, b[1] - j0]
        hl.append(float(np.linalg.norm(B - A))); hd.append(hinge_corner_distance(A, B, P))
    hl, hd = np.array(hl), np.array(hd)
    lengths = {}
    for x in hl: lengths['%.3f' % x] = lengths.get('%.3f' % x, 0) + 1
    vals['%s.hinge_corner_distance_max_m' % h] = float(hd.max())
    for r_ in (0.2, 0.15):
        vals['%s.hinges_beyond_r%g' % (h, r_)] = int((hd > r_).sum())
        vals['%s.hinges_longer_than_2r%g' % (h, r_)] = int((hl > 2 * r_).sum())
    far = [hinges[q] for q in np.nonzero(hd > 0.2)[0]]
    det['%s.hinge_lengths_m' % h] = lengths
    det['%s.hinges_beyond_r0.2' % h] = [[list(a), list(b)] for a, b in far] if len(far) < len(hinges) else 'all'
    print('  hinge lines, deployed: lengths (m) %s; largest distance from a hinge point to the nearest panel corner %.4f m; hinges '
          'with points farther than r from every corner: %d (r = 0.2 m), %d (r = 0.15 m) of %d; hinges longer than 2r: %d, %d'
          % (', '.join('%s (%d)' % kv for kv in sorted(lengths.items())), hd.max(), (hd > 0.2).sum(), (hd > 0.15).sum(), len(hinges),
             (hl > 0.4).sum(), (hl > 0.3).sum()))
    if 0 < len(far) < len(hinges):
        print('  the %d hinges with points farther than 0.2 m from every corner (V_ij - V_kl): %s' % (
            len(far), ', '.join('%s-%s' % (a, b) for a, b in far)))
    # bare-net self-intersections along the motion (faces of the hall only)
    tt = np.linspace(H['t_deployed'], H['t_transport'], NS)
    bad = 0; t1 = time.time()
    corners = lambda f: [(f[0], f[1]), (f[0] + 1, f[1]), (f[0] + 1, f[1] + 1), (f[0], f[1] + 1)]
    for t in tt:
        pos, _ = net.positions(float(t))
        pr = G.self_intersections(pos, faces, corners)
        bad += len(pr)
    vals['%s.bare_net_self_intersections' % h] = bad
    det['%s.self_intersection_samples' % h] = [float(x) for x in tt]
    print('  bare net, %d positions from t = %.4f to %.4f: %d intersecting face pairs (%.0f s)'
          % (NS, tt[0], tt[-1], bad, time.time() - t1))
path = HL.write_result('check_nets', dict(values=vals, details=det, seconds=time.time() - T0))
print('\nwritten %s (%.0f s)' % (os.path.relpath(path, HL.ROOT), time.time() - T0))
