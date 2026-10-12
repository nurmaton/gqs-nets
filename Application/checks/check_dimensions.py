# Sizes of the halls in both positions, from the stored vertices and 80 mm panels (data/), in the standing frame of halls.py.
# Quantities: profile length l (length of the midline) in both positions and its change; the mirror symmetry of the profile
# (the two feet of (i) are mirror images); deployed span and rise of the
# midline, the length L along the hall, the outer top of the crown above the crown of the midline; transport width W,
# height H and length L with the panels; the panel thickness; the lowest panel point below the foot line on each side
# (the plinths of (i) or the knee walls of (ii), (iii) carry it); the shape of the deployed profile: the root mean square of the distances
# of the midline vertices from the best-fitting parabola with vertical axis and from the best-fitting circle (orthogonal
# distances, least squares), and which of the two is nearer; the gap between the feet of the packed tube (iii); the fit of (i) on
# the low-bed semi-trailer (height above the road, road limits, extended length) and of (ii), (iii) through the door and on
# the floor of the 20-ft container (constants and their sources in data/halls.json).
# usage: python3 checks/check_dimensions.py            output: out/check_dimensions.json
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, time, itertools
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import halls as HL

T0 = time.time()
HL.header('Sizes of the halls (stored vertices and panels, standing frame)')
SP = HL.SPEC
vals, det = {}, {}


def convex_distance(A, B):
    """distance between the convex hulls of the point sets A and B (least squares over the two simplices, SLSQP)"""
    from scipy.optimize import minimize
    na, nb = len(A), len(B)

    def fun(x):
        q = x[:na] @ A - x[na:] @ B; return q @ q

    def jac(x):
        q = x[:na] @ A - x[na:] @ B; return np.concatenate([2 * A @ q, -2 * B @ q])
    x0 = np.concatenate([np.ones(na) / na, np.ones(nb) / nb])
    cons = [dict(type='eq', fun=lambda x: x[:na].sum() - 1), dict(type='eq', fun=lambda x: x[na:].sum() - 1)]
    r = minimize(fun, x0, jac=jac, bounds=[(0, 1)] * (na + nb), constraints=cons, method='SLSQP',
                 options=dict(ftol=1e-16, maxiter=500))
    return float(np.sqrt(max(r.fun, 0.0)))


def parabola_distances(c, P):
    """orthogonal distances of the points P (x, z) from the parabola z = c0 x^2 + c1 x + c2 (roots of the cubic of the foot point)"""
    a, b, c0 = c
    out = []
    for p, q in P:
        r = np.roots([2 * a * a, 3 * a * b, b * b + 2 * a * (c0 - q) + 1, b * (c0 - q) - p])
        x = r[np.abs(r.imag) < 1e-9].real
        out.append(np.hypot(x - p, a * x * x + b * x + c0 - q).min())
    return np.array(out)


def profile_fits(P):
    """least-squares parabola (vertical axis) and circle through the points P (x, z), orthogonal distances: rms, largest, parameters"""
    from scipy.optimize import least_squares
    x, z = P[:, 0], P[:, 1]
    c = least_squares(lambda c: parabola_distances(c, P), np.linalg.lstsq(np.c_[x * x, x, np.ones_like(x)], z, rcond=None)[0],
                      xtol=1e-14, ftol=1e-14).x
    dp = parabola_distances(c, P)
    s_ = np.linalg.lstsq(np.c_[x, z, np.ones_like(x)], x * x + z * z, rcond=None)[0]
    g = [s_[0] / 2, s_[1] / 2, np.sqrt(s_[2] + s_[0] ** 2 / 4 + s_[1] ** 2 / 4)]
    g = least_squares(lambda g: np.hypot(x - g[0], z - g[1]) - g[2], g, xtol=1e-14, ftol=1e-14).x
    dc = np.abs(np.hypot(x - g[0], z - g[1]) - g[2])
    rms = lambda d: float(np.sqrt(np.mean(d * d)))
    return dict(parabola=dict(rms=rms(dp), max=float(dp.max()), coef=c.tolist()), circle=dict(rms=rms(dc), max=float(dc.max()), centre_radius=list(g)))


for h in HL.HALL_KEYS:
    H = HL.HALLS[h]
    print('\nhall (%s)' % h)
    ell = {}
    for st in HL.STATES:
        V0 = HL.load_V(h, st); P0 = HL.load_panels(h, st)
        Vs, Ps, R, sh = HL.to_standing(V0, P0, h, st)
        # the stored arrays are in this frame already: the frame of halls.py must reproduce them
        frame_res = float(max(np.abs(Vs - V0).max(), np.abs(Ps - P0).max()))
        vals['%s.%s.frame_reproduced_m' % (h, st)] = frame_res
        pts = np.vstack([Vs.reshape(-1, 3), Ps.reshape(-1, 3)])
        lo, hi = pts.min(0), pts.max(0)
        mid = HL.midline(Vs)
        ell[st] = HL.polyline_length(mid)
        # mirror symmetry of the profile: the midline and the even-i and odd-i polylines against their mirror images (x -> -x,
        # j -> reversed) about the middle of the feet; for (i) this makes the two feet mirror images
        xc = 0.5 * (mid[0, 0] + mid[-1, 0])
        X = Vs[..., [0, 2]] - [xc, 0.0]
        polys = [HL.midline(Vs) - [xc, 0.0], X[0::2].mean(0), X[1::2].mean(0)]
        mdev = max(float(np.abs(P - np.stack([-P[::-1, 0], P[::-1, 1]], 1)).max()) for P in polys)
        vals['%s.profile_mirror_deviation_%s_m' % (h, st)] = mdev
        vals['%s.ell_%s_m' % (h, st)] = ell[st]
        # panel thickness: extent of each panel along the normal of its face
        Q = HL.face_quads(Vs)
        nrm = np.cross(Q[:, 2] - Q[:, 0], Q[:, 3] - Q[:, 1]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        proj = np.einsum('pkc,pc->pk', Ps, nrm)
        thick = proj.max(1) - proj.min(1)
        det['%s.%s' % (h, st)] = dict(box_lo=lo, box_hi=hi, midline_xz=mid, panel_thickness_min_median_max=[thick.min(), np.median(thick), thick.max()])
        print('  %-9s frame of halls.py reproduces the stored arrays to %.1e m; profile length l = %.4f m; box W x H x L = %.4f x %.4f x %.4f m; '
              'panel thickness %.5f-%.5f m; profile mirror-symmetric to %.1e m' % (st, frame_res, ell[st], hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1],
                                                                                   thick.min(), thick.max(), mdev))
        if st == 'deployed':
            vals['%s.panel_thickness_m' % h] = float(np.median(thick))
            span = float(abs(mid[-1, 0] - mid[0, 0])); rise = float(mid[:, 1].max() - mid[0, 1])
            vals['%s.span_m' % h] = span; vals['%s.rise_m' % h] = rise; vals['%s.L_deployed_m' % h] = float(hi[1] - lo[1])
            vals['%s.feet_level_difference_m' % h] = float(abs(mid[0, 1] - mid[-1, 1]))
            vals['%s.crown_offset_m' % h] = float(hi[2] - rise)
            fit = profile_fits(mid)
            vals['%s.profile_parabola_rms_m' % h] = fit['parabola']['rms']; vals['%s.profile_circle_rms_m' % h] = fit['circle']['rms']
            vals['%s.profile_nearer_parabola_than_circle' % h] = bool(fit['parabola']['rms'] < fit['circle']['rms'])
            det['%s.profile_fits' % h] = fit
            print('  deployed profile (midline, %d vertices): distance from the best parabola rms %.4f m (largest %.4f m), from the best '
                  'circle (radius %.3f m) rms %.4f m (largest %.4f m): nearer a %s' % (len(mid), fit['parabola']['rms'], fit['parabola']['max'],
                  fit['circle']['centre_radius'][2], fit['circle']['rms'], fit['circle']['max'],
                  'parabola' if fit['parabola']['rms'] < fit['circle']['rms'] else 'circle'))
            low = {}
            for sgn in (-1, 1):
                side = Ps.reshape(-1, 3)[sgn * Ps.reshape(-1, 3)[:, 0] > 0]
                low[sgn] = float(side[:, 2].min())
            vals['%s.lowest_panel_below_foot_line_m' % h] = -min(low.values())
            det['%s.lowest_panel_point_z_by_side' % h] = low
            print('  deployed: span %.4f m, rise %.4f m (feet level to %.1e m), L %.4f m; outer top of the crown %.4f m above the '
                  'crown of the midline; lowest panel point %.4f / %.4f m below the foot line (x < 0 / x > 0); deployed width %.4f m'
                  % (span, rise, vals['%s.feet_level_difference_m' % h], hi[1] - lo[1], hi[2] - rise, -low[-1], -low[1], hi[0] - lo[0]))
        else:
            W, Ht, L = float(hi[0] - lo[0]), float(hi[2] - lo[2]), float(hi[1] - lo[1])
            vals['%s.W_transport_m' % h] = W; vals['%s.H_transport_m' % h] = Ht; vals['%s.L_transport_m' % h] = L
            if h == 'iii':
                # gap between the feet of the packed tube: smallest distance between a panel of the first and of the last column
                i0, i1, j0, j1 = HL.window(h); nj = j1 - j0
                F = HL.face_labels(h)
                A_ = [k for k, f in enumerate(F) if f[1] == H['faces_j'][0]]; B_ = [k for k, f in enumerate(F) if f[1] == H['faces_j'][1]]
                best = (np.inf, None)
                for a, b in itertools.product(A_, B_):
                    if np.any(Ps[a].min(0) > Ps[b].max(0) + 0.2) or np.any(Ps[b].min(0) > Ps[a].max(0) + 0.2): continue
                    d = convex_distance(np.unique(Ps[a], axis=0), np.unique(Ps[b], axis=0))
                    if d < best[0]: best = (d, (F[a], F[b]))
                vals['iii.tube_gap_m'] = best[0]
                det['iii.tube_gap_faces'] = best[1]
                print('  transport: gap between the feet of the tube %.6f m (faces F%s and F%s)' % (best[0], best[1][0], best[1][1]))
    ch = ell['transport'] / ell['deployed'] - 1
    vals['%s.ell_change_pct' % h] = 100 * ch
    print('  profile length: %.4f -> %.4f m, change %+.2f %%' % (ell['deployed'], ell['transport'], 100 * ch))

# ------------------------------------------------------------------ transport: road (i) and container (ii), (iii)
tr, rd, ct = SP['trailer'], SP['road'], SP['container_20ft']
Hi = vals['i.H_transport_m']; Wi = vals['i.W_transport_m']; Li = vals['i.L_transport_m']
top = tr['well_top_above_road_m'] + Hi
ext = tr['length_closed_m'] + tr['extension_m']
vals['i.height_above_road_m'] = top
vals['i.fits_road_height'] = bool(top <= rd['max_height_m'])
vals['i.fits_road_width'] = bool(Wi <= rd['max_width_m'])
vals['i.trailer_extended_length_m'] = ext
vals['i.fits_extended_well'] = bool(Li <= tr['well_length_closed_m'] + tr['extension_m'])
vals['i.trailer_exceeds_length_limits'] = bool(ext > rd['max_length_articulated_m'] and ext - tr['kingpin_to_front_m'] > rd['max_kingpin_to_rear_m'])
print('\nhall (i) on the low-bed semi-trailer: well top %.2f m + H %.4f m = %.4f m above the road (limit %.2f m); W %.4f m (limit %.2f m); '
      'L %.4f m in the well of %.2f + %.2f = %.2f m; trailer %.2f + %.2f = %.2f m long (limits %.2f m; king pin to rear %.2f m > %.2f m)'
      % (tr['well_top_above_road_m'], Hi, top, rd['max_height_m'], Wi, rd['max_width_m'], Li, tr['well_length_closed_m'], tr['extension_m'],
         tr['well_length_closed_m'] + tr['extension_m'], tr['length_closed_m'], tr['extension_m'], ext, rd['max_length_articulated_m'],
         ext - tr['kingpin_to_front_m'], rd['max_kingpin_to_rear_m']))
for h in ('ii', 'iii'):
    W, Ht, L = vals['%s.W_transport_m' % h], vals['%s.H_transport_m' % h], vals['%s.L_transport_m' % h]
    dw, dh = ct['door_min_W_H_m']; fl = ct['internal_min_L_W_H_m'][0]
    vals['%s.fits_door' % h] = bool(W <= dw and Ht <= dh)
    vals['%s.fits_floor' % h] = bool(L <= fl)
    print('hall (%s) in the 20-ft container: W x H = %.4f x %.4f m through the door %.3f x %.3f m (margins %.3f, %.3f m); L %.4f m on the floor '
          '%.3f m long' % (h, W, Ht, dw, dh, dw - W, dh - Ht, L, fl))
path = HL.write_result('check_dimensions', dict(values=vals, details=det, seconds=time.time() - T0))
print('\nwritten %s (%.0f s)' % (os.path.relpath(path, HL.ROOT), time.time() - T0))
