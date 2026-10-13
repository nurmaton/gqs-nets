# Coincident faces between the axis-aligned box parts of one container (possible z-fighting or alpha-layer flicker).
# coincident(parts) -> list of dicts; parts = [(name, V)] with V the vertices of an axis-aligned box. Two faces of different
# parts coincide when they lie within PLANE_TOL of one plane and overlap in an area above AREA_TOL (1e-5 m2); harmless by
# construction: two opaque faces facing each other, and two opaque faces of the same material facing the same way;
# translucent faces never are. A coincidence is hidden when the overlap is covered by third opaque parts containing its
# plane, or faces down onto the ground.
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.
import numpy as np
PLANE_TOL, AREA_TOL = 2e-4, 1e-5
SKINS = {'c_wall_l': 'skin', 'c_wall_r': 'skin', 'c_wall_rear': 'skin', 'c_roof': 'roof'}   # translucent (MAT_SKIN, MAT_ROOF of scene.py)
def material(name):
    b = name.split('.')[0]
    return SKINS.get(b, 'floor' if b == 'c_floorplate' else 'steel')
def is_box(V, tol=1e-6):
    lo, hi = V.min(0), V.max(0)
    return bool(np.all(np.min(np.minimum(np.abs(V - lo), np.abs(V - hi)), axis=0) <= tol) and np.all(np.minimum(np.abs(V - lo), np.abs(V - hi)) <= tol))
def coincident(parts):
    B = [(n, V.min(0), V.max(0)) for n, V in parts]
    opaque = [(n, lo, hi) for n, lo, hi in B if material(n) in ('steel', 'floor')]
    out = []
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            ni, li, hi_ = B[i]; nj, lj, hj = B[j]
            for ax in range(3):
                o = [k for k in range(3) if k != ax]
                for si, sj in ((-1, -1), (1, 1), (-1, 1), (1, -1)):
                    sgn = si
                    pi = hi_[ax] if si > 0 else li[ax]; pj = hj[ax] if sj > 0 else lj[ax]
                    if abs(pi - pj) > PLANE_TOL: continue
                    a0 = max(li[o[0]], lj[o[0]]); a1 = min(hi_[o[0]], hj[o[0]]); b0 = max(li[o[1]], lj[o[1]]); b1 = min(hi_[o[1]], hj[o[1]])
                    if a1 - a0 <= 1e-4 or b1 - b0 <= 1e-4 or (a1 - a0) * (b1 - b0) <= AREA_TOL: continue   # float-noise slivers
                    plane = 0.5 * (pi + pj)
                    mi, mj = material(ni), material(nj); opq = mi in ('steel', 'floor') and mj in ('steel', 'floor')
                    if opq and si != sj: harmless = 'contact of two opaque solids'
                    elif opq and mi == mj: harmless = 'same opaque material, same facing'
                    else: harmless = None
                    ground = (ax == 2 and si < 0 and sj < 0 and abs(plane) <= 1e-6)
                    # hidden: the overlap rectangle is covered exactly by the union of third opaque parts that contain its plane
                    cov = [(nk, lk, hk) for nk, lk, hk in opaque if nk not in (ni, nj) and lk[ax] + 1e-5 < plane < hk[ax] - 1e-5]
                    rest = [(a0, a1, b0, b1)]; used = set()
                    for nk, lk, hk in cov:
                        c0, c1, d0, d1 = lk[o[0]], hk[o[0]], lk[o[1]], hk[o[1]]; new = []
                        for (r0, r1, s0, s1) in rest:
                            if c1 <= r0 or c0 >= r1 or d1 <= s0 or d0 >= s1: new.append((r0, r1, s0, s1)); continue
                            used.add(nk.split('.')[0])
                            if r0 < c0: new.append((r0, c0, s0, s1))                      # left of the cover
                            if c1 < r1: new.append((c1, r1, s0, s1))                      # right of the cover
                            m0, m1 = max(r0, c0), min(r1, c1)
                            if s0 < d0: new.append((m0, m1, s0, d0))                      # below
                            if d1 < s1: new.append((m0, m1, d1, s1))                      # above
                        rest = [r for r in new if r[1] - r[0] > 1e-7 and r[3] - r[2] > 1e-7]
                        if not rest: break
                    hider = '+'.join(sorted(used)) if (cov and not rest) else None
                    out.append(dict(a=ni, b=nj, axis='xyz'[ax], facing=[('+' if si > 0 else '-') + 'xyz'[ax], ('+' if sj > 0 else '-') + 'xyz'[ax]],
                                    plane=float(plane), gap=float(abs(pi - pj)), area_m2=float((a1 - a0) * (b1 - b0)), materials=[mi, mj],
                                    hidden=('on the ground' if ground else ('inside ' + hider if hider else None)), harmless=harmless,
                                    visible=not (ground or hider or harmless)))
    return out
