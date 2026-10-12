# Scene data of the renderings, from the stored positions and panels (data/): for each hall, in out/renderings/geo/,
#   hall_*_dep.obj, hall_*_tr.obj   the 80 mm panels of both positions as triangulated convex polytopes (one object per panel,
#                                   no joint: the stored panels exactly)
#   hall_*_dep_net.obj, ..._tr_net.obj  the faces of the bare net (quads)
#   hall_*_{dep,tr}_V.npy, hall_*_{dep,tr}_slabs_verified.npy  the stored vertices and panels (for the scene and its test)
#   hall_*.json                     frame data of the scene: per position t, theta_1, box, width W, length L, height H, the
#                                   midline, feet, crown; span and rise; panel thickness; the knee walls of (ii), (iii) (80 mm,
#                                   centred under the lowest panel corner of each side, top 0.1 mm below the lowest panel
#                                   point over the wall, length = panel material above it +- 0.15 m)
# and copies of renderings/constants.json and of the scene references (renderings/reference). Frame: x across the hall, y along
# it, z up (the standing frame of halls.py, data/halls.json).
# usage: python3 renderings/make_geo.py         requires numpy, scipy (a few seconds)
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, json, shutil
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
import halls as HL
from wallrange import wall_y_range

OUT = os.path.join(HL.OUT, 'renderings', 'geo')
JOINT = 0.0                        # m: no joint between panels; the stored panels exactly
KEY = {'deployed': 'dep', 'transport': 'tr'}


def polytope(pts):
    """the panel (convex hull of its points) with every side face moved inwards by JOINT/2; the two skins are kept"""
    pts = np.unique(np.round(np.asarray(pts, float), 12), axis=0)
    c = pts.mean(0)
    h = ConvexHull(pts)
    eq = np.unique(np.round(h.equations, 10), axis=0)
    areas = {}
    for simplex, e in zip(h.simplices, h.equations):
        key = tuple(np.round(e[:3], 6)); p = pts[simplex]
        areas[key] = areas.get(key, 0.0) + 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
    n_skin = np.array(max(areas, key=areas.get))
    side = np.abs(eq[:, :3] @ n_skin) < 0.999
    eq2 = eq.copy(); eq2[side, 3] += JOINT / 2
    q = HalfspaceIntersection(eq2, c).intersections
    return np.unique(np.round(q, 12), axis=0)


def hull_obj(fh, pts, name, base):
    q = polytope(pts)
    h = ConvexHull(q)
    fh.write('o %s\n' % name)
    for p in q:
        fh.write('v %.12f %.12f %.12f\n' % tuple(p))
    for simplex, eq in zip(h.simplices, h.equations):
        a, b, cc = simplex
        n = np.cross(q[b] - q[a], q[cc] - q[a])
        if n @ eq[:3] < 0:
            b, cc = cc, b
        fh.write('f %d %d %d\n' % (base + a + 1, base + b + 1, base + cc + 1))
    return len(q)


def write_obj(path, slabs, prefix='panel'):
    base = 0
    with open(path, 'w') as fh:
        for k, s in enumerate(slabs):
            base += hull_obj(fh, s, '%s_%03d' % (prefix, k), base)
    return base


def grid_obj(path, V):
    nc, nr = V.shape[:2]
    with open(path, 'w') as fh:
        fh.write('o net\n')
        for p in V.reshape(-1, 3):
            fh.write('v %.6f %.6f %.6f\n' % tuple(p))
        for i in range(nc - 1):
            for j in range(nr - 1):
                a = i * nr + j + 1
                fh.write('f %d %d %d %d\n' % (a, a + nr, a + nr + 1, a + 1))


def midline(V):
    X = V[..., [0, 2]]
    return (X[1::2].mean(0) + X[0::2].mean(0)) / 2


def case(h):
    name = 'hall_%s' % h
    H = HL.HALLS[h]
    C = json.load(open(os.path.join(HERE, 'constants.json')))
    info = dict(name=name, hw=H['feet_above_floor_m'], deck=None, door=None if h == 'i' else list(C['container_20ft']['door_min_W_H_m']),
                truck=(h == 'i'), states={})
    for st, key in KEY.items():
        V = HL.load_V(h, st); S = HL.load_panels(h, st)
        lo = np.minimum(V.reshape(-1, 3).min(0), S.reshape(-1, 3).min(0)); hi = np.maximum(V.reshape(-1, 3).max(0), S.reshape(-1, 3).max(0))
        obj = os.path.join(OUT, '%s_%s.obj' % (name, key))
        nv = write_obj(obj, [s for s in S])
        np.save(os.path.join(OUT, '%s_%s_slabs_verified.npy' % (name, key)), S)
        np.save(os.path.join(OUT, '%s_%s_V.npy' % (name, key)), V)
        grid_obj(os.path.join(OUT, '%s_%s_net.obj' % (name, key)), V)
        m2 = midline(V)
        ylo_rows = V[:, :, 1].min(0); yhi_rows = V[:, :, 1].max(0)
        info['states'][key] = dict(obj=os.path.basename(obj), t=H['t_%s' % st], th=HL.theta1_deg(H['t_%s' % st]), n_panels=int(len(S)),
                                   n_vertices_obj=nv, box_lo=lo.tolist(), box_hi=hi.tolist(), W=float(hi[0] - lo[0]), L=float(hi[1] - lo[1]),
                                   H=float(hi[2] - lo[2]), mid_x=m2[:, 0].tolist(), mid_z=m2[:, 1].tolist(),
                                   foot1=[float(m2[0, 0]), float(m2[0, 1])], foot2=[float(m2[-1, 0]), float(m2[-1, 1])],
                                   apex=[float(m2[np.argmax(m2[:, 1]), 0]), float(m2[:, 1].max())],
                                   feet_y_ranges=(float(V[:, 0, 1].min()), float(V[:, 0, 1].max()), float(V[:, -1, 1].min()), float(V[:, -1, 1].max())),
                                   ncol=int(V.shape[0] - 1), nrow=int(V.shape[1] - 1), L_full=float(yhi_rows.min() - ylo_rows.max()), extra={})
    m = midline(HL.load_V(h, 'deployed'))
    info['span'] = float(abs(m[-1, 0] - m[0, 0])); info['rise'] = float(m[:, 1].max() - m[0, 1])
    if info['truck']:
        info['deck'] = C['truck']['max_height_m'] - info['states']['tr']['H']
        info['deck_source'] = 'height limit %.2f m minus H_tr' % C['truck']['max_height_m']
    info['joint_m'] = JOINT
    ths = []
    for st in KEY:
        for q in HL.load_panels(h, st):
            hq = ConvexHull(np.unique(np.round(q, 12), axis=0)); areas = {}
            for simplex, e in zip(hq.simplices, hq.equations):
                kk = tuple(np.round(e[:3], 6)); p = hq.points[simplex]
                areas[kk] = areas.get(kk, 0.0) + 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
            n = np.array(max(areas, key=areas.get)); proj = q @ n; ths.append(proj.max() - proj.min())
    info['panel_thickness_m'] = float(np.median(ths)); info['panel_thickness_range_m'] = [float(min(ths)), float(max(ths))]
    walls = {}
    if info['hw'] > 0:
        Sv = HL.load_panels(h, 'deployed')
        Sd = [polytope(s) for s in Sv]
        allp = np.concatenate(list(Sv))
        for sgn in (-1, 1):
            side = allp[sgn * allp[:, 0] > 0]
            xl = float(side[np.argmin(side[:, 2]), 0])
            xa, xb = xl - 0.04, xl + 0.04
            zmin = np.inf
            for q in Sd:
                n = len(q)
                r = linprog(q[:, 2], A_ub=np.vstack([q[:, 0], -q[:, 0]]), b_ub=[xb, -xa], A_eq=np.ones((1, n)), b_eq=[1.0],
                            bounds=[(0, None)] * n, method='highs')
                if r.status == 0: zmin = min(zmin, r.fun)
            ylo_w, yhi_w = wall_y_range(Sv, xa, xb, zmin + 0.25)
            walls[str(sgn)] = dict(x=[xa, xb], x_lowest_corner=xl, z_lowest_panel_rel_footline=float(zmin), height=float(info['hw'] + zmin - 1e-4),
                                   y=[float(ylo_w - 0.15), float(yhi_w + 0.15)], panel_y_above=[float(ylo_w), float(yhi_w)])
    info['walls'] = walls
    json.dump(info, open(os.path.join(OUT, '%s.json' % name), 'w'), indent=1)
    return info


if __name__ == '__main__':
    os.makedirs(os.path.join(OUT, 'scene_reference'), exist_ok=True)
    shutil.copyfile(os.path.join(HERE, 'constants.json'), os.path.join(OUT, 'constants.json'))
    shutil.copyfile(os.path.join(HERE, 'reference', 'props_reference.npz'), os.path.join(OUT, 'props_reference.npz'))
    for h in HL.HALL_KEYS:
        src = os.path.join(HERE, 'reference', 'scene_hall_%s_two.npz' % h)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(OUT, 'scene_reference', 'hall_%s_two.npz' % h))
        d = case(h)
        print('hall (%s): span %.4f, rise %.4f | deployed W %.3f L %.3f H %.3f (%d panels) | transport W %.3f L %.3f H %.3f; walls %s'
              % (h, d['span'], d['rise'], d['states']['dep']['W'], d['states']['dep']['L'], d['states']['dep']['H'], d['states']['dep']['n_panels'],
                 d['states']['tr']['W'], d['states']['tr']['L'], d['states']['tr']['H'],
                 {k: round(v['height'], 4) for k, v in d['walls'].items()}))
    print('written %s' % os.path.relpath(OUT, HL.ROOT) if OUT.startswith(HL.ROOT) else 'written %s' % OUT)
