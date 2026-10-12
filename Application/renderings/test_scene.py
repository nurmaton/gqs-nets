# Geometry test of one rendered scene, run on the exact meshes that Blender renders (scene.py --dump <file>.npz writes every
# mesh object in world coordinates, after all modifiers). Checks: (a) every panel's outer-skin corners against the vertices of
# the stored positions, and the panel thickness; (b) the distance between neighbouring panels, for the rendered panels and,
# as a control, for the stored panels (they must touch); (c) no penetration: every pair of panels, and every panel against
# every prop (walls, container parts, deck, person) and the ground, by a linear program (largest ball inside both bodies);
# contacts are listed separately; (d) the positions t, theta_1 of the rendered states against data/halls.json; the whole
# scene against the reference scene of the printed rendering (renderings/reference); the props against their reference.
# usage: python3 renderings/test_scene.py <dump.npz> <out.json>       (scene data in out/renderings/geo, or RENDER_GEO)
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.spatial import ConvexHull
from scipy.optimize import nnls, linprog

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import halls as HL
GEO = os.environ.get('RENDER_GEO', os.path.join(HL.OUT, 'renderings', 'geo'))
PEN_TOL = 5e-6          # m: a ball of this radius inside both bodies counts as penetration (float32 noise is below 1e-6)
CONTACT_TOL = 1e-4      # m: closer than this counts as contact


# the parts of one schematic 20-ft container as built by bl_scene.build_container (every one must be present)
CONTAINER_PARTS = {'c_cast': 8, 'c_cross': 14, 'c_doorpost': 2, 'c_endrail': 2, 'c_floorplate': 1, 'c_header': 1, 'c_post': 4,
                   'c_roof': 1, 'c_siderail': 2, 'c_toprail_x': 2, 'c_toprail_y': 2, 'c_wall_l': 1, 'c_wall_r': 1, 'c_wall_rear': 1}


TRUCK_PARTS = {'deck': 1, 'wellbeam': 2, 'wellfloor': 1, 'neck': 1, 'gooseneck': 1, 'kingpin': 1, 'rearstep': 1, 'reardeck': 1, 'tyre': 8, 'rim': 8, 'hanger': 4,
               'axle': 2, 'leg': 2, 'leg_pad': 2, 'underrun': 1, 'underrun_strut': 2, 'rearbeam': 2, 'light': 2, 'mudflap': 2}   # bl_scene.build_truck (low-bed)


def closed_check(V, F):
    """every edge used by exactly two triangles of the object, and the mesh volume equal to the sum of its components' hull volumes"""
    from collections import Counter
    E = Counter()
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            E[(min(a, b), max(a, b))] += 1
    bad = sum(1 for c in E.values() if c != 2)
    vol = abs(np.sum(np.einsum('ij,ij->i', V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])))) / 6.0
    hv = sum(ConvexHull(p).volume for p in components(V, F) if len(p) >= 4)
    return dict(open_edges=int(bad), mesh_volume=float(vol), hull_volume=float(hv), rel_diff=float(abs(vol - hv) / hv))


class MappingError(Exception):
    pass


def components(V, F):
    parent = np.arange(len(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for f in F:
        ra = find(f[0])
        for v in f[1:]:
            rb = find(v)
            if ra != rb:
                parent[rb] = ra
    used = np.unique(F)
    comps = {}
    for i in used:
        comps.setdefault(find(i), []).append(i)
    return [V[np.array(ix)] for ix in comps.values()]


class Body:
    def __init__(self, pts, name):
        self.name = name
        self.pts = np.unique(np.round(np.asarray(pts, float), 12), axis=0)
        h = ConvexHull(self.pts)
        self.hv = self.pts[h.vertices]
        eq = h.equations
        self.A = eq[:, :3]
        self.c = -eq[:, 3]
        self.lo = self.pts.min(0)
        self.hi = self.pts.max(0)


def distance(P, Q):
    """distance between the convex hulls of the point sets P and Q (min-norm point of the Minkowski difference, NNLS)"""
    D = (P[:, None, :] - Q[None, :, :]).reshape(-1, 3)
    M = 1e4
    A = np.vstack([D.T, M * np.ones(len(D))])
    b = np.r_[0.0, 0.0, 0.0, M]
    try:
        w, _ = nnls(A, b, maxiter=50 * len(D))
        ok = abs(w.sum() - 1.0) < 1e-6
    except RuntimeError:
        ok = False
    if not ok:                                     # fall back to SLSQP on the two simplices
        from scipy.optimize import minimize
        n, m = len(P), len(Q)
        f = lambda z: float(np.sum((P.T @ z[:n] - Q.T @ z[n:]) ** 2))
        cons = [dict(type='eq', fun=lambda z: z[:n].sum() - 1), dict(type='eq', fun=lambda z: z[n:].sum() - 1)]
        r = minimize(f, np.r_[np.ones(n) / n, np.ones(m) / m], bounds=[(0, 1)] * (n + m), constraints=cons, method='SLSQP',
                     options=dict(ftol=1e-16, maxiter=500))
        if not r.success or abs(r.x[:n].sum() - 1) > 1e-6 or abs(r.x[n:].sum() - 1) > 1e-6:
            return float('nan')                                  # unknown distance: reported, never treated as contact
        return float(np.sqrt(max(r.fun, 0.0)))
    return float(np.linalg.norm(D.T @ w))


def ball(B1, B2):
    """largest s such that a ball of radius s fits in both bodies (s < 0: the bodies are apart); solved in coordinates
    centred on the pair, so that the conditioning does not depend on the distance from the world origin"""
    c0 = (B1.pts.mean(0) + B2.pts.mean(0)) / 2
    A = np.vstack([np.c_[B1.A, np.ones(len(B1.A))], np.c_[B2.A, np.ones(len(B2.A))]])
    b = np.r_[B1.c - B1.A @ c0, B2.c - B2.A @ c0]
    r = linprog([0, 0, 0, -1], A_ub=A, b_ub=b, bounds=[(None, None)] * 3 + [(None, 1.0)], method='highs')
    return float(-r.fun) if r.status == 0 else np.inf      # a failed LP counts as a penetration (never silently passes)


def close_pairs(bodies_a, bodies_b, tol, same=False):
    LA = np.array([b.lo for b in bodies_a]); HA = np.array([b.hi for b in bodies_a])
    LB = np.array([b.lo for b in bodies_b]); HB = np.array([b.hi for b in bodies_b])
    ok = np.all((LA[:, None, :] <= HB[None, :, :] + tol) & (LB[None, :, :] <= HA[:, None, :] + tol), axis=2)
    if same:
        ok = np.triu(ok, 1)
    return np.argwhere(ok)


def skin_normal(pts):
    h = ConvexHull(pts)
    areas = {}
    for simplex, e in zip(h.simplices, h.equations):
        k = tuple(np.round(e[:3], 6)); p = pts[simplex]
        areas[k] = areas.get(k, 0.0) + 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
    return np.array(max(areas, key=areas.get))


def panel_object_tests(name, comps, V, label):
    """(a), (b) and in-object (c) for one panel object; V = net vertices of the state in world coordinates (ncol, nrow, 3)"""
    nc, nr = V.shape[:2]
    faces = [(i, j) for i in range(nc - 1) for j in range(nr - 1)]
    C = np.array([(V[i, j] + V[i + 1, j] + V[i + 1, j + 1] + V[i, j + 1]) / 4 for i, j in faces])
    bodies = [Body(p, '%s_%d' % (name, k)) for k, p in enumerate(comps)]
    cen = np.array([b.pts.mean(0) for b in bodies])
    from scipy.spatial import cKDTree
    d_, f_ = cKDTree(C).query(cen)
    if not (len(set(f_.tolist())) == len(bodies) == len(faces)):
        raise MappingError('%s: %d panels map to %d distinct faces of %d' % (name, len(bodies), len(set(f_.tolist())), len(faces)))
    face_of = {faces[f]: k for k, f in enumerate(f_)}
    corner_dev, skin_off, outward, thick = [], [], [], []
    for (i, j), k in face_of.items():
        q = np.array([V[i, j], V[i + 1, j], V[i + 1, j + 1], V[i, j + 1]])
        n = np.cross(q[2] - q[0], q[3] - q[1]); n /= np.linalg.norm(n)
        P = bodies[k].pts
        dist_pl = (P - q[0]) @ n
        if np.median(dist_pl) > 0:            # orient n outwards (away from the slab)
            n = -n; dist_pl = -dist_pl
        for c in q:                            # the panel vertex at each net corner, and its distance to the face plane
            m_ = np.argmin(np.linalg.norm(P - c, axis=1))
            corner_dev.append(np.linalg.norm(P[m_] - c)); skin_off.append(abs(dist_pl[m_]))
        outward.append(dist_pl.max())          # no vertex outside the outer skin
        thick.append(-dist_pl.min())
    # neighbours (sharing an edge) and diagonal neighbours (sharing a vertex)
    nb = [((i, j), (i + 1, j)) for i, j in faces if (i + 1, j) in face_of] + [((i, j), (i, j + 1)) for i, j in faces if (i, j + 1) in face_of]
    dg = [((i, j), (i + 1, j + 1)) for i, j in faces if (i + 1, j + 1) in face_of] + [((i, j + 1), (i + 1, j)) for i, j in faces if (i + 1, j) in face_of and (i, j + 1) in face_of]
    joints = np.array([distance(bodies[face_of[a]].hv, bodies[face_of[b]].hv) for a, b in nb])
    diag = np.array([distance(bodies[face_of[a]].hv, bodies[face_of[b]].hv) for a, b in dg])
    # penetration between all close pairs of panels of this object
    pairs = close_pairs(bodies, bodies, 0.10, same=True)
    nbset = {tuple(sorted((face_of[a], face_of[b]))) for a, b in nb + dg}
    s_all, s_other, d_other = [], [], []
    for a, b in pairs:
        s = ball(bodies[a], bodies[b]); s_all.append(s)
        if (min(a, b), max(a, b)) not in nbset and np.all(bodies[a].lo <= bodies[b].hi + 0.10) and np.all(bodies[b].lo <= bodies[a].hi + 0.10):
            s_other.append(s); d_other.append(distance(bodies[a].hv, bodies[b].hv))
    s_all = np.array(s_all)
    res = dict(object=name, state=label, panels=len(bodies), faces=len(faces),
               corner_deviation_max_m=float(np.max(corner_dev)), corner_deviation_mean_m=float(np.mean(corner_dev)),
               skin_offplane_max_m=float(np.max(skin_off)), outside_outer_skin_max_m=float(np.max(outward)),
               thickness_min_m=float(np.min(thick)), thickness_max_m=float(np.max(thick)),
               neighbour_pairs=len(nb), joint_min_m=float(joints.min()), joint_max_m=float(joints.max()), joint_median_m=float(np.median(joints)),
               diagonal_pairs=len(dg), diagonal_min_m=float(diag.min()) if len(diag) else None,
               close_pairs_tested=int(len(pairs)), penetrations=int(np.sum(s_all > PEN_TOL)), largest_ball_m=float(np.nanmax(s_all)),
               other_pairs_within_100mm=len(d_other), other_pairs_min_distance_m=float(min(d_other)) if d_other else None,
               nonfinite_distances=int(np.sum(~np.isfinite(joints)) + np.sum(~np.isfinite(diag)) + np.sum(~np.isfinite(np.array(d_other, float)))))
    return res, bodies, face_of


def verified_neighbour_control(case, key, nc, nr):
    """the verified slabs (no joint): neighbours touch, none penetrates (control for the test itself)"""
    S = np.load(os.path.join(GEO, '%s_%s_slabs_verified.npy' % (case, key)), allow_pickle=True)
    V = np.load(os.path.join(GEO, '%s_%s_V.npy' % (case, key)))
    faces = [(i, j) for i in range(nc - 1) for j in range(nr - 1)]
    C = np.array([(V[i, j] + V[i + 1, j] + V[i + 1, j + 1] + V[i, j + 1]) / 4 for i, j in faces])
    bodies = [Body(s, 'v%d' % k) for k, s in enumerate(S)]
    from scipy.spatial import cKDTree
    _, f_ = cKDTree(C).query(np.array([b.pts.mean(0) for b in bodies]))
    face_of = {faces[f]: k for k, f in enumerate(f_)}
    nb = [((i, j), (i + 1, j)) for i, j in faces if (i + 1, j) in face_of] + [((i, j), (i, j + 1)) for i, j in faces if (i, j + 1) in face_of]
    d = np.array([distance(bodies[face_of[a]].hv, bodies[face_of[b]].hv) for a, b in nb])
    s = np.array([ball(bodies[face_of[a]], bodies[face_of[b]]) for a, b in nb])
    return dict(neighbour_distance_max_m=float(d.max()), neighbour_largest_ball_m=float(np.nanmax(s)), neighbour_pairs=len(nb))


def main(dump, out):
    Z = np.load(dump)
    meta = json.load(open(os.path.splitext(dump)[0] + '.meta.json'))
    case = meta['case']; cj = json.load(open(os.path.join(GEO, '%s.json' % case)))
    names = sorted({k[:-3] for k in Z.files})
    panel_objs = {'deployed': 'dep', 'packed': 'tr', 'packed_alone': 'tr'}
    R = dict(case=case, four=meta['four'], dump=os.path.basename(dump), panel_objects=[], props={})
    nonfin = [k for k in Z.files if np.issubdtype(Z[k].dtype, np.number) and not np.all(np.isfinite(Z[k]))]
    def _walk(x, path=''):
        if isinstance(x, dict): return [q for k_, v_ in x.items() for q in _walk(v_, path + '.' + str(k_))]
        if isinstance(x, (list, tuple)): return [q for k_, v_ in enumerate(x) for q in _walk(v_, path + '[%d]' % k_)]
        return [path] if isinstance(x, float) and not np.isfinite(x) else []
    nonfin += ['meta' + q for q in _walk(meta)]
    # the whole scene against the checked reference of this image (every object, triangle and winding; the meta); any difference
    # is a FAIL and stops the test (a deliberately changed scene has no reference)
    if not nonfin:
        import scene_ref as SR
        refp = os.path.join(GEO, 'scene_reference', '%s_%s.npz' % (case, 'four' if meta['four'] else 'two'))
        if os.path.exists(refp):
            sp_, sw_ = SR.compare(dump, refp); R['scene_reference'] = dict(file=os.path.relpath(refp, HERE), worst_vertex_m=sw_, problems=sp_)
            if sp_:
                R['fails'] = ['scene differs from the checked reference: ' + '; '.join(sp_[:4])]; R['pass'] = False
                json.dump(R, open(out, 'w'), indent=1, default=float)
                print('%s %s: FAIL: %s' % (case, 'four' if meta['four'] else 'two', R['fails'][0][:400])); return
        else:
            R['scene_reference'] = dict(file=None, note='no reference for this scene: all other checks run; make one with scene_ref.make_reference after checking the scene')
            print('note: no scene reference for %s %s' % (case, 'four' if meta['four'] else 'two'))
    if nonfin:                                                       # stop here: nothing else can be measured reliably
        R['fails'] = ['non-finite numbers in the dump: %s' % nonfin[:8]]; R['pass'] = False
        json.dump(R, open(out, 'w'), indent=1, default=float)
        print('%s %s: FAIL: %s' % (case, 'four' if meta['four'] else 'two', R['fails'][0])); return
    all_panels = []
    for nm, key in panel_objs.items():
        if nm not in names:
            continue
        comps = components(Z[nm + '__V'], Z[nm + '__F'])
        R.setdefault('closed_meshes', {})[nm] = closed_check(Z[nm + '__V'], Z[nm + '__F'])
        V = np.load(os.path.join(GEO, '%s_%s_V.npy' % (case, key))) + np.array(meta['placements'][nm])
        try:
            res, bodies, _ = panel_object_tests(nm, comps, V, key)
        except MappingError as e:
            R['fails_early'] = R.get('fails_early', []) + [str(e)]; continue
        R['panel_objects'].append(res); all_panels += bodies
    # panels of different objects against each other
    by_obj = {}
    for b_ in all_panels: by_obj.setdefault(b_.name.rsplit('_', 1)[0], []).append(b_)
    objs = list(by_obj); cross = []
    for i1 in range(len(objs)):
        for i2 in range(i1 + 1, len(objs)):
            for a, b in close_pairs(by_obj[objs[i1]], by_obj[objs[i2]], 0.05):
                s = ball(by_obj[objs[i1]][a], by_obj[objs[i2]][b])
                if not s <= PEN_TOL: cross.append(dict(a=by_obj[objs[i1]][a].name, b=by_obj[objs[i2]][b].name, ball_m=s))
    R['cross_object_penetrations'] = cross
    # rendered panels against the verified slabs (two-sided vertex-to-hull distance), per panel object
    R['hull_vs_verified'] = {}
    for nm, key in panel_objs.items():
        if nm not in names: continue
        Sv = [np.asarray(s_, float) + np.array(meta['placements'][nm]) for s_ in np.load(os.path.join(GEO, '%s_%s_slabs_verified.npy' % (case, key)), allow_pickle=True)]
        comps = components(Z[nm + '__V'], Z[nm + '__F'])
        from scipy.spatial import cKDTree
        cv = np.array([s_.mean(0) for s_ in Sv]); _, idx = cKDTree(cv).query(np.array([p.mean(0) for p in comps]))
        worst = 0.0
        for p, k in zip(comps, idx):
            hb = Body(Sv[k], 'v'); hp = Body(p, 'r')
            d1 = max(max(0.0, float((hb.A @ x - hb.c).max())) for x in p)          # rendered vertex outside the verified slab
            d2 = max(max(0.0, float((hp.A @ x - hp.c).max())) for x in hb.pts)     # verified vertex outside the rendered panel
            worst = max(worst, d1, d2)
        R['hull_vs_verified'][nm] = worst
    # control on the verified slabs
    R['verified_slab_control'] = {}
    for key in ('dep', 'tr'):
        V = np.load(os.path.join(GEO, '%s_%s_V.npy' % (case, key)))
        R['verified_slab_control'][key] = verified_neighbour_control(case, key, V.shape[0], V.shape[1])
    # props: every other mesh object (split into convex parts), except the shadow-catcher floor
    props = []
    skipped = []
    for nm in names:
        if nm in panel_objs or nm == 'floor':
            continue
        for k, p in enumerate(components(Z[nm + '__V'], Z[nm + '__F'])):
            try:
                props.append(Body(p, '%s#%d' % (nm, k)))
            except Exception:
                skipped.append('%s#%d' % (nm, k))
    pen, contacts, mind, nanprop = [], [], {}, 0
    for a, b in close_pairs(all_panels, props, 0.05):
        s = ball(all_panels[a], props[b])
        base = props[b].name.split('#')[0].split('.')[0]
        if not s <= PEN_TOL:
            pen.append(dict(panel=all_panels[a].name, prop=props[b].name, depth_m=s))
        d = distance(all_panels[a].hv, props[b].hv) if props[b].hv.shape[0] <= 64 else np.nan
        if props[b].hv.shape[0] <= 64 and not np.isfinite(d): nanprop += 1
        if np.isfinite(d):
            mind[base] = min(mind.get(base, np.inf), d)
            if d < CONTACT_TOL:
                contacts.append(base)
    R['props_skipped'] = skipped
    # knee walls: each wall must lie under its own foot row (its y-range covers the lowest panel points above it)
    R['walls'] = {}
    for nm in [n_ for n_ in names if n_.startswith('wall')]:
        Wv = Z[nm + '__V']; wlo, whi = Wv.min(0), Wv.max(0)
        from wallrange import wall_y_range
        dep_bodies = [b_.pts for b_ in all_panels if b_.name.startswith('deployed_')]
        ylo, yhi = wall_y_range(dep_bodies, wlo[0], whi[0], whi[2] + 0.25)   # panel material above the wall, from the rendered mesh
        R['walls'][nm] = dict(wall_y=[float(wlo[1]), float(whi[1])], foot_y=[ylo, yhi],
                              covers=bool(np.isfinite(ylo) and np.isfinite(yhi) and wlo[1] <= ylo + 1e-6 and whi[1] >= yhi - 1e-6),
                              x=[float(wlo[0]), float(whi[0])], top=float(whi[2]))
    R['props'] = dict(n_convex_parts=len(props), nonfinite_distances=nanprop, penetrations=pen, min_distance_to_panels_m={k: float(v) for k, v in sorted(mind.items())},
                      contacts={k: contacts.count(k) for k in sorted(set(contacts))})
    # ground
    R['ground'] = {nm: float(Z[nm + '__V'][:, 2].min()) for nm in names if nm != 'floor'}
    R['ground_min_m'] = float(min(R['ground'].values()))
    # person: height, and no part of it inside a panel or a prop
    if 'person' in names:
        P = Z['person__V']
        pparts = [Body(p, 'person#%d' % k) for k, p in enumerate(components(P, Z['person__F'])) if len(p) >= 4]
        others = all_panels + [b_ for b_ in props if not b_.name.startswith('person')]
        ppen = []
        for a, b in close_pairs(pparts, others, 0.02):
            s = ball(pparts[a], others[b])
            if not s <= PEN_TOL: ppen.append(dict(person=pparts[a].name, other=others[b].name, ball_m=s))
        R['person'] = dict(height_m=float(P[:, 2].max() - P[:, 2].min()), feet_z_m=float(P[:, 2].min()), penetrations=ppen)
    # packed state against the container or truck limits
    if 'packed' in names:
        Pk = Z['packed__V']; lo, hi = Pk.min(0), Pk.max(0)
        R['packed_box'] = dict(W=float(hi[0] - lo[0]), L=float(hi[1] - lo[1]), H=float(hi[2] - lo[2]), z_bottom=float(lo[2]), z_top=float(hi[2]))
    # (d) states
    hk = case.split('_', 1)[1]                                       # hall_i -> i
    st = cj['states']
    R['states'] = dict(
        deployed=dict(t=st['dep']['t'], theta1=st['dep']['th'], t_data=HL.HALLS[hk]['t_deployed'], theta1_data=HL.theta1_deg(HL.HALLS[hk]['t_deployed'])),
        transport=dict(t=st['tr']['t'], theta1=st['tr']['th'], t_data=HL.HALLS[hk]['t_transport'], theta1_data=HL.theta1_deg(HL.HALLS[hk]['t_transport'])))
    for k, v in R['states'].items():
        v['t_match'] = bool(abs(v['t'] - v['t_data']) < 1e-4)
        v['theta1_match'] = bool(abs(v['theta1'] - v['theta1_data']) < 1e-3)
    # verdict
    fails = []
    for r in R['panel_objects']:
        bad = [k for k, v in r.items() if isinstance(v, float) and not np.isfinite(v)]
        if bad or r.get('nonfinite_distances', 0): fails.append('%s: non-finite results %s (%d distances)' % (r['object'], bad, r.get('nonfinite_distances', 0)))
        if r['penetrations']: fails.append('%s: %d panel penetrations' % (r['object'], r['penetrations']))
        if r['corner_deviation_max_m'] > 1e-5: fails.append('%s: panel corner %.2e m from the net vertex' % (r['object'], r['corner_deviation_max_m']))
        if r['outside_outer_skin_max_m'] > 1e-5: fails.append('%s: vertex %.2e m outside the outer skin' % (r['object'], r['outside_outer_skin_max_m']))
        if abs(r['thickness_max_m'] - 0.08) > 1e-5 or abs(r['thickness_min_m'] - 0.08) > 1e-5: fails.append('%s: thickness %.6f-%.6f m' % (r['object'], r['thickness_min_m'], r['thickness_max_m']))
        if r['joint_max_m'] > cj['joint_m'] + 1e-5: fails.append('%s: joint up to %.2e m > stated %.4f m' % (r['object'], r['joint_max_m'], cj['joint_m']))
    if pen: fails.append('%d panel/prop penetrations' % len(pen))
    # presence of the expected objects
    truck = meta.get('truck', 'deck' in names)
    need = ['deployed', 'packed', 'person', 'floor'] + (['packed_alone'] if meta['four'] else []) + (['wall+1', 'wall-1'] if meta['hw'] > 0 else [])
    need += ['deck'] if truck else ['c_wall_l', 'c_wall_r', 'c_wall_rear', 'c_roof', 'c_header', 'c_doorpost', 'c_floorplate']
    missing = [n_ for n_ in need if n_ not in names]
    if missing: fails.append('missing objects: %s' % missing)
    # support: every panel object rests on its support (contact within 1 mm), no floating
    md = R['props']['min_distance_to_panels_m']
    badmd = [k for k, v in md.items() if not np.isfinite(v)] + (['%d panel-prop distances' % R['props']['nonfinite_distances']] if R['props']['nonfinite_distances'] else [])
    if badmd: fails.append('prop distances not computed (non-finite): %s' % badmd)
    sup = {}
    if meta['hw'] > 0:
        sup['deployed on the walls'] = min(md.get('wall+1', np.inf), md.get('wall-1', np.inf))
        sup['each wall under the deployed hall'] = max(md.get('wall+1', np.inf), md.get('wall-1', np.inf))
    else:
        sup['deployed on the ground'] = R['ground'].get('deployed', np.inf)
    floor_key = 'wellfloor' if 'wellfloor' in md else ('deck' if 'deck' in md else 'c_floorplate')
    sup['packed on its floor or deck'] = md.get(floor_key, np.inf)
    if meta['four']: sup['packed_alone on the ground'] = R['ground'].get('packed_alone', np.inf)
    sup['person on the ground'] = R.get('person', {}).get('feet_z_m', np.inf)
    R['support_gaps_m'] = {k: float(v) for k, v in sup.items()}
    for k, v in sup.items():
        if not (abs(v) <= 1e-3): fails.append('support: %s gap %.4f m' % (k, v))
    # scene against the labelled constants and the stated design
    C = json.load(open(os.path.join(GEO, 'constants.json'))); ext = {}
    if 'floor' in names:
        ext['ground plane z'] = float(Z['floor__V'][:, 2].mean())
        if np.abs(Z['floor__V'][:, 2] + 1e-4).max() > 1e-6: fails.append('ground plane not flat at z = -1e-4 m (max dev %.2e)' % np.abs(Z['floor__V'][:, 2] + 1e-4).max())
    ext['deployed foot line z'] = meta['zf']
    kw = C['knee_walls_footline_m'].get(case)
    if kw is not None and abs(meta['zf'] - kw) > 1e-6: fails.append('deployed foot line %.4f m, design value %.2f m' % (meta['zf'], kw))
    if kw is None and meta['hw'] > 0: fails.append('no foot-line height in constants.json for %s' % case)
    # the foot line measured on the rendered mesh: the mean height of the foot-row vertices (zigzag levels average to the foot line)
    if 'deployed' in names:
        pz = meta['placements']['deployed'][2]
        if kw is not None and abs(pz - kw) > 1e-6: fails.append('deployed placement z %.4f m, design value %.2f m' % (pz, kw))
        # foot line from the rendered mesh: the nearest dumped vertex to every foot-row net vertex, averaged
        Vn = np.load(os.path.join(GEO, '%s_dep_V.npy' % case)) + np.array(meta['placements']['deployed'])
        from scipy.spatial import cKDTree
        tree = cKDTree(Z['deployed__V'])
        zrows, dmax = [], 0.0
        for j_ in (0, -1):                                            # foot line = mean of the two zigzag levels of a foot row
            dd, ii = tree.query(Vn[:, j_]); dmax = max(dmax, dd.max()); zz = Z['deployed__V'][ii, 2]
            zrows.append(0.5 * (zz[0::2].mean() + zz[1::2].mean()))
        dd = np.array([dmax]); zfm = float(np.mean(zrows)); ext['foot line from the mesh'] = zfm
        if dd.max() > 1e-5: fails.append('foot-row vertices not found on the mesh (%.2e m)' % dd.max())
        expected = kw if kw is not None else meta['zf']
        if abs(zfm - expected) > 1e-4: fails.append('foot line on the mesh %.4f m, expected %.4f m' % (zfm, expected))
    Pk = Z['packed__V'] if 'packed' in names else None
    xpk = 0.5 * (Pk[:, 0].min() + Pk[:, 0].max()) if Pk is not None else 0.0
    def carrier(base):
        """the vertices of the parts named base* that belong to the carrier holding the packed net (x within 2 m of its centre)"""
        parts = [Z[n_ + '__V'] for n_ in names if n_.split('.')[0] == base and abs(Z[n_ + '__V'][:, 0].mean() - xpk) < 2.0]
        return np.concatenate(parts) if parts else None
    if not truck and Pk is not None and any(carrier(b_) is None for b_ in ('c_doorpost', 'c_header', 'c_floorplate', 'c_wall_rear')):
        fails.append('container parts missing at the packed net: %s' % [b_ for b_ in ('c_doorpost', 'c_header', 'c_floorplate', 'c_wall_rear') if carrier(b_) is None])
    elif not truck and Pk is not None:
        PV = carrier('c_doorpost'); xc_ = PV[:, 0].mean()
        inner = [PV[PV[:, 0] < xc_, 0].max(), PV[PV[:, 0] > xc_, 0].min()]
        hdr = carrier('c_header'); fl = carrier('c_floorplate'); rear = carrier('c_wall_rear')
        door_w, door_h = inner[1] - inner[0], hdr[:, 2].min() - fl[:, 2].max()
        ext['door opening'] = [float(door_w), float(door_h)]
        dW, dH = C['container_20ft']['door_min_W_H_m']
        if abs(door_w - dW) > 1e-3 or abs(door_h - dH) > 1e-3: fails.append('door opening %.4f x %.4f m, label %.3f x %.3f m' % (door_w, door_h, dW, dH))
        # the packed net inside the container: within the door width and height, the internal length and above the floor plate
        IL, IW, IH = C['container_20ft']['internal_min_L_W_H_m']
        lo, hi = Pk.min(0), Pk.max(0)
        y_door = PV[:, 1].max(); y_rear = rear[:, 1].min()             # inner face of the door frame and of the rear wall, from the scene
        ext['container interior length (door frame to rear wall)'] = float(y_rear - y_door)
        if not (abs(y_rear - y_door - IL) <= 1e-4): fails.append('container interior length %.4f m (ISO min %.3f m)' % (y_rear - y_door, IL))
        inside = (lo[0] >= inner[0] - 1e-6 and hi[0] <= inner[1] + 1e-6 and lo[2] >= fl[:, 2].max() - 1e-6 and hi[2] <= hdr[:, 2].min() + 1e-6
                  and lo[1] >= y_door - 1e-6 and hi[1] <= y_rear + 1e-6 and lo[1] >= fl[:, 1].min() - 1e-6 and hi[1] <= fl[:, 1].max() + 1e-6)
        ext['packed inside the container'] = bool(inside)
        if not inside: fails.append('packed net not inside the container door/interior')
    if truck and Pk is not None and carrier('deck') is not None:
        dk = carrier('deck'); lo, hi = Pk.min(0), Pk.max(0)
        dw_ = dk[:, 0].max() - dk[:, 0].min(); ext['deck width'] = float(dw_); ext['top of the load'] = float(hi[2])
        if abs(dw_ - C['truck']['vehicle']['width_m']) > 1e-3 or dw_ > C['truck']['max_width_m'] + 1e-6:
            fails.append('well width %.4f m, vehicle %.3f m, limit %.2f m' % (dw_, C['truck']['vehicle']['width_m'], C['truck']['max_width_m']))
        if hi[2] > C['truck']['max_height_m'] + 1e-6: fails.append('load above the height limit: %.4f m' % hi[2])
        if lo[0] < dk[:, 0].min() - 1e-6 or hi[0] > dk[:, 0].max() + 1e-6: fails.append('packed net wider than the deck')
    from wallrange import z_min_over
    dep_pts = [b_.pts for b_ in all_panels if b_.name.startswith('deployed_')]
    for nm, w in R['walls'].items():
        zl = z_min_over(dep_pts, w['x'][0] - 0.10, w['x'][1] + 0.10); ext['%s lowest panel within 0.1 m' % nm] = float(zl)
        if zl < w['top'] - 1e-4: fails.append('%s: panel material %.4f m below its top next to it' % (nm, w['top'] - zl))
        th = w['x'][1] - w['x'][0]; ext['%s thickness' % nm] = th
        if abs(th - 0.08) > 1e-4: fails.append('%s thickness %.4f m' % (nm, th))
        if np.isfinite(w['foot_y'][0]) and (abs(w['wall_y'][0] - (w['foot_y'][0] - 0.15)) > 1e-3 or abs(w['wall_y'][1] - (w['foot_y'][1] + 0.15)) > 1e-3):
            fails.append('%s length does not match the panel material above it (+-0.15 m)' % nm)
    # props against the checked reference layout: every unit (container or trailer) has exactly the reference parts, of the
    # same shape and size at the same place; 1 unit in a two-object scene, 2 in a four-object scene; nothing else in the scene
    import props_ref as PR
    ref = PR.load_reference(os.path.join(GEO, 'props_reference.npz'))
    kind, other = ('truck', 'container') if truck else ('container', 'truck')
    U = PR.units(Z, names, PR.TRUCK_BASES if truck else PR.CONTAINER_BASES)
    n_exp = 2 if meta['four'] else 1
    if len(U) != n_exp: fails.append('%d %s units in the scene, %d expected' % (len(U), kind, n_exp))
    if PR.units(Z, names, PR.CONTAINER_BASES if truck else PR.TRUCK_BASES): fails.append('%s parts in a %s scene' % (other, kind))
    anchors = []
    for g in U:
        pr, a_ = PR.compare_unit(Z, g, ref[kind]); anchors.append(a_)
        if pr: fails.append('%s at x %.1f differs from the reference layout: %s%s' % (kind, a_[0], '; '.join(pr[:4]), ' ...' if len(pr) > 4 else ''))
    ext['%s anchors (x, y, z)' % kind] = [[float(v) for v in a_] for a_ in anchors]
    zmins = [min(Z[n_ + '__V'][:, 2].min() for n_ in g) for g in U]
    if anchors and (np.ptp([a_[1] for a_ in anchors]) > 1e-6 or np.ptp(zmins) > 1e-6):
        fails.append('%s units not at the same depth and height' % kind)
    if not truck and anchors:
        CLn, ILn = C['container_20ft']['external_L_W_H_m'][0], C['container_20ft']['internal_min_L_W_H_m'][0]
        if any(abs(a_[1] + (CLn - ILn) / 2) > 1e-5 for a_ in anchors): fails.append('container not placed with its interior from y = 0')
    allowed = set(n_ for g in U for n_ in g) | {'person', 'floor', 'deployed', 'packed'} | ({'packed_alone'} if meta['four'] else set()) \
        | ({'wall+1', 'wall-1'} if meta['hw'] > 0 else set())
    extra = [n_ for n_ in names if n_ not in allowed]
    if extra: fails.append('objects not expected in the scene: %s' % extra[:6])
    # clear volumes against ISO 668 (independent of the reference): no part reaches into the door opening extended through the
    # whole container, and the internal height and width are the ISO minima
    if not truck:
        IL_, IW_, IH_ = C['container_20ft']['internal_min_L_W_H_m']; DW_, DH_ = C['container_20ft']['door_min_W_H_m']
        for g, a_ in zip(U, anchors):
            fpn = [n_ for n_ in g if PR.base(n_) == 'c_floorplate']
            if not fpn: continue
            ft_ = Z[fpn[0] + '__V'][:, 2].max(); xc0 = a_[0]
            box_lo = np.array([xc0 - DW_ / 2, a_[1], ft_]); box_hi = np.array([xc0 + DW_ / 2, IL_, ft_ + DH_])
            for n_ in g:
                v = Z[n_ + '__V']; ov = np.minimum(v.max(0), box_hi) - np.maximum(v.min(0), box_lo)
                if np.all(ov > 1e-5): fails.append('container at x %.1f: %s reaches into the clear door opening (%.4f x %.4f x %.4f m)' % (xc0, n_, ov[0], ov[1], ov[2]))
            rf = [n_ for n_ in g if PR.base(n_) == 'c_roof']; wl = [n_ for n_ in g if PR.base(n_) == 'c_wall_l']; wr = [n_ for n_ in g if PR.base(n_) == 'c_wall_r']
            if rf and wl and wr:
                ih = Z[rf[0] + '__V'][:, 2].min() - ft_; iw = Z[wr[0] + '__V'][:, 0].min() - Z[wl[0] + '__V'][:, 0].max()
                ext['internal height, width at x %.1f' % xc0] = [float(ih), float(iw)]
                if abs(ih - IH_) > 1e-4 or abs(iw - IW_) > 1e-4: fails.append('container at x %.1f: internal height x width %.4f x %.4f m, ISO %.3f x %.3f' % (xc0, ih, iw, IH_, IW_))
    if not truck:
        # one group per container, anchored on its floor plate; its door posts are the two within 1.5 m of the plate's centre
        groups = []
        for fpn in [n_ for n_ in names if n_.split('.')[0] == 'c_floorplate']:
            xm = float(Z[fpn + '__V'][:, 0].mean())
            groups.append(dict(x=xm, n=[n_ for n_ in names if n_.split('.')[0] == 'c_doorpost' and abs(Z[n_ + '__V'][:, 0].mean() - xm) < 1.5]))
        dW, dH = C['container_20ft']['door_min_W_H_m']
        n_exp = 2 if meta['four'] else 1
        if len(groups) != n_exp: fails.append('%d containers in the scene, %d expected' % (len(groups), n_exp))
        for gr in groups:
          try:
            if len(gr['n']) != 2: fails.append('container at x %.1f: %d door posts' % (gr['x'], len(gr['n']))); continue
            PVg = np.concatenate([Z[n_ + '__V'] for n_ in gr['n']]); xc_ = gr['x']
            near = lambda base: [Z[n_ + '__V'] for n_ in names if n_.split('.')[0] == base and abs(Z[n_ + '__V'][:, 0].mean() - xc_) < 2.0]
            hd = near('c_header'); fp = near('c_floorplate'); cp = near('c_post')
            cnt = {b_: len(near(b_)) for b_ in CONTAINER_PARTS}
            ext['parts of the container at x %.1f' % xc_] = cnt
            wrong = {b_: (cnt[b_], k_) for b_, k_ in CONTAINER_PARTS.items() if cnt[b_] != k_}
            if wrong: fails.append('container at x %.1f: part counts (found, expected) %s' % (xc_, wrong))
            allp = [v for b_ in CONTAINER_PARTS for v in near(b_)]
            if allp:
                AV = np.concatenate(allp); ext_ = AV.max(0) - AV.min(0)
                ext['external size at x %.1f (W, L, H)' % xc_] = [float(ext_[0]), float(ext_[1]), float(ext_[2])]
                eL, eW, eH = C['container_20ft']['external_L_W_H_m']
                if abs(ext_[0] - eW) > 1e-4 or abs(ext_[1] - eL) > 1e-4 or abs(ext_[2] - eH) > 1e-4:
                    fails.append('container at x %.1f: external size %.4f x %.4f x %.4f m, nominal %.3f x %.3f x %.3f' % (xc_, ext_[0], ext_[1], ext_[2], eW, eL, eH))
                cs_ = near('c_cast')
                if cs_:
                    CS = np.concatenate(cs_)
                    if np.abs(CS.min(0) - AV.min(0)).max() > 1e-6 or np.abs(CS.max(0) - AV.max(0)).max() > 1e-6:
                        fails.append('container at x %.1f: a part reaches outside the corner castings' % xc_)
                for b_ in ('c_post', 'c_siderail', 'c_endrail'):
                    for v in near(b_):
                        if abs(v[:, 2].min()) > 1e-3: fails.append('container at x %.1f: a %s is not on the ground (bottom %.4f m)' % (xc_, b_, v[:, 2].min()))
                nb_low = sum(1 for v in cs_ if abs(v[:, 2].min()) <= 1e-6)
                nb_top = sum(1 for v in cs_ if abs(v[:, 2].max() - AV[:, 2].max()) <= 1e-6)
                if nb_low != 4 or nb_top != 4: fails.append('container at x %.1f: %d castings on the ground, %d at the top (4 and 4 expected)' % (xc_, nb_low, nb_top))
                for b_ in ('c_cross', 'c_toprail_x', 'c_toprail_y', 'c_post', 'c_siderail', 'c_endrail', 'c_doorpost'):
                    zz = np.array([(v[:, 2].min(), v[:, 2].max()) for v in near(b_)]).reshape(-1, 2)
                    if len(zz) > 1 and np.ptp(zz, axis=0).max() > 1e-5: fails.append('container at x %.1f: the %s parts differ in height by %.2e m' % (xc_, b_, np.ptp(zz, axis=0).max()))
                # every part but the castings at least 0.4 mm inside the nominal external size (bottom parts may reach z = 0)
                elo, ehi = AV.min(0), AV.max(0)
                for b_ in CONTAINER_PARTS:
                    if b_ == 'c_cast': continue
                    for v in near(b_):
                        m_lo = v.min(0) - elo; m_hi = ehi - v.max(0)
                        if min(m_lo[0], m_lo[1], m_hi[0], m_hi[1], m_hi[2]) < 4e-4 or m_lo[2] < -1e-6:
                            fails.append('container at x %.1f: a %s is not inside the castings (margins %.2e m)' % (xc_, b_, min(m_lo[0], m_lo[1], m_hi[0], m_hi[1], m_hi[2])))
                # one casting at each of the 8 corners of the external size, all of the same size
                corners = [(xx, yy, zz) for xx in (elo[0], ehi[0]) for yy in (elo[1], ehi[1]) for zz in (elo[2], ehi[2])]
                at = [sum(1 for v in cs_ if np.all(np.min(np.abs(v - np.array(cn)), axis=0) <= 1e-6) and np.all((v.min(0) <= np.array(cn) + 1e-6) & (v.max(0) >= np.array(cn) - 1e-6))) for cn in corners]
                if at != [1] * 8: fails.append('container at x %.1f: castings at the 8 corners %s (one each expected)' % (xc_, at))
                sz = np.array([v.max(0) - v.min(0) for v in cs_])
                if len(sz) and np.ptp(sz, axis=0).max() > 1e-5: fails.append('container at x %.1f: castings of different sizes' % xc_)
                # coincident faces of different parts (z-fighting, alpha-layer flicker): none may be visible
                from coplanar_faces import coincident, is_box
                gparts = [(n_, Z[n_ + '__V']) for n_ in names if n_.split('.')[0] in CONTAINER_PARTS and abs(Z[n_ + '__V'][:, 0].mean() - xc_) < 2.0]
                nb_ = [n_ for n_, v in gparts if not is_box(v)]
                if nb_: fails.append('container at x %.1f: parts that are not boxes %s' % (xc_, nb_))
                co = coincident(gparts); vis = [c_ for c_ in co if c_['visible']]
                ext['coincident faces at x %.1f (all, harmless, hidden, visible)' % xc_] = [len(co), sum(1 for c_ in co if c_['harmless']), sum(1 for c_ in co if c_['hidden'] and not c_['harmless']), len(vis)]
                if vis: fails.append('container at x %.1f: %d visible coincident faces, e.g. %s/%s %s' % (xc_, len(vis), vis[0]['a'], vis[0]['b'], vis[0]['facing']))
            rr = near('c_wall_rear')
            if rr:
                HDg = np.concatenate(hd) if hd else PVg
                if abs(PVg[:, 1].max() - HDg[:, 1].max()) > 1e-6: fails.append('container at x %.1f: door posts and header end at different depths (%.4f, %.4f)' % (xc_, PVg[:, 1].max(), HDg[:, 1].max()))
                y_door_g = max(PVg[:, 1].max(), HDg[:, 1].max()); y_rear_g = np.concatenate(rr)[:, 1].min()
                IL_ = C['container_20ft']['internal_min_L_W_H_m'][0]
                ext['interior length at x %.1f (door frame to rear wall)' % xc_] = float(y_rear_g - y_door_g)
                if not (abs(y_rear_g - y_door_g - IL_) <= 1e-4):
                    fails.append('container at x %.1f: interior length %.4f m (ISO min %.3f m)' % (xc_, y_rear_g - y_door_g, IL_))
            if hd and fp:
                inner = [PVg[PVg[:, 0] < xc_, 0].max(), PVg[PVg[:, 0] > xc_, 0].min()]
                if cp:                                                     # the clear width is limited by the corner posts too
                    CP = np.concatenate(cp); inner = [max(inner[0], CP[CP[:, 0] < xc_, 0].max()), min(inner[1], CP[CP[:, 0] > xc_, 0].min())]
                dw_, dh_ = inner[1] - inner[0], np.concatenate(hd)[:, 2].min() - np.concatenate(fp)[:, 2].max()
                ext['door opening at x %.1f' % xc_] = [float(dw_), float(dh_)]
                if abs(dw_ - dW) > 1e-3 or abs(dh_ - dH) > 1e-3: fails.append('container at x %.1f: clear door %.4f x %.4f m, label %.3f x %.3f' % (xc_, dw_, dh_, dW, dH))
            low = min(v[:, 2].min() for base in ('c_post', 'c_cast', 'c_siderail', 'c_endrail') for v in near(base)) if cp else np.inf
            ext['container at x %.1f bottom z' % xc_] = float(low)
            if abs(low) > 1e-3: fails.append('container at x %.1f not on the ground (bottom %.4f m)' % (xc_, low))
          except Exception as e_:                                     # a crash is a FAIL, never an unreported exception
            fails.append('container at x %.1f: check crashed (%s: %s)' % (gr['x'], type(e_).__name__, e_))
    else:
        tyres = [Z[n_ + '__V'] for n_ in names if n_.split('.')[0] == 'tyre']
        if not tyres: fails.append('truck tyres missing')
        else:
            tz = max(abs(v[:, 2].min()) for v in tyres); ext['tyres bottom z (largest)'] = float(tz)
            if tz > 1e-3: fails.append('truck not on the ground (a tyre bottom at %.4f m)' % tz)
        # one group per trailer, anchored on its deck: every part present, each tyre and leg pad on the ground, same deck height
        dks = [n_ for n_ in names if n_.split('.')[0] == 'deck']
        n_exp = 2 if meta['four'] else 1
        if len(dks) != n_exp: fails.append('%d trailers in the scene, %d expected' % (len(dks), n_exp))
        for dkn in dks:
            try:
                xd = float(Z[dkn + '__V'][:, 0].mean())
                nearT = lambda base: [Z[n_ + '__V'] for n_ in names if n_.split('.')[0] == base and abs(Z[n_ + '__V'][:, 0].mean() - xd) < 2.0]
                cnt = {b_: len(nearT(b_)) for b_ in TRUCK_PARTS}; ext['parts of the trailer at x %.1f' % xd] = cnt
                wrong = {b_: (cnt[b_], k_) for b_, k_ in TRUCK_PARTS.items() if cnt[b_] != k_}
                if wrong: fails.append('trailer at x %.1f: part counts (found, expected) %s' % (xd, wrong))
                for b_ in ('tyre', 'leg_pad'):
                    for v in nearT(b_):
                        if abs(v[:, 2].min()) > 1e-3: fails.append('trailer at x %.1f: a %s is not on the ground (bottom %.4f m)' % (xd, b_, v[:, 2].min()))
                wfl = nearT('wellfloor'); dtop = float(wfl[0][:, 2].max()) if wfl else float(Z[dkn + '__V'][:, 2].max()); ext['well top at x %.1f' % xd] = dtop
                for wb_ in nearT('wellbeam'):
                    if abs(float(wb_[:, 2].max()) - dtop) > 1e-4: fails.append('trailer at x %.1f: side frame top %.4f m, floor %.4f m' % (xd, wb_[:, 2].max(), dtop))
                Vh_ = C['truck']['vehicle']
                if abs(dtop - Vh_['well_top_loaded_m']) > 1e-3: fails.append('trailer at x %.1f: well top %.4f m, maker %.3f m' % (xd, dtop, Vh_['well_top_loaded_m']))
                dbot = float(Z[dkn + '__V'][:, 2].min()); ext['well underside at x %.1f' % xd] = dbot
                if abs(dbot - Vh_['ground_clearance_m']) > 1e-3: fails.append('trailer at x %.1f: well underside %.4f m, maker clearance %.3f m' % (xd, dbot, Vh_['ground_clearance_m']))
                # sourced dimensions of the extended trailer (maker page; tyre and axle data of the same class)
                allT = np.concatenate([v for b_ in TRUCK_PARTS for v in nearT(b_)])
                Lx = allT[:, 1].max() - allT[:, 1].min(); ext['trailer length at x %.1f' % xd] = float(Lx)
                if abs(Lx - (Vh_['length_closed_m'] + Vh_['extension_m'])) > 1e-3: fails.append('trailer at x %.1f: length %.4f m, maker %.3f + %.3f m' % (xd, Lx, Vh_['length_closed_m'], Vh_['extension_m']))
                wl = Z[dkn + '__V'][:, 1]; ext['well length at x %.1f' % xd] = float(np.ptp(wl))
                if abs(np.ptp(wl) - (Vh_['well_length_closed_m'] + Vh_['extension_m'])) > 1e-3: fails.append('trailer at x %.1f: well length %.4f m' % (xd, np.ptp(wl)))
                # further sourced sizes: gooseneck length and coupling height, deck over the axles, tyre width, twin spacing, width over the tyres
                gn = nearT('gooseneck'); rd = nearT('reardeck'); Om_ = C['truck']['other_models']
                if gn:
                    gl_ = float(gn[0][:, 1].max() - Z[dkn + '__V'][:, 1].max()); h5_ = float(gn[0][:, 2].min())
                    ext['gooseneck length, coupling height at x %.1f' % xd] = [gl_, h5_]
                    if abs(gl_ - Vh_['gooseneck_length_m']) > 1e-3 or abs(h5_ - Vh_['coupling_height_m']) > 1e-3: fails.append('trailer at x %.1f: gooseneck %.4f m long, underside %.4f m' % (xd, gl_, h5_))
                if rd and abs(float(rd[0][:, 2].max()) - Vh_['bogie_deck_height_m']) > 1e-3: fails.append('trailer at x %.1f: deck over the axles %.4f m' % (xd, rd[0][:, 2].max()))
                tyx = sorted(float(0.5 * (v[:, 0].min() + v[:, 0].max())) - xd for v in nearT('tyre'))
                tyw = [float(np.ptp(v[:, 0])) for v in nearT('tyre')]
                if tyx:
                    wot = (max(v[:, 0].max() for v in nearT('tyre')) - min(v[:, 0].min() for v in nearT('tyre')))
                    ext['width over the tyres at x %.1f' % xd] = float(wot)
                    if abs(wot - Om_['width_over_tyres_m']) > 1e-3: fails.append('trailer at x %.1f: width over the tyres %.4f m' % (xd, wot))
                    if max(abs(w_ - C['truck']['tyre']['section_width_m']) for w_ in tyw) > 1e-3: fails.append('trailer at x %.1f: tyre widths %s' % (xd, sorted(set(round(w_, 4) for w_ in tyw))))
                    pos = []
                    for x_ in sorted(abs(x_) for x_ in tyx):                    # tyre centres across the trailer, grouped within 1 mm
                        if not pos or x_ - pos[-1][-1] > 1e-3: pos.append([x_])
                        else: pos[-1].append(x_)
                    pos = [float(np.mean(g_)) for g_ in pos]
                    if len(pos) != 2 or abs(pos[1] - pos[0] - C['truck']['tyre']['twin_spacing_m']) > 1e-3: fails.append('trailer at x %.1f: twin tyre centres %s' % (xd, np.round(pos, 4).tolist()))
                tys = nearT('tyre'); Ty_ = C['truck']['tyre']
                dia = [float(v[:, 2].max() - v[:, 2].min()) for v in tys]
                if max(abs(d_ - Ty_['diameter_m']) for d_ in dia) > 2e-3: fails.append('trailer at x %.1f: tyre diameters %s, maker %.3f m' % (xd, sorted(set(round(d_, 4) for d_ in dia)), Ty_['diameter_m']))
                yax = sorted(set(round(float(0.5 * (v[:, 1].min() + v[:, 1].max())), 4) for v in tys))
                kpv = nearT('kingpin'); ext['axle y at x %.1f' % xd] = yax
                if len(yax) != Vh_['axles']: fails.append('trailer at x %.1f: %d axle positions, maker %d axles' % (xd, len(yax), Vh_['axles']))
                elif kpv:
                    yk_ = float(kpv[0][:, 1].mean()); wbx = yk_ - 0.5 * (yax[0] + yax[-1]); spx = yax[-1] - yax[0]
                    ext['king pin to axle centre, axle spacing at x %.1f' % xd] = [wbx, spx]
                    if abs(wbx - (Vh_['wheelbase_closed_m'] + Vh_['extension_m'])) > 1e-3 or abs(spx - C['truck']['other_models']['axle_spacing_m']) > 1e-3:
                        fails.append('trailer at x %.1f: king pin to axles %.4f m, axle spacing %.4f m' % (xd, wbx, spx))
                if 'packed' in names and abs(Z['packed__V'][:, 0].mean() - xd) < 2.0:     # the loaded trailer: load in the well, centred
                    Pq = Z['packed__V']; ylo_, yhi_ = Pq[:, 1].min(), Pq[:, 1].max()
                    if ylo_ < wl.min() - 1e-6 or yhi_ > wl.max() + 1e-6: fails.append('trailer at x %.1f: load outside the well along the trailer' % xd)
                    if abs((ylo_ + yhi_) / 2 - (wl.min() + wl.max()) / 2) > 1e-3: fails.append('trailer at x %.1f: load not centred in the well' % xd)
                    xw_ = Z[dkn + '__V'][:, 0]
                    if abs((Pq[:, 0].min() + Pq[:, 0].max()) / 2 - (xw_.min() + xw_.max()) / 2) > 1e-3: fails.append('trailer at x %.1f: load not centred across the well' % xd)
                dwx = float(np.ptp(Z[dkn + '__V'][:, 0]))
                dx0, dx1 = Z[dkn + '__V'][:, 0].min(), Z[dkn + '__V'][:, 0].max()
                outside = [b_ for b_ in TRUCK_PARTS for v in nearT(b_) if v[:, 0].min() < dx0 - 1e-5 or v[:, 0].max() > dx1 + 1e-5]
                if outside: fails.append('trailer at x %.1f: parts wider than the deck (%.2f m): %s' % (xd, dwx, sorted(set(outside))))
                if abs(dwx - C['truck']['vehicle']['width_m']) > 1e-3: fails.append('trailer at x %.1f: well width %.4f m' % (xd, dwx))
            except Exception as e_:
                fails.append('trailer check crashed (%s: %s)' % (type(e_).__name__, e_))
    for nm in [n_ for n_ in names if n_.startswith('wall')]:
        wz = float(Z[nm + '__V'][:, 2].min()); ext['%s bottom z' % nm] = wz
        if abs(wz) > 1e-3: fails.append('%s not on the ground (bottom %.4f m)' % (nm, wz))
    if 'person' in names:
        Pp = Z['person__V']; wd_, dp_ = Pp[:, 0].max() - Pp[:, 0].min(), Pp[:, 1].max() - Pp[:, 1].min()
        ext['person width, depth'] = [float(wd_), float(dp_)]
        if not (0.5 < wd_ < 0.75 and 0.2 < dp_ < 0.45): fails.append('person proportions %.2f x %.2f m' % (wd_, dp_))
    R['scene_vs_design'] = ext
    fails += R.get('fails_early', [])
    for nm, cc in R.get('closed_meshes', {}).items():
        if cc['open_edges'] or cc['rel_diff'] > 1e-4: fails.append('%s: mesh not closed (%d open edges, volume diff %.2e)' % (nm, cc['open_edges'], cc['rel_diff']))
    if skipped: fails.append('%d props could not be tested: %s' % (len(skipped), skipped[:5]))
    for nm, w in R['walls'].items():
        if not w['covers']: fails.append('%s does not lie under its own foot row: wall y %s, foot y %s' % (nm, np.round(w['wall_y'], 3), np.round(w['foot_y'], 3)))
    if R['ground_min_m'] < -1e-6: fails.append('object below the ground: %.4f m' % R['ground_min_m'])
    if 'person' in R and abs(R['person']['height_m'] - 1.80) > 1e-3: fails.append('person height %.4f m' % R['person']['height_m'])
    if 'person' in R and R['person']['penetrations']: fails.append('person passes through %d bodies' % len(R['person']['penetrations']))
    if R['cross_object_penetrations']: fails.append('%d penetrations between panel objects' % len(R['cross_object_penetrations']))
    for nm, w in R['hull_vs_verified'].items():
        if w > 1e-5: fails.append('%s: rendered panels differ from the verified slabs by %.2e m' % (nm, w))
    for k, v in R['verified_slab_control'].items():
        if not v['neighbour_largest_ball_m'] <= PEN_TOL: fails.append('verified slabs (%s) penetrate: %.2e m' % (k, v['neighbour_largest_ball_m']))
        if not v['neighbour_distance_max_m'] <= 1e-5: fails.append('verified slabs (%s): neighbours do not touch (%.2e m)' % (k, v['neighbour_distance_max_m']))
    for r in R['panel_objects']:
        if not np.isfinite(r['largest_ball_m']): fails.append('%s: LP failure' % r['object'])
    for k, v in R['states'].items():
        if not (v['t_match'] and v['theta1_match']): fails.append('state %s does not match data/halls.json' % k)
    R['fails'] = fails; R['pass'] = not fails
    json.dump(R, open(out, 'w'), indent=1, default=float)
    print('%s %s: %s' % (case, 'four' if meta['four'] else 'two', 'PASS' if R['pass'] else 'FAIL: ' + '; '.join(fails)))


if __name__ == '__main__':
    try:
        main(sys.argv[1], sys.argv[2])
    except Exception as e:                                           # any crash is reported as FAIL on stdout and in the json
        import traceback; traceback.print_exc()
        json.dump(dict(dump=sys.argv[1], fails=['test crashed: %s: %s' % (type(e).__name__, ' '.join(str(e).split()))], **{'pass': False}), open(sys.argv[2], 'w'), indent=1)
        print('%s: FAIL: test crashed (%s: %s)' % (os.path.basename(sys.argv[1]), type(e).__name__, ' '.join(str(e).split())))
