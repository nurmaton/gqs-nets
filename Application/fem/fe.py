# Snow check of the deployed halls with CalculiX: models, loads, input decks, static runs with capacities, buckling runs
# with the classification of the modes.
# Models (shellnet.py): the deployed outer skin (fem/input/hall_*_fe_V.npy, metres), k x k S8R elements per face, the 80 mm
# sandwich as an equivalent shell with core shear, hinges on the outer face sheet (offset), lower foot vertices supported.
# Variants: V1 both feet anchored; V3 foot 1 anchored, foot 2 on rails (free across the hall), one spindle between the
# feet and one point latch on a hinge. Loads: self-weight G and snow S1 (undrifted), S2+ and S2- (drifted, on either side)
# at s_k = 1 kN/m2; SLS = G + S, ULS = 1.35 G + 1.5 S.
#   static: mechanism gate, the four load cases, displacements, stresses, capacities s_k,max per limit (SLS vertical and
#           total displacement span/200, face von Mises 280/1.1 MPa, face compression 71.6 MPa, core shear 0.10 MPa),
#           stresses read outside a radius r of every net vertex (and, for V3 of (i), outside 0.15 m of the latch); the
#           place where each limit is reached first (nearest net vertex and whether it is a support or a spindle end,
#           distance to the nearest hinge line, position across the span); the output also keeps the load combinations and
#           the capacities by radius in the form of the stored results (compare_static.py compares the two)
#   buckle: *BUCKLE under ULS (scaled by --scale; factors are given for the unscaled load), every mode classified by rule A
#           (classify.py) and rule B (broad if at least 8 face centres move 30 % of the peak or more); lambda_1 = the
#           lowest broad mode; local modes at supports, panel corners and the spindle end are not counted
#   decks:  the input decks of a case without running ccx (see make_decks.py)
# usage: python3 fem/fe.py static HALL V K [--threads N] [--work DIR]
#        python3 fem/fe.py buckle HALL V K CASE NMODES [--scale S] [--threads N] [--work DIR]
#        HALL = i | ii | iii, V = V1 | V3, CASE = S1 | S2+ | S2-.  Output: out/fe/<name>.json, ccx files in the work folder
#        (default out/fe/work/<name>; .frd files of k = 12 runs are several GB).  Peak memory of ccx (maximum resident
#        set size): about 15.7 GB (14.7 GiB) at k = 12 for (i), 9.2 GB (8.6 GiB) for (ii); 14.4 GB (13.4 GiB) at k = 8
#        for (iii). Run one ccx job at a time.
# Requires numpy, scipy and ccx 2.23 (environment variable CCX or 'ccx' on the PATH).
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, json, time, argparse
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import shellnet as S
import classify as CL

ROOT = os.path.dirname(HERE)
OUT = os.path.join(os.environ.get('APP_OUT', os.path.join(ROOT, 'out')), 'fe')

# per hall: input grid, model frame and supports (as in the runs that gave the reported numbers)
FE = {
    'i': dict(grid='hall_i_fe_V.npy', up=(0.0, 0.0, 1.0), along=(0.0, 1.0, 0.0), span=4.00, rise=2.84, outer_sign=+1,
              rail_dofs=[2, 3], spindle=(('vertex', (9, 0)), ('vertex', (9, 17))), spindle_L=4.00, latch=('h', 8, 11),
              latch_zone=0.15, radius=0.2, k=12),
    'ii': dict(grid='hall_ii_fe_V.npy', up=(0.0, 0.0, 1.0), along=(0.0, 1.0, 0.0), span=4.50, rise=2.54, outer_sign=-1,
               rail_dofs=[2, 3], spindle=(('vertex', (5, 0)), ('vertex', (5, 17))), spindle_L=4.50, latch=('v', 6, 4),
               latch_zone=None, radius=0.2, k=12),
    'iii': dict(grid='hall_iii_fe_V.npy', up=(0.0, 1.0, 0.0), along=(0.0, 0.0, 1.0), span=3.955, rise=2.40, outer_sign=+1,
                rail_dofs=[3, 2], spindle=(('vertices', [(8, 0), (10, 0)]), ('vertices', [(8, 25), (10, 25)])), spindle_L=None,
                latch=('v', 17, 20), latch_zone=None, radius=0.15, k=8),
}
CASES = ('S1', 'S2+', 'S2-')
N_FC, N_FC_SENS = 8, (5, 8, 11)


def load_pos(h):
    V = np.load(os.path.join(HERE, 'input', FE[h]['grid']))
    m, n = V.shape[0] - 1, V.shape[1] - 1
    return {(i, j): V[i, j].copy() for i in range(m + 1) for j in range(n + 1)}, m, n


def make_model(h, variant, k):
    """the shell model of hall h, variant V1 or V3, k x k elements per face"""
    c = FE[h]
    pos, m, n = load_pos(h)
    lat = [(c['latch'], 0.5)] if variant == 'V3' else []
    M = S.NetModel(pos, m, n, k=k, section=S.panel_section(), hinge_offset='outer', outer_sign=c['outer_sign'], up=c['up'],
                   latches=lat, name='hall_%s_%s' % (h, variant))
    f1 = S.lower_vertices(M, [(i, 0) for i in range(m + 1)])
    f2 = S.lower_vertices(M, [(i, n) for i in range(m + 1)])
    M.add_support(('vertices', f1), [1, 2, 3], 'foot1')
    if variant == 'V1':
        M.add_support(('vertices', f2), [1, 2, 3], 'foot2')
    else:
        M.add_support(('vertices', f2), list(c['rail_dofs']), 'foot2')        # rails: along and vertical fixed, free across
        ta, tb = c['spindle']
        M.add_spindle(ta, tb, E=210e9, d=0.04, L=c['spindle_L'])
        M.spindle_vertices = (ta[1], tb[1])
    M.feet = (f1, f2)
    return M


def loads(M, h):
    """self-weight G and the snow cases S1, S2+, S2- at s_k = 1 kN/m2 (forces per node)"""
    along = FE[h]['along']
    G = S.gravity_forces(M)
    s1 = S.snow_cylindrical(M, 1000.0, 'S1', along=along, grid=0.01)
    snow = {'S1': s1['forces']}
    for case, dr in (('S2+', +1), ('S2-', -1)):
        snow[case] = S.snow_cylindrical(M, 1000.0, 'S2', along=along, drift=dr, reading='triangles', rc=s1['rc'], env=s1['env'])['forces']
    info = dict(l_s=s1['l_s'], h_over_b=s1['h_over_b'], mu3=s1['mu3'], G_total_N=float(-np.dot(S.force_total(G), M.up)),
                totals_N={k_: float(-np.dot(S.force_total(F), M.up)) for k_, F in snow.items()})
    return G, snow, info


def uls(G, F, scale=1.0):
    return S.add_forces(G, F, factors=[1.35 * scale, 1.5 * scale])


def deck_names(h, variant, k, kind, case=None, nmodes=None, scale=1.0):
    base = 'hall_%s_%s_k%d' % (h, variant, k)
    if kind == 'static':
        return [base + '_static'] if variant == 'V1' else [base + '_static_c%d' % q for q in range(4)]
    tag = case.replace('+', 'p').replace('-', 'm')
    return [base + '_buckle_%s_n%d%s' % (tag, nmodes, '' if scale == 1.0 else '_s%g' % scale)]


def write_static_decks(M, G, snow, h, variant, k, outdir):
    """the static decks as the runs wrote them: one deck with four steps (V1), one deck per load case with a spindle (V3)"""
    cases = {'G': G, 'S1': snow['S1'], 'S2+': snow['S2+'], 'S2-': snow['S2-']}
    names = deck_names(h, variant, k, 'static')
    if variant == 'V1':
        M.write_inp(os.path.join(outdir, names[0] + '.inp'), analysis='static', cases=cases)
    else:
        for q, (nm, F) in enumerate(cases.items()):
            M.write_inp(os.path.join(outdir, names[q] + '.inp'), analysis='static', cases={nm: F})
    return names


def write_buckle_deck(M, G, snow, h, variant, k, case, nmodes, scale, outdir):
    name = deck_names(h, variant, k, 'buckle', case, nmodes, scale)[0]
    M.write_inp(os.path.join(outdir, name + '.inp'), analysis='buckle', cases=uls(G, snow[case], scale), nmodes=nmodes, buckle_accuracy=1e-7)
    return name


def dump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (set, tuple)): return list(o)
        return str(o)

    def fix(x):
        if isinstance(x, dict): return {(k_ if isinstance(k_, str) else str(k_)): fix(v) for k_, v in x.items()}
        if isinstance(x, (list, tuple)): return [fix(v) for v in x]
        return x
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh: json.dump(fix(obj), fh, indent=1, default=conv)


# ------------------------------------------------------------------------------------------------ static
def static(h, variant, k, threads, work):
    c = FE[h]; name = 'hall_%s_%s_k%d_static' % (h, variant, k)
    wd = work or os.path.join(OUT, 'work', name); os.makedirs(wd, exist_ok=True)
    t0 = time.time()
    M = make_model(h, variant, k)
    print('%s: %d S8R elements, %d nodes' % (name, len(M.elems), len(M.xyz)), flush=True)
    keep = {}
    orig = M.run_static

    def run_static(*a, **kw):
        r = orig(*a, **kw); keep['R'] = r; return r
    M.run_static = run_static
    out = S.analyse_hall(M, wd, c['span'], c['rise'], along=c['along'], s_k=1000.0, tag=name, buckle=False, threads=threads)
    res = dict(name=name, hall=h, variant=variant, k=k, elements=len(M.elems), nodes=len(M.xyz), mechanism=out['mechanism'],
               f1_Hz=out['f1_Hz'], snow=out.get('snow'), G_total_N=out.get('G_total'))
    if 'refused' in out:
        res['refused'] = out['refused']; dump(res, os.path.join(OUT, name + '.json')); print('REFUSED:', out['refused']); return res
    # the load combinations and the capacities by radius as the stored results give them (fem/results/hall_*/static_*.json)
    res['combos'] = out['combos']
    res['capacity_kN_per_m2_by_vertex_radius'] = out['capacity_kN_per_m2_by_vertex_radius']
    R = keep['R']
    lim = c['span'] / 200
    up = np.asarray(c['up'])
    nodes = sorted(nd for nd in M.xyz if nd in R.U['G'])
    UG = np.array([R.U['G'][nd] for nd in nodes])
    sls, sls_pt = {}, {}
    for cs in CASES:
        US = np.array([R.U[cs][nd] for nd in nodes]); T = UG + US
        a2, b2, ab = (UG * UG).sum(1), (US * US).sum(1), (UG * US).sum(1)
        disc = ab ** 2 - b2 * (a2 - lim ** 2)
        st = np.where(b2 > 0, (-ab + np.sqrt(np.maximum(disc, 0))) / np.where(b2 > 0, b2, 1), np.inf)
        st = np.where(a2 >= lim ** 2, 0.0, st)
        az, bz = UG @ up, US @ up
        sv = np.where(bz < 0, (lim + az) / np.where(bz < 0, -bz, 1), np.where(bz > 0, (lim - az) / np.where(bz > 0, bz, 1), np.inf))
        sv = np.where(np.abs(az) >= lim, 0.0, sv)
        sls[cs] = dict(u_tot_max_mm=1e3 * float(np.linalg.norm(T, axis=1).max()), u_vert_max_mm=1e3 * float(np.abs(T @ up).max()),
                       cap_SLS_total=float(st.min()), cap_SLS_vertical=float(sv.min()))
        sls_pt[cs] = {'SLS_total_span/200': nodes[int(np.argmin(st))], 'SLS_deflection_span/200': nodes[int(np.argmin(sv))]}
    res['G_only_u_tot_max_mm'] = 1e3 * float(np.linalg.norm(UG, axis=1).max())
    if variant == 'V3':
        f2 = M.feet[1]
        across = np.cross(up, np.asarray(c['along']))
        res['rail_slide_G_max_mm'] = 1e3 * max(abs(float(np.dot(R.U['G'][M.vertex_master(v)], across))) for v in f2)
    res['SLS'] = sls
    # stress capacities outside r of every net vertex (and of the latch for V3 where set), per snow case
    fc = S.face_and_core(M, R.S['G'])
    keys = fc['keys']
    dv = S.vertex_distance(M, [nd for nd, f in keys])
    XS = np.array([M.xyz[nd] for nd, f in keys])
    latch_zone = c['latch_zone'] if variant == 'V3' else None
    dl = np.linalg.norm(XS - M.xyz[M.latch_nodes[0]], axis=1) if (latch_zone and M.latch_nodes) else np.full(len(keys), np.inf)
    caps = {}
    for r_ in (0.1, 0.15, 0.2):
        caps['%g' % r_] = {}
        for cs in CASES:
            ex = (dv <= r_) | (dl <= (latch_zone or 0.0))
            cp = S.capacity(M, R, cs, span=c['span'], sk_ref=1.0, exclude=ex)['limits']
            cp['SLS_total_span/200'] = sls[cs]['cap_SLS_total']
            caps['%g' % r_][cs] = cp
    res['capacity_by_radius'] = caps
    r0 = '%g' % c['radius']
    allc = [(v_, l_, c_) for c_, d in caps[r0].items() for l_, v_ in d.items()]
    s_gov, lim_gov, case_gov = min(allc)
    # the place where each limit is reached first, at its governing snow case (stresses outside r0 and the latch zone)
    ex0 = (dv <= c['radius']) | (dl <= (latch_zone or 0.0))
    at = {}
    for l_ in caps[r0][CASES[0]]:
        s_, c_ = min((caps[r0][c_][l_], c_) for c_ in CASES)
        if l_ in sls_pt[c_]:
            pt, ratio = (sls_pt[c_][l_], None), None
        else:
            pt, ratio = stress_point(M, R, fc, c_, l_, s_, ex0)
        at[l_] = dict(s_k_max=s_, case=c_, limit_ratio=ratio, **place(M, h, pt[0], pt[1]))
    res['capacity'] = dict(radius=c['radius'], latch_zone=latch_zone, s_k_max=s_gov, governing=lim_gov, case=case_gov,
                           at=at[lim_gov])
    res['capacity_at_by_limit'] = at
    res['wrinkling_capacity_by_radius'] = {r_: min(caps[r_][c_]['ULS_face_compression_sigma_w'] for c_ in CASES) for r_ in caps}
    res['seconds'] = time.time() - t0
    dump(res, os.path.join(OUT, name + '.json'))
    print('%s: gate %s (SVD zero %d, FE zero %d), f1 %.3f Hz; snow totals %s N, G %.0f N' % (
        name, out['mechanism']['ok'], out['mechanism']['svd_zero'], out['mechanism']['fe_zero'], out['f1_Hz'],
        {k_: round(v_) for k_, v_ in out['snow']['totals'].items()}, out['G_total']))
    for c_ in CASES:
        print('  SLS %-4s u_vert %.2f mm, u_tot %.2f mm; capacities (r = %s m): %s' % (
            c_, sls[c_]['u_vert_max_mm'], sls[c_]['u_tot_max_mm'], r0, {k_: round(v_, 3) for k_, v_ in caps[r0][c_].items()}))
    for l_, a in at.items():
        print('  %-30s %.3f (%s): %s' % (l_, a['s_k_max'], a['case'], describe(a)))
    print('  capacity s_k,max = %.3f kN/m2 (%s, %s); wrinkling alone at r = 0.2 / 0.15 / 0.1 m: %.3f / %.3f / %.3f; %.0f s' % (
        res['capacity']['s_k_max'], res['capacity']['governing'], res['capacity']['case'], res['wrinkling_capacity_by_radius']['0.2'],
        res['wrinkling_capacity_by_radius']['0.15'], res['wrinkling_capacity_by_radius']['0.1'], time.time() - t0), flush=True)
    return res


# ------------------------------------------------------------------------------------------------ where a limit is reached
def _inplane_min(T, nrm):
    """smallest in-plane principal value of the tensors T (in-plane = normal to nrm)"""
    ev, evec = np.linalg.eigh(T)
    dots = np.abs(np.einsum('kij,ki->kj', evec, nrm))
    drop = np.argmax(dots, axis=1)
    return np.where(np.arange(3)[None, :] == drop[:, None], np.inf, ev).min(1)


def stress_point(M, R, fcG, case, limit, s, exclude):
    """the stress point (node, face) outside the excluded zone where the ULS stress of 1.35 G + 1.5 s S (S of the snow case)
    is largest relative to the limit, with that ratio: at s = s_k,max of the limit, the point where the limit is reached"""
    info = M.section.info
    keys = fcG['keys']
    if limit == 'ULS_core_shear':
        tG = S._core_vectors(M, R.S['G'], keys); tS = S._core_vectors(M, R.S[case], keys)
        val = np.linalg.norm(1.35 * tG + 1.5 * s * tS, axis=1) / info['tau_c_max']
    else:
        fcS = S.face_and_core(M, R.S[case])
        assert fcS['keys'] == keys
        val = np.zeros(len(keys))
        for side in ('sig_outer', 'sig_inner'):
            T = 1.35 * fcG[side] + 1.5 * s * fcS[side]
            if limit == 'ULS_face_compression_sigma_w':
                val = np.maximum(val, -_inplane_min(T, fcG['normals']) / info['sigma_w'])
            else:
                vm = np.sqrt(0.5 * ((T[:, 0, 0] - T[:, 1, 1]) ** 2 + (T[:, 1, 1] - T[:, 2, 2]) ** 2 + (T[:, 2, 2] - T[:, 0, 0]) ** 2)
                             + 3 * (T[:, 0, 1] ** 2 + T[:, 1, 2] ** 2 + T[:, 0, 2] ** 2))
                val = np.maximum(val, vm / (info['f_y'] / info['gamma_M']))
    val = np.where(np.asarray(exclude), -np.inf, val)
    q = int(np.argmax(val))
    return keys[q], float(val[q])


def _seg_dist(x, A, B):
    """distance of the point x from the segments [A_q, B_q]"""
    d = B - A
    s = np.clip(np.einsum('qi,qi->q', x - A, d) / np.einsum('qi,qi->q', d, d), 0.0, 1.0)
    return np.linalg.norm(A + s[:, None] * d - x, axis=1)


def place(M, h, node, face=None):
    """where a node of the model lies (outer skin): the nearest net vertex (grid indices of fem/input, distance, whether it is
    a support vertex or a spindle end), the distance to the nearest hinge line and to the nearest free edge of the net, and
    the position across the span (0 at foot 1, 1 at foot 2, from the mean positions of the supported vertices of each foot)"""
    c = FE[h]
    x = np.asarray(M.xyz[node], float)
    vk = list(M.pos); VP = np.array([M.pos[v] for v in vk])
    dv = np.linalg.norm(VP - x, axis=1); q = int(np.argmin(dv)); v = tuple(int(a) for a in vk[q])
    supV = set(M.feet[0]) | set(M.feet[1])
    spinV = set()
    for tgt in getattr(M, 'spindle_vertices', ()):        # one vertex (i, j) or a list of vertices per spindle end
        spinV |= {tuple(w) for w in tgt} if isinstance(tgt, list) else {tuple(tgt)}
    hinge = [M.edge_ends(e) for e in M.interior_edges() if M.conn[e] == 'hinge']
    free = [M.edge_ends(e) for e in M.boundary_edges()]
    dh = _seg_dist(x, np.array([M.pos[a] for a, b in hinge]), np.array([M.pos[b] for a, b in hinge]))
    df = _seg_dist(x, np.array([M.pos[a] for a, b in free]), np.array([M.pos[b] for a, b in free]))
    qh = int(np.argmin(dh))
    up = np.asarray(c['up']); across = np.cross(up, np.asarray(c['along']))
    p1 = np.mean([M.pos[w] for w in M.feet[0]], axis=0); p2 = np.mean([M.pos[w] for w in M.feet[1]], axis=0)
    u = float(np.dot(x - p1, across) / np.dot(p2 - p1, across))
    return dict(node=int(node), face=(list(face) if face is not None else None), xyz=[round(float(a), 4) for a in x],
                vertex=list(v), vertex_dist=float(dv[q]), vertex_is_support=bool(v in supV), vertex_is_spindle_end=bool(v in spinV),
                hinge_dist=float(dh[qh]), hinge=[list(hinge[qh][0]), list(hinge[qh][1])], free_edge_dist=float(df.min()),
                across=u, height_above_foot1=float(np.dot(x - p1, up)))


def describe(a):
    on = 'on the hinge line %s-%s' % tuple(tuple(p) for p in a['hinge']) if a['hinge_dist'] < 1e-6 else \
        ('on a free edge' if a['free_edge_dist'] < 1e-6 else '%.3f m from the nearest hinge line' % a['hinge_dist'])
    kind = 'support vertex' if a['vertex_is_support'] else ('spindle end' if a['vertex_is_spindle_end'] else 'vertex')
    return '%s, %.3f m from the %s %s; across the span %.2f (0 foot 1, 1 foot 2), %.2f m above foot 1' % (
        on, a['vertex_dist'], kind, tuple(a['vertex']), a['across'], a['height_above_foot1'])


# ------------------------------------------------------------------------------------------------ buckling
def buckle(h, variant, k, case, nmodes, scale, threads, work):
    import subprocess
    name = deck_names(h, variant, k, 'buckle', case, nmodes, scale)[0]
    wd = work or os.path.join(OUT, 'work', name); os.makedirs(wd, exist_ok=True)
    t0 = time.time()
    M = make_model(h, variant, k)
    mech = M.check_mechanisms(wd, 'mech', nmodes=6, threads=threads)
    print('%s: %d S8R elements; gate %s (SVD zero %d, FE zero %d, negative %d)' % (
        name, len(M.elems), mech['ok'], mech['svd']['n_zero'], mech['fe_zero'], mech['fe_negative']), flush=True)
    assert mech['ok']
    G, snow, info = loads(M, h)
    write_buckle_deck(M, G, snow, h, variant, k, case, nmodes, scale, wd)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), CCX_NPROC_EQUATION_SOLVER=str(threads))
    p = subprocess.run([S.CCX, '-i', name], cwd=wd, env=env, capture_output=True, text=True)
    open(os.path.join(wd, name + '.log'), 'w').write(p.stdout + p.stderr)
    assert p.returncode == 0 and '*ERROR' not in p.stdout + p.stderr, 'ccx failed, see %s.log' % name
    raw = [b[1] for b in S.parse_dat(os.path.join(wd, name + '.dat'))['buckling']]
    fac = [scale * x for x in raw]
    vals, blocks = CL.read_frd_disp(os.path.join(wd, name + '.frd'))
    bmap = CL.map_blocks(vals, raw)
    C = CL.Classifier(M)
    faces = M.faces(); fc_idx = []
    for f in faces:
        i, j = f
        cc = np.mean([M.pos[(i, j)], M.pos[(i + 1, j)], M.pos[(i, j + 1)], M.pos[(i + 1, j + 1)]], axis=0)
        nds = M.face_nodes(f)
        fc_idx.append(C.idx[nds[int(np.argmin([np.linalg.norm(M.xyz[nd] - cc) for nd in nds]))]])
    fc_idx = np.array(fc_idx)
    spin, sup = set(C.spin_v), set(C.sup_v)
    rows = []
    for q, lam in enumerate(fac):
        ids, U = blocks[bmap[q]]
        r, A = C.mode(ids, U)
        fcu = A[fc_idx]; pk = A.max(); n30 = int((fcu >= 0.3 * pk).sum())
        site = ('Lspin' if r['peak_vertex'] in spin else 'Lsup' if r['peak_vertex'] in sup else 'Lvtx') if r['peak_vertex_dist'] < 0.3 else 'Loth'
        rows.append(dict(mode=q + 1, factor=lam, classA=r['class'], A30=r['A30'], R_far=r['R_far05'], peak_vertex=r['peak_vertex'],
                         peak_vertex_dist=r['peak_vertex_dist'], fc_n30=n30, classB=('broad' if n30 >= N_FC else site),
                         classB_sens={n_: ('broad' if n30 >= n_ else site) for n_ in N_FC_SENS}))
    bA = next((rw for rw in rows if rw['classA'] == 'broad'), None)
    bB = next((rw for rw in rows if rw['classB'] == 'broad'), None)
    L = max(raw)
    res = dict(name=name, hall=h, variant=variant, k=k, case=case, nmodes=nmodes, scale=scale, factors=fac,
               window=[scale * L / (2 * L - 1), scale * L], negative=sum(1 for x in fac if x < 0), modes=rows,
               lowest_broad_A=bA, lowest_broad_B=bB, loads=info, seconds=time.time() - t0)
    dump(res, os.path.join(OUT, name + '.json'))
    print('  factors %s' % ' '.join('%.4f' % x for x in fac))
    for lab, b in (('A', bA), ('B', bB)):
        print('  lowest broad mode, rule %s: %s' % (lab, ('mode %d, lambda = %.4f, peak near %s' % (b['mode'], b['factor'], b['peak_vertex'])) if b else 'none'))
    print('  modes returned by ccx cover the factors %.4f .. %.4f (%.0f s)' % (res['window'][0], res['window'][1], time.time() - t0), flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=('static', 'buckle')); ap.add_argument('hall', choices=tuple(FE)); ap.add_argument('variant', choices=('V1', 'V3'))
    ap.add_argument('k', type=int); ap.add_argument('rest', nargs='*')
    ap.add_argument('--scale', type=float, default=1.0); ap.add_argument('--threads', type=int, default=4); ap.add_argument('--work', default=None)
    a = ap.parse_args()
    if a.kind == 'static':
        static(a.hall, a.variant, a.k, a.threads, a.work)
    else:
        buckle(a.hall, a.variant, a.k, a.rest[0], int(a.rest[1]), a.scale, a.threads, a.work)
