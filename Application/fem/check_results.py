# The numbers of the snow check, read from the stored results of the full CalculiX runs (fem/results), and the FE input
# geometry compared with the stored positions. Quantities, per hall and support variant (V1 both feet anchored; V3 foot 2 on
# rails, one spindle, one locked hinge): the capacity s_k,max (smallest over the limits and the three snow cases, stresses
# outside the radius r of every net vertex, r = 0.2 m for (i), (ii) and 0.15 m for (iii); for V3 of (i) also outside 0.15 m of
# the latch) with the governing limit and snow case; lambda_1 (the lowest broad mode by rule A); the lowest local mode at a
# support of (ii) and at the spindle end of (i); the wrinkling capacity of (i) with both feet anchored at r = 0.2, 0.15, 0.1
# m; the change of the capacities of (i) and (ii), both variants, from k = 8 to k = 12 (the stored k = 8 run of (ii) with
# spindle and lock gives no total-displacement capacity; it follows from the largest total displacements under G + S and 1.35
# G + 1.5 S, both at the same node of the rail foot); the spread of the rail foot of (ii) and (iii) under self-weight alone
# and with snow, and that the largest displacement under self-weight and snow is at the foot on the rails; the vertical
# displacement of (iii) at k = 4, 6, 8 and its largest change from k = 6 to 8 over the stored series; the overhang of the open
# ends of (i) past the last supported vertex; the place where the governing limit is reached (at the open end for (i) with
# both feet anchored, next to a support for (ii), on a hinge line for (iii), on a hinge line on the side of the foot on the
# rails for (i) with spindle and lock; for (ii) and (iii) from the fe.py reruns in results/reruns, which also equal the stored
# results); the capacity of (i) with spindle and lock when the latch zone is not left out (the factored self-weight alone
# exceeds the wrinkling limit there); the local buckling modes whose factors fall with the mesh (at a support of (ii), k = 4,
# 8, 12; at the end of the spindle of (i), k = 8, 12); elements per panel edge k and elements as the stored runs record them,
# element length and peak memory of ccx (the job ledgers give the maximum resident set size in GiB = 2^30 bytes; GB = 10^9
# bytes).
# usage: python3 fem/check_results.py           output: out/check_fe_results.json
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, json, glob, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import halls as HL
sys.path.insert(0, HERE)
import fe
import compare_static as CS

T0 = time.time()
R = os.path.join(HERE, 'results')
J = lambda *p: json.load(open(os.path.join(R, *p)))
CASES = ('S1', 'S2+', 'S2-')
vals, det = {}, {}
HL.header('Snow check: numbers from the stored results of the CalculiX runs')

# ------------------------------------------------------------------ FE input geometry = stored deployed positions
for h in HL.HALL_KEYS:
    Vfe = np.load(os.path.join(HERE, 'input', 'hall_%s_fe_V.npy' % h))
    Vd = HL.load_V(h, 'deployed')
    _, _, res = HL.kabsch(Vfe, Vd)
    vals['%s.fe.input_matches_positions_m' % h] = res
    print('hall (%s): FE input grid %s = stored deployed positions after a rigid motion, residual %.1e m' % (h, Vfe.shape, res))

# ------------------------------------------------------------------ hall (i), k = 12
C = J('hall_i', 'collected.json')
st, bk = C['static'], C['buckling']
v1 = st['V1_k12']['governing_r0p2']; v3 = st['V3_k12']['governing_all']['r0.2_l0.15']
vals['i.fe.capacity_V1'] = v1['s_k_max']; det['i.V1'] = (v1['limit'], v1['case'])
vals['i.fe.capacity_V3'] = v3['s_k_max']; det['i.V3'] = (v3['limit'], v3['case'])
vals['i.fe.governing_V1_is_sls_total_S1'] = bool(v1['limit'] == 'SLS total displacement' and v1['case'] == 'S1')
vals['i.fe.governing_V3_is_wrinkling_S2p'] = bool('wrinkling' in v3['limit'] and v3['case'] == 'S2+')
w = st['V1_k12']['wrinkling_capacity']
for r_ in ('0.2', '0.15', '0.1'):
    vals['i.fe.wrinkling_V1_r%s' % r_] = w['r%s_l0' % r_][0]
lam = {}
for V in ('V1', 'V3'):
    cs = bk['%s_k12' % V]['cases']
    best = min((cs[c]['lowest_broad_A']['factor'], c) for c in CASES)
    lam[V] = best; vals['i.fe.lambda1_%s' % V] = best[0]
loc = []
for f in glob.glob(os.path.join(R, 'hall_i', 'modes', 'hall_i_V3_k12_buckle_*_modes.json')):
    d = json.load(open(f))
    rows = d['rows'] if isinstance(d, dict) and 'rows' in d else d
    for rw in rows:
        if rw.get('class') == 'local-spindle': loc.append(rw['factor'])
vals['i.fe.local_spindle_min'] = min(loc)
k8 = {'V1': st['V1_k8']['governing_r0p2']['s_k_max'], 'V3': st['V3_k8']['governing_all']['r0.2_l0.15']['s_k_max']}
ch_i = {V: 100 * (vals['i.fe.capacity_%s' % V] / k8[V] - 1) for V in ('V1', 'V3')}
# overhang of the open ends past the last supported (lower foot) vertex, along the hall
Vfe = np.load(os.path.join(HERE, 'input', 'hall_i_fe_V.npy'))
up, al = np.array([0, 0, 1.0]), np.array([0, 1.0, 0])
sup = []
for j in (0, Vfe.shape[1] - 1):
    hgt = Vfe[:, j] @ up; lo_, hi_ = hgt.min(), hgt.max()
    sup += [Vfe[i, j] for i in range(Vfe.shape[0]) if hgt[i] <= lo_ + 0.5 * (hi_ - lo_)]
ys = Vfe.reshape(-1, 3) @ al; ysup = np.array(sup) @ al
over = max(ysup.min() - ys.min(), ys.max() - ysup.max())
vals['i.fe.overhang_m'] = float(over)
# where the governing limits are reached (stored point, rounded to 1 mm): the model node at that point and its place
def place_of_point(M, h, xyz):
    X = np.array([M.xyz[nd] for nd in M.xyz]); keys = list(M.xyz)
    q = int(np.argmin(np.linalg.norm(X - np.asarray(xyz), axis=1)))
    p = fe.place(M, h, keys[q]); p['dist_to_stored_point_m'] = float(np.linalg.norm(X[q] - np.asarray(xyz)))
    return p


Mi1, Mi3 = fe.make_model('i', 'V1', 12), fe.make_model('i', 'V3', 12)
p1 = place_of_point(Mi1, 'i', v1['at']['xyz']); p3 = place_of_point(Mi3, 'i', v3['at']['xyz'])
mi = Mi1.m
vals['i.fe.governing_V1_at_open_end'] = bool(p1['vertex'][0] in (0, mi) and p1['free_edge_dist'] < 1e-9 and p1['dist_to_stored_point_m'] < 1e-3)
vals['i.fe.governing_V3_on_hinge_line_rail_side'] = bool(p3['hinge_dist'] < 1e-9 and p3['vertex_dist'] > 0.1 and p3['across'] > 0.5
                                                         and p3['dist_to_stored_point_m'] < 1e-3)
det['i.V1_place'] = p1; det['i.V3_place'] = p3
# with spindle and lock, the latch zone (0.15 m) not left out: the factored self-weight alone exceeds the wrinkling limit
lz = st['V3_k12']['governing_r0p2_latch_not_excluded']
vals['i.fe.capacity_V3_latch_zone_included'] = lz['s_k_max']
vals['i.fe.latch_zone_governed_by_wrinkling'] = bool('wrinkling' in lz['limit'])
vals['i.fe.latch_zone_point_dist_m'] = lz['at']['dist_latch']
det['i.latch_zone_point'] = lz['at']
# local modes at the end of the spindle: lowest factor at k = 8 and k = 12
loc8 = []
for f in glob.glob(os.path.join(R, 'hall_i', 'modes', 'hall_i_V3_k8_buckle_*_modes.json')):
    d = json.load(open(f))
    rows = d['rows'] if isinstance(d, dict) and 'rows' in d else d
    loc8 += [rw['factor'] for rw in rows if rw.get('class') == 'local-spindle']
vals['i.fe.local_spindle_min_k8'] = min(loc8)
vals['i.fe.local_spindle_falls_with_k'] = bool(min(loc) < min(loc8))
print('\nhall (i), k = 12: capacity with both feet anchored %.4f (%s, %s), with spindle and lock %.4f (%s, %s; latch zone 0.15 m left out)'
      % (v1['s_k_max'], v1['limit'], v1['case'], v3['s_k_max'], v3['limit'], v3['case']))
print('  lambda_1 (rule A): V1 %.4f (%s), V3 %.4f (%s); lowest local mode at the spindle end %.4f; wrinkling alone (V1) at r = 0.2 / 0.15 / '
      '0.1 m: %.3f / %.3f / %.3f' % (lam['V1'][0], lam['V1'][1], lam['V3'][0], lam['V3'][1], min(loc), w['r0.2_l0'][0], w['r0.15_l0'][0], w['r0.1_l0'][0]))
print('  k = 8 -> 12: capacity V1 %.4f -> %.4f (%+.1f %%), V3 %.4f -> %.4f (%+.1f %%); open ends overhang the last supported vertex by up '
      'to %.3f m' % (k8['V1'], v1['s_k_max'], ch_i['V1'], k8['V3'], v3['s_k_max'], ch_i['V3'], over))
print('  both feet anchored, %s reached %s' % (v1['limit'], fe.describe(p1)))
print('  spindle and lock, %s reached %s' % (v3['limit'], fe.describe(p3)))
print('  spindle and lock with the latch zone included: capacity %.3f (%s, %s), at %.3f m from the latch'
      % (lz['s_k_max'], lz['limit'], lz['case'], lz['at']['dist_latch']))
print('  lowest local mode at the end of the spindle: %.4f at k = 8, %.4f at k = 12' % (min(loc8), min(loc)))

# ------------------------------------------------------------------ hall (ii), k = 12
st1, st3 = J('hall_ii', 'static_V1_k12.json'), J('hall_ii', 'static_V3_k12.json')
cu = J('hall_ii', 'sls_total_capacity_k12.json')


def cap_ii(stat, sls, r_):
    cands = []
    for c in CASES:
        cp = stat['capacity_kN_per_m2_by_vertex_radius'][c][r_]
        cands += [(v, k_, c) for k_, v in cp['limits'].items()]
        cands.append((sls[c]['s_max_total'] if 's_max_total' in sls[c] else sls[c]['s_k_max_total'], 'SLS_total_span/200', c))
    return min(cands)


c1, c3 = cap_ii(st1, cu['V1'], '0.2'), cap_ii(st3, cu['V3'], '0.2')
vals['ii.fe.capacity_V1'] = c1[0]; vals['ii.fe.capacity_V3'] = c3[0]
vals['ii.fe.governing_V1_is_wrinkling_S2p'] = bool('compression' in c1[1] and c1[2] == 'S2+')
vals['ii.fe.governing_V3_is_sls_total'] = bool(c3[1] == 'SLS_total_span/200')
B2 = J('hall_ii', 'buckle_k12_summary.json')
lam2 = {V: min((B2[V][c]['lowest_broad']['factor'], c) for c in CASES) for V in ('V1', 'V3')}
vals['ii.fe.lambda1_V1'] = lam2['V1'][0]; vals['ii.fe.lambda1_V3'] = lam2['V3'][0]
# lowest local mode at a point support, V1
locs = [(f, c, B2['V1'][c]['peak_vertex'][q]) for c in CASES for q, (f, cl) in enumerate(zip(B2['V1'][c]['factors'], B2['V1'][c]['classes'])) if cl == 'local-support']
ls = min(locs)
vals['ii.fe.local_support_min'] = ls[0]
vals['ii.fe.spread_G_mm'] = cu['V3']['G_only_utot_max_mm']
vals['ii.fe.spread_snow_mm'] = max(cu['V3'][c]['utot_at_sk1_mm'] for c in CASES)

Mii3 = fe.make_model('ii', 'V3', 12)
pl = [fe.place(Mii3, 'ii', st3['combos']['SLS_' + c]['u_tot_node']) for c in CASES]
vals['ii.fe.spread_snow_at_rail_foot'] = bool(all(q['vertex'] in [list(v) for v in Mii3.feet[1]] and q['vertex_dist'] < 0.05 for q in pl))
det['ii.spread_snow_place'] = pl
# the local mode at the support vertex (11, 0) under S2+, both feet anchored, at k = 4, 8, 12 (the same mode: MAC between
# successive meshes)
tr = J('hall_ii', 'mesh_trend', 'trend_V1_S2p_hall_ii_V1_k12_bk_S2p_n80_bk.json')
f_sup = {kk: tr[kk]['fac'][0] for kk in ('k4', 'k8', 'k12')}
same = all(tr[kk]['rows'][0]['vtx_is_support'] and tr[kk]['rows'][0]['peak_vertex'] == tr['k12']['rows'][0]['peak_vertex'] for kk in f_sup)
mac = (tr['match_12_8'][0]['mac'], tr['match_8_4'][0]['mac'])
for kk, v in f_sup.items(): vals['ii.fe.local_support_mode_%s' % kk] = v
vals['ii.fe.local_support_mode_falls_with_k'] = bool(same and min(mac) > 0.95 and f_sup['k4'] > f_sup['k8'] > f_sup['k12']
                                                     and tr['match_12_8'][0]['best'] == 1 and tr['match_8_4'][0]['best'] == 1)
det['ii.local_support_mode'] = dict(factors=f_sup, peak_vertex=tr['k12']['rows'][0]['peak_vertex'], mac_12_8_and_8_4=mac)
# where the capacity with both feet anchored is reached (fe.py rerun of the stored k = 12 run)
rr2 = J('reruns', 'hall_ii_V1_k12_static.json')
w2 = rr2['capacity']['at']
vals['ii.fe.rerun_V1_k12_max_rel_diff'] = CS.compare(rr2, st1, show=False)[0]
vals['ii.fe.governing_V1_next_to_support'] = bool(rr2['capacity']['governing'] == 'ULS_face_compression_sigma_w'
                                                  and w2['vertex_is_support'] and w2['vertex_dist'] < 0.3)
det['ii.V1_place'] = w2
st1k8 = J('hall_ii', 'static_V1_k8.json')
c1k8 = min((st1k8['capacity_kN_per_m2_by_vertex_radius'][c]['0.2']['s_k_max'], c) for c in CASES)
ch_ii = 100 * (c1[0] / c1k8[0] - 1)
# spindle and lock, k = 8: the stored run gives no capacity for the total displacement span/200. Its largest total
# displacements under G + S and 1.35 G + 1.5 S lie at the same node of the rail foot, where the displacements under G and S
# are collinear; there u_G = (1.5 u_SLS - u_ULS) / 0.15, and span/200 is reached at s_k = (span/200 - u_G) / (u_SLS - u_G).
# Checks: the same node in both combinations, the same u_G from the three snow cases, and, applied to the stored k = 12
# run, its stored u_G and capacity
def sls_total_from_combos(stat):
    out = {}
    for c in CASES:
        s_, u_ = stat['combos']['SLS_' + c], stat['combos']['ULS_' + c]
        us, uu, lim = s_['u_tot_max'], u_['u_tot_max'], stat['span'] / 200
        uG = (1.5 * us - uu) / 0.15
        out[c] = dict(s_max_total=(lim - uG) / (us - uG), uG_mm=1e3 * uG, same_node=s_['u_tot_node'] == u_['u_tot_node'])
    return out


st3k8 = J('hall_ii', 'static_V3_k8.json')
sl3 = {'k8': sls_total_from_combos(st3k8), 'k12': sls_total_from_combos(st3)}
c3k8 = cap_ii(st3k8, sl3['k8'], '0.2')
ch_ii3 = 100 * (c3[0] / c3k8[0] - 1)
uG_spread = max(max(d[c]['uG_mm'] for c in CASES) - min(d[c]['uG_mm'] for c in CASES) for d in sl3.values())
k12_diff = max(max(abs(sl3['k12'][c]['s_max_total'] / cu['V3'][c]['s_max_total'] - 1),
                   abs(sl3['k12'][c]['uG_mm'] / cu['V3'][c]['uG_at_node_mm'] - 1)) for c in CASES)
vals['ii.fe.capacity_V3_k8'] = c3k8[0]
vals['ii.fe.capacity_V3_k8_from_rail_foot_valid'] = bool(all(d[c]['same_node'] for d in sl3.values() for c in CASES)
                                                         and uG_spread < 1e-5 and k12_diff < 1e-6)
det['ii.capacity_V3_k8'] = dict(capacity=c3k8, uG_spread_mm=uG_spread, k12_max_rel_diff=k12_diff, by_case=sl3)
print('\nhall (ii), k = 12: capacity with both feet anchored %.4f (%s, %s), with spindle and lock %.4f (%s, %s)' % (c1 + c3))
print('  lambda_1 (rule A): V1 %.4f (%s), V3 %.4f (%s); lowest local mode at a point support %.4f (%s, near vertex %s)'
      % (lam2['V1'][0], lam2['V1'][1], lam2['V3'][0], lam2['V3'][1], ls[0], ls[1], ls[2]))
print('  rail foot (V3): %.1f mm under self-weight, up to %.1f mm under self-weight and snow; k = 8 -> 12: capacity V1 %.4f -> %.4f (%+.1f %%),'
      ' V3 %.4f -> %.4f (%+.1f %%)' % (vals['ii.fe.spread_G_mm'], vals['ii.fe.spread_snow_mm'], c1k8[0], c1[0], ch_ii, c3k8[0], c3[0], ch_ii3))
print('  V3, k = 8: capacity %.4f (%s, %s), u_G at the rail foot %.2f mm from each snow case (spread %.1e mm); at k = 12 the same rule '
      'gives the stored capacity and u_G to %.1e' % (c3k8 + (sl3['k8'][c3k8[2]]['uG_mm'], uG_spread, k12_diff)))
print('  rail foot (V3): the largest displacement under self-weight and snow lies %s' % fe.describe(pl[0]))
print('  local mode at the support vertex %s (S2+): %.3f / %.3f / %.3f at k = 4 / 8 / 12 (MAC %.3f, %.3f)'
      % (tuple(tr['k12']['rows'][0]['peak_vertex']), f_sup['k4'], f_sup['k8'], f_sup['k12'], mac[1], mac[0]))
print('  fe.py rerun, both feet anchored, k = 12: capacity %.4f (%s, %s), largest relative difference to the stored run %.1e; reached %s'
      % (rr2['capacity']['s_k_max'], rr2['capacity']['governing'], rr2['capacity']['case'], vals['ii.fe.rerun_V1_k12_max_rel_diff'], fe.describe(w2)))
det['fe.capacity_change_k8_k12_pct'] = {'i.V1': ch_i['V1'], 'i.V3': ch_i['V3'], 'ii.V1': ch_ii, 'ii.V3': ch_ii3}
vals['fe.capacity_change_k8_k12_max_pct'] = max(abs(x) for x in det['fe.capacity_change_k8_k12_pct'].values())

# ------------------------------------------------------------------ hall (iii), k = 8
s1, s3 = J('hall_iii', 'static_V1_k8.json'), J('hall_iii', 'static_V3_k8.json')
ct = J('hall_iii', 'sls_total_capacity_k8.json')
d1, d3 = cap_ii(s1, ct['hall_iii_V1_k8'], '0.15'), cap_ii(s3, ct['hall_iii_V3_k8'], '0.15')
vals['iii.fe.capacity_V1'] = d1[0]; vals['iii.fe.capacity_V3'] = d3[0]
vals['iii.fe.governing_V1_is_wrinkling_S2p'] = bool('compression' in d1[1] and d1[2] == 'S2+')
vals['iii.fe.governing_V3_is_sls_total'] = bool(d3[1] == 'SLS_total_span/200')
B3 = J('hall_iii', 'buckle_k8_results.json')
lam3 = {V: min((B3['hall_iii_%s_k8_bk_s0.5' % V][c]['lowest_broad_A']['factor'], c) for c in CASES) for V in ('V1', 'V3')}
vals['iii.fe.lambda1_V1'] = lam3['V1'][0]; vals['iii.fe.lambda1_V3'] = lam3['V3'][0]
vals['iii.fe.spread_G_mm'] = ct['hall_iii_V3_k8']['S1']['u_tot_G_mm']
vals['iii.fe.spread_snow_mm'] = max(1e3 * s3['combos']['SLS_' + c]['u_tot_max'] for c in CASES)
Miii3 = fe.make_model('iii', 'V3', 8)
pl3 = [fe.place(Miii3, 'iii', s3['combos']['SLS_' + c]['u_tot_node']) for c in CASES]
vals['iii.fe.spread_snow_at_rail_foot'] = bool(all(q['vertex'] in [list(v) for v in Miii3.feet[1]] and q['vertex_dist'] < 0.05 for q in pl3))
det['iii.spread_snow_place'] = pl3
vals['fe.spread_snow_max_mm'] = max(vals['ii.fe.spread_snow_mm'], vals['iii.fe.spread_snow_mm'])
rr3 = J('reruns', 'hall_iii_V1_k8_static.json')
w3 = rr3['capacity']['at']
vals['iii.fe.rerun_V1_k8_max_rel_diff'] = CS.compare(rr3, s1, show=False)[0]
vals['iii.fe.governing_V1_on_hinge_line'] = bool(rr3['capacity']['governing'] == 'ULS_face_compression_sigma_w'
                                                 and w3['hinge_dist'] < 1e-9 and w3['vertex_dist'] > 0.1)
det['iii.V1_place'] = w3
MS = J('hall_iii', 'mesh_series_k4_k6_k8.json')
ser = MS['V1_S2+']['u_v']
for kk in ('k4', 'k6', 'k8'): vals['iii.fe.u_vert_V1_S2p_%s_mm' % kk] = ser[kk]
vals['iii.fe.u_vert_change_k6_k8_max_pct'] = max(100 * abs(MS[s]['u_v']['k8'] / MS[s]['u_v']['k6'] - 1) for s in MS)
print('\nhall (iii), k = 8: capacity with both feet anchored %.4f (%s, %s), with spindle and lock %.4f (%s, %s)' % (d1 + d3))
print('  lambda_1 (rule A, load scaled by 0.5 so that factors down to about 0.26 are returned): V1 %.4f (%s), V3 %.4f (%s)'
      % (lam3['V1'][0], lam3['V1'][1], lam3['V3'][0], lam3['V3'][1]))
print('  rail foot (V3): %.1f mm under self-weight, up to %.1f mm under self-weight and snow, the largest of them lying %s'
      % (vals['iii.fe.spread_G_mm'], vals['iii.fe.spread_snow_mm'], fe.describe(pl3[0])))
print('  largest vertical displacement (V1, SLS S2+) at k = 4 / 6 / 8: %.2f / %.2f / %.2f mm; largest change from k = 6 to 8 over the %d series: %.2f %%'
      % (ser['k4'], ser['k6'], ser['k8'], len(MS), vals['iii.fe.u_vert_change_k6_k8_max_pct']))
print('  fe.py rerun, both feet anchored, k = 8: capacity %.4f (%s, %s), largest relative difference to the stored run %.1e; reached %s'
      % (rr3['capacity']['s_k_max'], rr3['capacity']['governing'], rr3['capacity']['case'], vals['iii.fe.rerun_V1_k8_max_rel_diff'], fe.describe(w3)))

# ------------------------------------------------------------------ all halls
vals['fe.all_lambda1_above_1'] = bool(min(vals[k_] for k_ in vals if '.lambda1_' in k_) > 1)
vals['fe.capacity_V1_min'] = min(vals['%s.fe.capacity_V1' % h] for h in HL.HALL_KEYS)
vals['fe.capacity_V1_max'] = max(vals['%s.fe.capacity_V1' % h] for h in HL.HALL_KEYS)
vals['iii.fe.u_vert_grows_with_k'] = bool(ser['k4'] < ser['k6'] < ser['k8'])
# elements per panel edge k and elements as the stored static runs of both variants record them (a list, and the element
# length nan, when the runs or the two directions of a face differ; the comparison then fails)
info = lambda r_: r_.get('model_info') or r_['meta']
for h, runs in (('i', (J('hall_i', 'static_V1_k12.json'), J('hall_i', 'static_V3_k12.json'))), ('ii', (st1, st3)), ('iii', (s1, s3))):
    ks = sorted({k_ for r_ in runs for k_ in r_['k']})
    ne = sorted({info(r_).get('elems', info(r_).get('n_elems')) for r_ in runs})
    vals['%s.fe.elements_per_panel_edge' % h] = ks[0] if len(ks) == 1 else ks
    vals['%s.fe.elements' % h] = ne[0] if len(ne) == 1 else ne
    vals['%s.fe.element_length_mm' % h] = 1e3 * HL.HALLS[h]['metres_per_unit'] / ks[0] if len(ks) == 1 else float('nan')
mem = []
for f in glob.glob(os.path.join(R, 'jobs', '*.tsv')):
    for line in open(f):
        p = line.rstrip('\n').split('\t')
        try: mem.append((float(p[5]), os.path.basename(f), p[2]))
        except (ValueError, IndexError): pass
GIB = 2 ** 30 / 1e9                     # the ledgers (column max_rss_GB) give the maximum resident set size in GiB
mi = max(m_ for m_ in mem if 'hall_iii' in m_[1])[0]
m1 = max(m_ for m_ in mem if 'hall_i.' in m_[1] or m_[1].endswith('hall_i.tsv'))[0]
vals['iii.fe.peak_memory_GiB'] = mi; vals['iii.fe.peak_memory_GB'] = mi * GIB
vals['i.fe.peak_memory_GiB'] = m1; vals['i.fe.peak_memory_GB'] = m1 * GIB
print('\nelements per panel edge k (stored runs): %s; elements: %s; element length on the typical edge: %s mm; peak memory of ccx: '
      '(iii) k = 8 %.2f GiB = %.2f GB, (i) k = 12 %.2f GiB = %.2f GB'
      % ({h: vals['%s.fe.elements_per_panel_edge' % h] for h in HL.HALL_KEYS}, {h: vals['%s.fe.elements' % h] for h in HL.HALL_KEYS},
         {h: round(vals['%s.fe.element_length_mm' % h], 1) for h in HL.HALL_KEYS}, mi, mi * GIB, m1, m1 * GIB))
print('all lambda_1 > 1: %s; capacities with both feet anchored %.3f .. %.3f; largest change k = 8 -> 12: %.1f %%'
      % (vals['fe.all_lambda1_above_1'], vals['fe.capacity_V1_min'], vals['fe.capacity_V1_max'], vals['fe.capacity_change_k8_k12_max_pct']))
# section and snow parameters used by the runs
sys.path.insert(0, HERE)
import shellnet as S
sec = S.panel_section().info
vals['fe.wrinkling_stress_MPa'] = sec['sigma_w'] / 1e6
vals['fe.mass_per_area_kg_m2'] = sec['mass_per_area']
vals['fe.hinge_offset_mm'] = 1e3 * S.panel_section().e_ref
vals['fe.face_stress_limit_MPa'] = sec['f_y'] / sec['gamma_M'] / 1e6
hb = {'i': st['V1_k12']['snow']['h_over_b'], 'ii': st1['snow']['h_over_b'], 'iii': s1['snow']['h_over_b']}
mu3 = {'i': st['V1_k12']['snow']['mu3'], 'ii': st1['snow']['mu3'], 'iii': s1['snow']['mu3']}
vals['fe.h_over_b_min'] = min(hb.values()); vals['fe.mu3_all_2'] = bool(all(abs(v - 2.0) < 1e-12 for v in mu3.values()))
print('section: sigma_w = 0.5 (E_f E_c G_c)^(1/3) = %.2f MPa, %.2f kg/m2, hinges %.0f mm from the mid-surface, face limit %.1f MPa; '
      'snow h/b = %s, mu_3 = %s' % (vals['fe.wrinkling_stress_MPa'], vals['fe.mass_per_area_kg_m2'], vals['fe.hinge_offset_mm'],
                                   vals['fe.face_stress_limit_MPa'], {h: round(v, 3) for h, v in hb.items()}, mu3))
HL.write_result('check_fe_results', dict(values=vals, details=det, seconds=time.time() - T0))
print('\nwritten out/check_fe_results.json (%.0f s)' % (time.time() - T0))
