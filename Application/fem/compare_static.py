# Compares a static run of fe.py (out/fe/hall_<h>_<V>_k<k>_static.json) with the stored result of the same hall, support
# variant and k (fem/results/hall_<h>/static_<V>_k<k>.json): the snow totals, the self-weight, the first frequency, the largest
# displacements and stresses of every load combination (SLS and ULS, three snow cases), the capacities per limit, snow case
# and radius r around the net vertices, and the displacements and capacities fe.py computes itself (r = 0.2 m; for V3 of (i)
# fe.py also leaves out the latch zone, so its own capacities are not compared there).
# usage: python3 fem/compare_static.py RUN.json [STORED.json] [--tol T]     (default tol 1e-6, relative)
# output: one line per quantity and the largest relative difference; exit code 1 if it exceeds tol
# Written with the assistance of the Claude models Opus 5.5, Fable 5, and Fable 5.1 (Anthropic); all code was checked by the authors.

import os, sys, json, math
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CASES = ('S1', 'S2+', 'S2-')
COMBO_QUANTITIES = (('u_vert_max', 'largest vertical displacement (m)'), ('u_tot_max', 'largest total displacement (m)'),
                    ('face_vm_max', 'largest face von Mises stress (Pa)'), ('face_comp_max', 'largest face compression (Pa)'),
                    ('core_tau_max', 'largest core shear stress (Pa)'))


def stored_for(run):
    """the stored result of the same hall, variant and k"""
    return os.path.join(HERE, 'results', 'hall_%s' % run['hall'], 'static_%s_k%d.json' % (run['variant'], run['k']))


def pairs(a, b):
    """(quantity, this run, stored) for every quantity both results give"""
    out = [('snow total %s (N)' % c, a['snow']['totals'][c], b['snow']['totals'][c]) for c in CASES]
    out.append(('self-weight (N)', a['G_total_N'], b['G_total']))
    out.append(('first frequency (Hz)', a['f1_Hz'], b['f1_Hz']))
    for cb in sorted(b['combos']):
        for q, nm in COMBO_QUANTITIES:
            if q in b['combos'][cb] and q in a.get('combos', {}).get(cb, {}):
                out.append(('%s %s' % (cb, nm), a['combos'][cb][q], b['combos'][cb][q]))
    ca, cb_ = a.get('capacity_kN_per_m2_by_vertex_radius', {}), b['capacity_kN_per_m2_by_vertex_radius']
    for c in CASES:
        for r in sorted(cb_[c], key=float):
            if r not in ca.get(c, {}): continue
            for lim, v in cb_[c][r]['limits'].items():
                out.append(('capacity %s %s, r = %s m (kN/m2)' % (c, lim, r), ca[c][r]['limits'][lim], v))
    # the values fe.py computes itself
    for c in CASES:
        out += [('fe.py: SLS %s largest vertical displacement (mm)' % c, a['SLS'][c]['u_vert_max_mm'], 1e3 * b['combos']['SLS_' + c]['u_vert_max']),
                ('fe.py: SLS %s largest total displacement (mm)' % c, a['SLS'][c]['u_tot_max_mm'], 1e3 * b['combos']['SLS_' + c]['u_tot_max'])]
        if not a['capacity'].get('latch_zone'):
            for lim, v in b['capacity_kN_per_m2_by_vertex_radius'][c]['0.2']['limits'].items():
                out.append(('fe.py: capacity %s %s, r = 0.2 m (kN/m2)' % (c, lim), a['capacity_by_radius']['0.2'][c][lim], v))
    return out


def rel(x, y):
    x, y = float(x), float(y)
    if x == y: return 0.0
    if math.isinf(x) or math.isinf(y): return math.inf
    return abs(x - y) / max(abs(y), 1e-30)


def compare(a, b, show=True):
    """largest relative difference over all pairs, and the pairs"""
    P = pairs(a, b)
    worst = 0.0
    for nm, x, y in P:
        d = rel(x, y); worst = max(worst, d)
        if show:
            print('  %-66s this run %16.9g   stored %16.9g   rel. diff %.1e' % (nm, x, y, d))
    return worst, P


if __name__ == '__main__':
    args = [x for x in sys.argv[1:] if not x.startswith('--')]
    tol = float(sys.argv[sys.argv.index('--tol') + 1]) if '--tol' in sys.argv else 1e-6
    if '--tol' in sys.argv: args.remove(sys.argv[sys.argv.index('--tol') + 1])
    run = json.load(open(args[0]))
    path = args[1] if len(args) > 1 else stored_for(run)
    print('this run: %s\nstored:   %s' % (os.path.basename(args[0]), os.path.relpath(path, os.path.dirname(HERE))))
    worst, P = compare(run, json.load(open(path)))
    print('%d quantities; largest relative difference %.1e (tolerance %.0e): %s' % (len(P), worst, tol, 'same' if worst <= tol else 'DIFFERENT'))
    sys.exit(0 if worst <= tol else 1)
